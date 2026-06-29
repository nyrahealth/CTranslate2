#include "ctranslate2/models/whisper.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/mean.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/slide.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    const Vocabulary& WhisperModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t WhisperModel::current_spec_revision() const {
      return 3;
    }

    void WhisperModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "<|endoftext|>";
      vocab_info.bos_token = "<|startoftranscript|>";
      vocab_info.eos_token = "<|endoftext|>";

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
    }

    bool WhisperModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool WhisperModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> WhisperModel::clone() const {
      return std::make_unique<WhisperModel>(*this);
    }


    std::unique_ptr<WhisperReplica> WhisperReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const WhisperModel*>(&model))
        throw std::invalid_argument("The model is not a Whisper model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const WhisperModel>(model_ptr);
      return std::make_unique<WhisperReplica>(concrete_model);
    }

    WhisperReplica::WhisperReplica(const std::shared_ptr<const WhisperModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::WhisperEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::WhisperDecoder>(*model, "decoder"))
    {
      const auto& vocabulary = model->get_vocabulary();
      _sot_id = vocabulary.bos_id();
      _eot_id = vocabulary.eos_id();
      _no_timestamps_id = vocabulary.to_id("<|notimestamps|>");
      _no_speech_id = vocabulary.to_id("<|nospeech|>");
      if (_no_speech_id == vocabulary.unk_id())
        _no_speech_id = vocabulary.to_id("<|nocaptions|>");
      _is_multilingual = vocabulary.to_id("") != vocabulary.unk_id();
      _n_mels = _encoder->input_size();
      // vocab: text tokens..., <|endoftext|>, <|startoftranscript|>,
      // lang tokens..., <|translate|>, <|transcribe|>, <|startoflm|>,
      // <|startofprev|>, <|nospeech|>, <|notimestamps|>, time tokens...
      _num_languages = _no_speech_id - _sot_id - 5;
    }

    StorageView WhisperReplica::encode(StorageView features, const bool to_cpu) {
      PROFILE("WhisperReplica::encode");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();
      features.move_to(device, dtype);

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);

      if (to_cpu) {
        if (device != Device::CPU)
          encoder_output = encoder_output.to(Device::CPU);
        return encoder_output;
      }

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);

      return encoder_output;
    }

    StorageView WhisperReplica::maybe_encode(StorageView features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      features.move_to(device, dtype);

      if (_encoder->is_encoded(features))
        return features;

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);
      return encoder_output;
    }

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(StorageView features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const WhisperOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(std::move(features), vocabulary.to_ids(prompts), options);
    }

    static std::vector<float> get_no_speech_probs_from_logits(const StorageView& logits,
                                                              const size_t no_speech_id) {
      const Device device = logits.device();
      const DataType dtype = logits.dtype();

      StorageView probs(dtype, device);
      ops::SoftMax()(logits, probs);

      StorageView gather_ids({probs.dim(0)}, int32_t(no_speech_id), device);
      StorageView no_speech_probs(dtype, device);
      ops::Gather(/*axis=*/1, /*batch_dims=*/1)(probs, gather_ids, no_speech_probs);

      if (no_speech_probs.dtype() != DataType::FLOAT32)
        no_speech_probs = no_speech_probs.to_float32();
      return no_speech_probs.to_vector<float>();
    }

    static size_t get_sot_index(const std::vector<size_t>& prompt, const size_t sot_id) {
      const auto sot_it = std::find(prompt.begin(), prompt.end(), sot_id);
      if (sot_it == prompt.end())
          throw std::invalid_argument("<|startoftranscript|> token was not found in the prompt");

      return std::distance(prompt.begin(), sot_it);
    }

    static size_t get_prompt_length(const std::vector<size_t>& prompt,
                                    const size_t sot_id,
                                    const size_t no_timestamps_id) {
      size_t index = get_sot_index(prompt, sot_id);
      while (index < prompt.size() && prompt[index] >= sot_id && prompt[index] <= no_timestamps_id)
        index++;
      return index;
    }

    static void check_prompts(const std::vector<std::vector<size_t>>& prompts,
                              const size_t sot_id,
                              const size_t no_timestamps_id,
                              size_t& sot_index,
                              size_t& prompt_length) {
      bool first = true;

      for (const auto& prompt : prompts) {
        const auto batch_sot_index = get_sot_index(prompt, sot_id);
        const auto batch_prompt_length = get_prompt_length(prompt, sot_id, no_timestamps_id);

        if (first) {
          sot_index = batch_sot_index;
          prompt_length = batch_prompt_length;
        } else if (batch_sot_index != sot_index) {
          throw std::invalid_argument("The generate method currently requires the "
                                      "<|startoftranscript|> token to be at the same position "
                                      "in all batches. To work around this limitation, "
                                      "simply adapt the number of previous text tokens in each "
                                      "batch.");
        } else if (batch_prompt_length != prompt_length) {
          throw std::invalid_argument("The generate method currently requires each batch to have "
                                      "the same number of task tokens after <|startoftranscript|>.");
        }

        first = false;
      }
    }

    class ApplyTimestampRules;

    class GetNoSpeechProbs : public LogitsProcessor {
    private:
      const size_t _no_speech_id;
      std::vector<float> _no_speech_probs;

    public:
      GetNoSpeechProbs(const size_t no_speech_id)
        : _no_speech_id(no_speech_id)
      {
      }

      const std::vector<float>& get_no_speech_probs() const {
        return _no_speech_probs;
      }

      bool apply_first() const override {
        return true;
      }

      void apply(dim_t step,
                 StorageView& logits,
                 DisableTokens&,
                 const StorageView&,
                 const std::vector<dim_t>& batch_offset,
                 const std::vector<std::vector<size_t>>*) override {
        if (step == 0) {
          const auto no_speech_probs = get_no_speech_probs_from_logits(logits, _no_speech_id);

          const size_t batch_size = batch_offset.size();
          const size_t beam_size = logits.dim(0) / batch_size;

          _no_speech_probs.reserve(batch_size);
          for (size_t i = 0; i < batch_size; ++i)
            _no_speech_probs.emplace_back(no_speech_probs[i * beam_size]);
        }
      }
    };

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(StorageView features,
                             const std::vector<std::vector<size_t>>& prompts,
                             const WhisperOptions& options) {
      PROFILE("WhisperReplica::generate");
      if (prompts.empty())
        return {};

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      size_t sot_index = 0;
      size_t prompt_length = 0;  // Length of the prompt before the text tokens.
      check_prompts(prompts, _sot_id, _no_timestamps_id, sot_index, prompt_length);

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const bool sot_is_start_token = (sot_index == prompt_length - 1);
      std::vector<std::vector<size_t>> start_tokens;
      std::vector<float> no_speech_probs;
      dim_t start_step = 0;

      if (prompt_length == 1) {
        start_tokens = prompts;

      } else {
        std::vector<std::vector<size_t>> prompt_tokens;
        prompt_tokens.reserve(prompts.size());
        start_tokens.reserve(prompts.size());
        for (const auto& prompt : prompts) {
          prompt_tokens.emplace_back(prompt.begin(), prompt.begin() + prompt_length - 1);
          start_tokens.emplace_back(prompt.begin() + prompt_length - 1, prompt.end());
        }

        const Device device = _decoder->device();
        const DataType dtype = _decoder->output_type();
        const StorageView inputs = layers::make_sequence_inputs(prompt_tokens, device);

        // Initialize the decoder state with the prompt.
        if (!options.return_no_speech_prob || sot_is_start_token)
          _decoder->forward_prompt(inputs, state);
        else {
          StorageView outputs(dtype, device);
          _decoder->forward_prompt(inputs, state, &outputs);

          // Get the probability of the no speech token at the start of transcript step.
          StorageView sot_index_batch({inputs.dim(0)}, int32_t(sot_index), device);
          StorageView logits(dtype, device);
          _decoder->compute_logits_for_steps(outputs, sot_index_batch, logits);
          no_speech_probs = get_no_speech_probs_from_logits(logits, _no_speech_id);
        }

        start_step = inputs.dim(1);
      }

      const dim_t total_max_length = options.max_length;

      DecodingOptions decoding_options;
      decoding_options.start_step = start_step;
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = std::min(total_max_length / 2, total_max_length - start_step);
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_logits_vocab = options.return_logits_vocab;
      decoding_options.include_eos_in_hypotheses = false;

      for (const auto& id : options.suppress_tokens) {
        if (id >= 0)
          decoding_options.disable_ids.push_back(id);
        else if (id == -1) {
          for (const auto& default_id : _model->config["suppress_ids"])
            decoding_options.disable_ids.push_back(default_id);
        }
      }

      if (options.suppress_blank) {
        for (const auto& id : _model->config["suppress_ids_begin"])
          decoding_options.disable_ids_begin.push_back(id);
      }

      std::shared_ptr<GetNoSpeechProbs> no_speech_probs_processor;
      if (options.return_no_speech_prob && sot_is_start_token) {
        // If SOT is the start token, we need to get the no speech prob in the first decoding loop.
        no_speech_probs_processor = std::make_shared<GetNoSpeechProbs>(_no_speech_id);
        decoding_options.logits_processors.emplace_back(no_speech_probs_processor);
      }

      if (prompts[0][prompt_length - 1] != _no_timestamps_id) {
        const size_t timestamp_begin_id = _no_timestamps_id + 1;
        const size_t timestamp_end_id = vocabulary.size() - 1;
        const size_t max_initial_timestamp_id = timestamp_begin_id + options.max_initial_timestamp_index;
        decoding_options.logits_processors.emplace_back(
          std::make_shared<ApplyTimestampRules>(_eot_id,
                                                _no_timestamps_id,
                                                timestamp_begin_id,
                                                timestamp_end_id,
                                                max_initial_timestamp_id));
      }

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_tokens,
                                                   {_eot_id},
                                                   decoding_options);

      if (no_speech_probs_processor)
        no_speech_probs = no_speech_probs_processor->get_no_speech_probs();

      std::vector<WhisperGenerationResult> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        WhisperGenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);
        final_result.logits = std::move(result.logits_vocab);
        if (options.return_no_speech_prob)
          final_result.no_speech_prob = no_speech_probs[i];

        final_results.emplace_back(std::move(final_result));
      }

      return final_results;
    }

    static void remove_padding(StorageView& x, dim_t axis, dim_t size) {
      const dim_t max_size = x.dim(axis);

      if (size < max_size) {
        StorageView content(x.dtype(), x.device());
        StorageView padding(x.dtype(), x.device());

        const ops::Split split_op(axis, {size, max_size - size});
        split_op(x, content, padding);

        x = std::move(content);
      }
    }

    static std::vector<std::vector<std::pair<dim_t, dim_t>>>
    compute_alignments(StorageView& attention_probs,
                       const std::vector<size_t>& start_sequence,
                       const std::vector<std::vector<size_t>>& text_tokens,
                       const dim_t median_filter_width) {
      const ops::MedianFilter median_filter_op(median_filter_width);
      const dim_t batch_size = attention_probs.dim(0);

      ops::LayerNorm(-2, 0)(attention_probs);

      StorageView median_filter(attention_probs.dtype(), attention_probs.device());
      median_filter_op(attention_probs, median_filter);

      StorageView weights(median_filter.dtype(), median_filter.device());
      ops::Mean(1)(median_filter, weights);

      // The remaining operations are not implemented on GPU, so move back to CPU.
      synchronize_stream(weights.device());
      weights.move_to(Device::CPU, DataType::FLOAT32);

      std::vector<std::vector<std::pair<dim_t, dim_t>>> alignments;
      alignments.reserve(batch_size);

      for (dim_t b = 0; b < batch_size; ++b) {
        const dim_t text_length = text_tokens[b].size();
        const dim_t sot_length = start_sequence.size();

        StorageView matrix(Shape{text_length + 1, weights.dim(2)});
        if (weights)
          matrix.view(weights.index<float>({b, sot_length, 0}), matrix.shape());

        alignments.emplace_back(negative_dtw(matrix));
      }

      return alignments;
    }

    std::vector<WhisperAlignmentResult>
    WhisperReplica::align(StorageView features,
                          const std::vector<size_t>& start_sequence,
                          const std::vector<std::vector<size_t>>& text_tokens,
                          std::vector<size_t> num_frames,
                          dim_t median_filter_width) {
      PROFILE("WhisperReplica::align");

      const dim_t batch_size = text_tokens.size();

      if (batch_size == 0)
        return {};

      if (num_frames.size() != size_t(batch_size))
        throw std::invalid_argument("Invalid batch size for argument num_frames");

      const auto alignment_heads = _model->config.find("alignment_heads");
      if (alignment_heads == _model->config.end())
        throw std::runtime_error("The model configuration does not contain the field "
                                 "'alignment_heads' which lists the cross-attention heads "
                                 "that are highly correlated to the word-level timing. "
                                 "Please reconvert this model with the current version "
                                 "of ctranslate2.");

      _decoder->set_alignment_heads(alignment_heads->get<std::vector<std::pair<dim_t, dim_t>>>());
      // ``align()`` expects raw pre-softmax cross-attention scores so it
      // can median-filter and softmax them itself.  Reset the flag here
      // in case a previous ``*_with_attention`` call set it to true.
      _decoder->set_return_normalized_attention(false);

      std::vector<std::vector<size_t>> input_tokens;
      std::vector<std::vector<size_t>> output_tokens;
      input_tokens.reserve(batch_size);
      output_tokens.reserve(batch_size);

      for (const auto& text_sequence : text_tokens) {
        std::vector<size_t> input_sequence = start_sequence;
        input_sequence.push_back(_no_timestamps_id);
        input_sequence.insert(input_sequence.end(), text_sequence.begin(), text_sequence.end());
        input_sequence.push_back(_eot_id);

        std::vector<size_t> output_sequence(input_sequence.begin() + 1, input_sequence.end());
        output_sequence.push_back(0);

        input_tokens.emplace_back(std::move(input_sequence));
        output_tokens.emplace_back(std::move(output_sequence));
      }

      const auto scoped_device_setter = _model->get_scoped_device_setter();

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", maybe_encode(std::move(features)));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const DataType dtype = _decoder->output_type();
      const Device device = _decoder->device();

      StorageView lengths(DataType::INT32, device);
      StorageView input_ids = layers::make_sequence_inputs(input_tokens,
                                                           device,
                                                           1,
                                                           &lengths);
      StorageView output_ids = layers::make_sequence_inputs(output_tokens, device);

      StorageView logits(dtype, device);
      StorageView attention_weights(dtype, device);
      (*_decoder)(input_ids, lengths, state, logits, &attention_weights);

      StorageView token_probs(dtype, device);

      {
        // Get the probabilities of the text tokens.
        StorageView text_vocab_size({logits.dim(0), logits.dim(1)}, int32_t(_eot_id), device);
        ops::SoftMax()(logits, text_vocab_size, logits);
        StorageView probs = std::move(logits);

        ops::Gather(/*axis=*/-1, /*batch_dims=*/2)(probs, output_ids, token_probs);
      }

      bool variable_num_frames = false;
      for (size_t& size : num_frames) {
        size /= 2;  // The second convolution layer uses a stride of 2.
        if (size != num_frames[0])
          variable_num_frames = true;
      }

      std::vector<std::vector<std::pair<dim_t, dim_t>>> alignments;

      if (variable_num_frames) {
        const StorageView frame_sizes({batch_size},
                                      std::vector<int32_t>(num_frames.begin(), num_frames.end()),
                                      device);
        const StorageView frame_sizes_mask(
          layers::MultiHeadAttention::prepare_length_mask(frame_sizes,
                                                          attention_weights.dim(1),
                                                          attention_weights.dim(2)));

        ops::SoftMax()(attention_weights, frame_sizes_mask, attention_weights);

        alignments.reserve(batch_size);

        for (dim_t b = 0; b < batch_size; ++b) {
          // Retrieve attention probs for batch and remove padding.
          StorageView batch_id({1}, int32_t(b), device);
          StorageView attention_probs(dtype, device);
          ops::Gather()(attention_weights, batch_id, attention_probs);

          remove_padding(attention_probs, 3, num_frames[b]);
          remove_padding(attention_probs, 2, input_tokens[b].size());

          alignments.emplace_back(compute_alignments(attention_probs,
                                                     start_sequence,
                                                     {text_tokens[b]},
                                                     median_filter_width)[0]);
        }

      } else {
        remove_padding(attention_weights, 3, num_frames[0]);
        ops::SoftMax()(attention_weights);

        alignments = compute_alignments(attention_weights,
                                        start_sequence,
                                        text_tokens,
                                        median_filter_width);
      }

      token_probs.move_to(Device::CPU, DataType::FLOAT32);

      std::vector<WhisperAlignmentResult> results;
      results.reserve(batch_size);

      for (dim_t b = 0; b < batch_size; ++b) {
        WhisperAlignmentResult result;

        const dim_t length = text_tokens[b].size();
        const dim_t offset = start_sequence.size();

        result.alignments = std::move(alignments[b]);

        for (dim_t t = 0; t < length; ++t)
          result.text_token_probs.emplace_back(token_probs.at<float>({b, offset + t}));

        results.emplace_back(std::move(result));
      }

      return results;
    }

    std::vector<std::vector<std::pair<std::string, float>>>
    WhisperReplica::detect_language(StorageView features) {
      if (!is_multilingual())
        throw std::runtime_error("detect_language can only be called on multilingual models");

      PROFILE("WhisperReplica::detect_language");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();
      const auto device = _model->device();

      const int32_t sot = vocabulary.bos_id();
      std::vector<int32_t> lang_ids;
      for (const auto& id : _model->config["lang_ids"])
        lang_ids.push_back(id);

      const dim_t batch_size = features.dim(0);
      const dim_t num_langs = lang_ids.size();

      StorageView start_ids({batch_size}, sot, device);
      StorageView score_ids({batch_size, num_langs}, DataType::INT32);
      for (dim_t i = 0; i < batch_size; ++i) {
        for (dim_t j = 0; j < num_langs; ++j)
          score_ids.at<int32_t>({i, j}) = lang_ids[j];
      }
      if (score_ids.device() != device)
        score_ids = score_ids.to(device);

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));

      StorageView logits(_decoder->output_type(), device);
      StorageView lang_probs(logits.dtype(), device);
      (*_decoder)(0, start_ids, state, &logits);
      ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, score_ids, lang_probs);
      ops::SoftMax()(lang_probs);

      if (lang_probs.dtype() != DataType::FLOAT32)
        lang_probs = lang_probs.to_float32();
      if (lang_probs.device() != Device::CPU)
        lang_probs = lang_probs.to(Device::CPU);

      std::vector<std::vector<std::pair<std::string, float>>> results;
      results.reserve(batch_size);

      for (dim_t i = 0; i < batch_size; ++i) {
        std::vector<std::pair<std::string, float>> result;
        result.reserve(num_langs);

        for (dim_t j = 0; j < num_langs; ++j) {
          const size_t lang_id = lang_ids[j];
          const float prob = lang_probs.at<float>({i, j});
          result.emplace_back(vocabulary.to_token(lang_id), prob);
        }

        std::sort(result.begin(), result.end(),
                  [](const std::pair<std::string, float>& a,
                     const std::pair<std::string, float>& b) {
                    return a.second > b.second;
                  });

        results.emplace_back(std::move(result));
      }

      return results;
    }


    bool Whisper::is_multilingual() const {
      const auto& replica = get_first_replica();
      return replica.is_multilingual();
    }

    size_t Whisper::n_mels() const {
      const auto& replica = get_first_replica();
      return replica.n_mels();
    }

    size_t Whisper::num_languages() const {
      const auto& replica = get_first_replica();
      return replica.num_languages();
    }

    std::future<StorageView> Whisper::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>(
        [features = features.sync_copy(), to_cpu](WhisperReplica& replica) mutable {
          return replica.encode(std::move(features), to_cpu);
        });
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(const StorageView& features,
                      std::vector<std::vector<std::string>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = features.sync_copy(),
         prompts = std::move(prompts),
         options = std::move(options)]
        (WhisperReplica& replica) mutable {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(const StorageView& features,
                      std::vector<std::vector<size_t>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = features.sync_copy(),
         prompts = std::move(prompts),
         options = std::move(options)]
        (WhisperReplica& replica) mutable {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<std::vector<std::pair<std::string, float>>>>
    Whisper::detect_language(const StorageView& features) {
      const size_t batch_size = features.dim(0);
      return post_batch<std::vector<std::pair<std::string, float>>>(
        [features = features.sync_copy()](WhisperReplica& replica) mutable {
          return replica.detect_language(std::move(features));
        },
        batch_size);
    }

    std::vector<std::future<WhisperAlignmentResult>>
    Whisper::align(const StorageView& features,
                   std::vector<size_t> start_sequence,
                   std::vector<std::vector<size_t>> text_tokens,
                   std::vector<size_t> num_frames,
                   dim_t median_filter_width) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperAlignmentResult>(
        [features = features.sync_copy(),
         start_sequence = std::move(start_sequence),
         text_tokens = std::move(text_tokens),
         num_frames = std::move(num_frames),
         median_filter_width]
        (WhisperReplica& replica) mutable {
          return replica.align(std::move(features),
                               start_sequence,
                               text_tokens,
                               std::move(num_frames),
                               median_filter_width);
        },
        batch_size);
    }


    std::pair<WhisperDecoderState, StorageView>
    WhisperReplica::prefill(StorageView features,
                            const std::vector<size_t>& prompt) {
      PROFILE("WhisperReplica::prefill");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));
      _decoder->update_output_layer(_model->preferred_size_multiple());

      if (prompt.size() > 1) {
        std::vector<std::vector<size_t>> prompt_batch = {
          std::vector<size_t>(prompt.begin(), prompt.end() - 1)
        };
        const StorageView inputs = layers::make_sequence_inputs(prompt_batch, device);
        _decoder->forward_prompt(inputs, state);
      }

      const dim_t start_step = prompt.size() > 1
          ? static_cast<dim_t>(prompt.size()) - 1
          : 0;

      StorageView last_id({1}, int32_t(prompt.back()), device);
      StorageView logits(_decoder->output_type(), device);
      (*_decoder)(start_step, last_id, state, &logits);

      WhisperDecoderState wds;
      wds.state = std::move(state);
      wds.current_step = static_cast<dim_t>(prompt.size());

      return {std::move(wds), std::move(logits)};
    }

    StorageView
    WhisperReplica::forward_step(WhisperDecoderState& wds,
                                 size_t token_id) {
      PROFILE("WhisperReplica::forward_step");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      StorageView ids({1}, int32_t(token_id), device);
      StorageView logits(_decoder->output_type(), device);
      (*_decoder)(wds.current_step, ids, wds.state, &logits);
      wds.current_step++;

      return logits;
    }

    StorageView
    WhisperReplica::forward_batch(WhisperDecoderState& wds,
                                  const std::vector<size_t>& token_ids) {
      PROFILE("WhisperReplica::forward_batch");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      std::vector<std::vector<size_t>> batch = {token_ids};
      const StorageView ids = layers::make_sequence_inputs(batch, device);

      StorageView logits(_decoder->output_type(), device);
      _decoder->forward_with_logits(ids, wds.current_step, wds.state, logits);
      wds.current_step += static_cast<dim_t>(token_ids.size());

      return logits;
    }

    void
    WhisperDecoderState::truncate_to_step(dim_t target_step) {
      if (target_step >= current_step)
        return;
      const dim_t steps_to_drop = current_step - target_step;
      for (auto& [key, tensor] : state) {
        if (key.compare(0, 10, "self_keys_") == 0
            || key.compare(0, 12, "self_values_") == 0) {
          const dim_t time_dim = 2;
          const dim_t current_len = tensor.dim(time_dim);
          const dim_t keep_len = current_len - steps_to_drop;
          if (keep_len > 0 && keep_len < current_len) {
            StorageView tmp(tensor.dtype(), tensor.device());
            ops::Slide(time_dim, 0, keep_len)(tensor, tmp);
            tensor = std::move(tmp);
          }
        }
      }
      // Keep the cross-attention buffer aligned with the kept tokens.
      // collected_attention[k] corresponds to predicting the (start+k)-th
      // newly-generated token, so we resize from the right.
      if (!collected_attention.empty()) {
        const dim_t kept = static_cast<dim_t>(collected_attention.size())
                           - steps_to_drop;
        if (kept <= 0)
          collected_attention.clear();
        else
          collected_attention.resize(static_cast<size_t>(kept));
      }
      current_step = target_step;
    }

    size_t
    WhisperReplica::forward_step_greedy(WhisperDecoderState& wds,
                                        size_t token_id,
                                        const std::vector<size_t>& suppress_tokens) {
      PROFILE("WhisperReplica::forward_step_greedy");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      StorageView ids({1}, int32_t(token_id), device);
      StorageView logits(_decoder->output_type(), device);
      (*_decoder)(wds.current_step, ids, wds.state, &logits);
      wds.current_step++;

      // Mask suppressed tokens on the device before argmax so suppression
      // stays on the fast greedy path (no full-vocab transfer to Python).
      if (!suppress_tokens.empty()) {
        DisableTokens disable(logits);
        for (const size_t tid : suppress_tokens)
          disable.add(static_cast<dim_t>(tid));
        disable.apply();
      }

      StorageView best_ids(DataType::INT32, device);
      StorageView best_scores(logits.dtype(), device);
      ops::TopK(1)(logits, best_scores, best_ids);

      StorageView best_ids_cpu(DataType::INT32);
      best_ids_cpu.copy_from(best_ids);
      return static_cast<size_t>(best_ids_cpu.scalar_at<int32_t>({0, 0}));
    }

    std::vector<size_t>
    WhisperReplica::forward_batch_greedy(WhisperDecoderState& wds,
                                         const std::vector<size_t>& token_ids,
                                         const std::vector<size_t>& suppress_tokens) {
      PROFILE("WhisperReplica::forward_batch_greedy");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      std::vector<std::vector<size_t>> batch = {token_ids};
      const StorageView ids = layers::make_sequence_inputs(batch, device);

      StorageView logits(_decoder->output_type(), device);
      _decoder->forward_with_logits(ids, wds.current_step, wds.state, logits);
      wds.current_step += static_cast<dim_t>(token_ids.size());

      // Mask suppressed tokens at every position on the device before the
      // per-position argmax.  ``forward_with_logits`` returns logits shaped
      // ``[1, seq, vocab]``; DisableTokens treats dim(0) as batch and
      // dim(1) as vocab, so reshape to ``[seq, vocab]`` first.  ``add(tid)``
      // then disables the token for all positions, matching a
      // non-speculative suppressed decode.
      if (!suppress_tokens.empty()) {
        const dim_t vocab = logits.dim(-1);
        const Shape saved_shape = logits.shape();
        logits.reshape({logits.size() / vocab, vocab});
        DisableTokens disable(logits);
        for (const size_t tid : suppress_tokens)
          disable.add(static_cast<dim_t>(tid));
        disable.apply();
        logits.reshape(saved_shape);
      }

      StorageView best_ids(DataType::INT32, device);
      StorageView best_scores(logits.dtype(), device);
      ops::TopK(1)(logits, best_scores, best_ids);

      StorageView best_ids_cpu(DataType::INT32);
      best_ids_cpu.copy_from(best_ids);

      const dim_t n = static_cast<dim_t>(token_ids.size());
      std::vector<size_t> result(n);
      for (dim_t i = 0; i < n; ++i)
        result[i] = static_cast<size_t>(best_ids_cpu.at<int32_t>(i));
      return result;
    }

    // -----------------------------------------------------------------------
    // Cross-attention extraction for word timings (used by CrisperWhisper).
    //
    // ``set_alignment_heads`` selects a fixed list of (layer, head) pairs
    // that the decoder will gather and concatenate into the ``attention``
    // output of every subsequent decode step.  ``*_with_attention``
    // variants of ``prefill`` / ``forward_step`` / ``forward_step_greedy``
    // capture that per-step row into ``state.collected_attention`` so the
    // caller can do a single bulk GPU->CPU transfer at the end of a
    // generation run.
    //
    // The selected heads are post-softmax cross-attention probabilities
    // (rows sum to 1 over encoder frames), which is what
    // CrisperWhisper-style Viterbi/peak-mass timing extractors expect.
    // -----------------------------------------------------------------------

    void
    WhisperReplica::set_alignment_heads(
        const std::vector<std::pair<dim_t, dim_t>>& heads) {
      _decoder->set_alignment_heads(heads);
      // Empty list disables collection; in that case keep the decoder in
      // its default raw-attention mode (used by ``align()``).  Otherwise
      // request post-softmax rows so timing extractors get probability
      // distributions over encoder frames.
      _decoder->set_return_normalized_attention(!heads.empty());
    }

    static void require_alignment_heads_configured(const layers::WhisperDecoder& dec) {
      // ``decode()`` only populates the ``attention`` output when at
      // least one ``(layer, head)`` pair is configured.  Catch the
      // common mistake of calling ``*_with_attention`` before
      // ``set_alignment_heads`` and give a useful error message.
      if (!dec.return_normalized_attention())
        throw std::runtime_error(
            "WhisperReplica: no alignment heads configured for "
            "*_with_attention(); call set_alignment_heads([...]) first.");
    }

    std::tuple<WhisperDecoderState, StorageView, StorageView>
    WhisperReplica::prefill_with_attention(StorageView features,
                                           const std::vector<size_t>& prompt) {
      PROFILE("WhisperReplica::prefill_with_attention");
      require_alignment_heads_configured(*_decoder);

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));
      _decoder->update_output_layer(_model->preferred_size_multiple());

      if (prompt.size() > 1) {
        std::vector<std::vector<size_t>> prompt_batch = {
          std::vector<size_t>(prompt.begin(), prompt.end() - 1)
        };
        const StorageView inputs = layers::make_sequence_inputs(prompt_batch, device);
        // We discard the prompt's per-step attention here: those rows
        // correspond to prompt tokens, not to generated tokens.
        _decoder->forward_prompt(inputs, state);
      }

      const dim_t start_step = prompt.size() > 1
          ? static_cast<dim_t>(prompt.size()) - 1
          : 0;

      StorageView last_id({1}, int32_t(prompt.back()), device);
      StorageView logits(_decoder->output_type(), device);
      StorageView attention(_decoder->output_type(), device);
      (*_decoder)(start_step, last_id, state, &logits, &attention);

      WhisperDecoderState wds;
      wds.state = std::move(state);
      wds.current_step = static_cast<dim_t>(prompt.size());
      // The attention row at this step corresponds to predicting the
      // first new (generated) token: append it to the buffer.
      wds.collected_attention.emplace_back(attention);

      return {std::move(wds), std::move(logits), std::move(attention)};
    }

    std::pair<StorageView, StorageView>
    WhisperReplica::forward_step_with_attention(WhisperDecoderState& wds,
                                                size_t token_id) {
      PROFILE("WhisperReplica::forward_step_with_attention");
      require_alignment_heads_configured(*_decoder);

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      StorageView ids({1}, int32_t(token_id), device);
      StorageView logits(_decoder->output_type(), device);
      StorageView attention(_decoder->output_type(), device);
      (*_decoder)(wds.current_step, ids, wds.state, &logits, &attention);
      wds.current_step++;
      wds.collected_attention.emplace_back(attention);

      return {std::move(logits), std::move(attention)};
    }

    std::pair<size_t, StorageView>
    WhisperReplica::forward_step_greedy_with_attention(WhisperDecoderState& wds,
                                                       size_t token_id) {
      PROFILE("WhisperReplica::forward_step_greedy_with_attention");
      require_alignment_heads_configured(*_decoder);

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      StorageView ids({1}, int32_t(token_id), device);
      StorageView logits(_decoder->output_type(), device);
      StorageView attention(_decoder->output_type(), device);
      (*_decoder)(wds.current_step, ids, wds.state, &logits, &attention);
      wds.current_step++;
      wds.collected_attention.emplace_back(attention);

      StorageView best_ids(DataType::INT32, device);
      StorageView best_scores(logits.dtype(), device);
      ops::TopK(1)(logits, best_scores, best_ids);

      StorageView best_ids_cpu(DataType::INT32);
      best_ids_cpu.copy_from(best_ids);
      const size_t picked = static_cast<size_t>(best_ids_cpu.scalar_at<int32_t>({0, 0}));
      return {picked, std::move(attention)};
    }

    std::pair<WhisperDecoderState, std::vector<size_t>>
    WhisperReplica::generate_greedy_with_attention(
        StorageView features,
        const std::vector<size_t>& prompt,
        size_t max_new_tokens,
        size_t eot_id,
        const std::vector<size_t>& suppress_tokens,
        const std::vector<size_t>& ban_first_tokens) {
      PROFILE("WhisperReplica::generate_greedy_with_attention");
      require_alignment_heads_configured(*_decoder);

      if (prompt.empty())
        throw std::invalid_argument(
            "generate_greedy_with_attention: prompt must be non-empty");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));
      _decoder->update_output_layer(_model->preferred_size_multiple());

      // Prefill all but the last prompt token; the last one drives the
      // first generation step (where we want attention + logits).
      if (prompt.size() > 1) {
        std::vector<std::vector<size_t>> prompt_batch = {
          std::vector<size_t>(prompt.begin(), prompt.end() - 1)
        };
        const StorageView inputs = layers::make_sequence_inputs(prompt_batch, device);
        _decoder->forward_prompt(inputs, state);
      }

      WhisperDecoderState wds;
      wds.state = std::move(state);
      wds.current_step = prompt.size() > 1
          ? static_cast<dim_t>(prompt.size()) - 1
          : 0;
      wds.collected_attention.reserve(max_new_tokens);

      std::vector<size_t> generated;
      generated.reserve(max_new_tokens);

      size_t cur_token = prompt.back();

      for (size_t step = 0; step < max_new_tokens; ++step) {
        StorageView ids({1}, int32_t(cur_token), device);
        StorageView logits(_decoder->output_type(), device);
        StorageView attention(_decoder->output_type(), device);
        (*_decoder)(wds.current_step, ids, wds.state, &logits, &attention);
        wds.current_step++;
        wds.collected_attention.emplace_back(std::move(attention));

        // Mask suppressed tokens (and the loop-starting "ban" tokens on
        // step 0 only) directly on the device before argmax.
        {
          DisableTokens disable(logits);
          for (const size_t tid : suppress_tokens)
            disable.add(static_cast<dim_t>(tid));
          if (step == 0) {
            for (const size_t tid : ban_first_tokens)
              disable.add(static_cast<dim_t>(tid));
          }
          disable.apply();
        }

        StorageView best_ids(DataType::INT32, device);
        StorageView best_scores(logits.dtype(), device);
        ops::TopK(1)(logits, best_scores, best_ids);

        StorageView best_ids_cpu(DataType::INT32);
        best_ids_cpu.copy_from(best_ids);
        const size_t picked = static_cast<size_t>(
            best_ids_cpu.scalar_at<int32_t>({0, 0}));

        generated.push_back(picked);
        if (picked == eot_id)
          break;
        cur_token = picked;
      }

      return {std::move(wds), std::move(generated)};
    }

    // --- Helpers for generate_speculative -------------------------------

    namespace {
      // ``table[id]`` with empty-table == identity and out-of-range == -1.
      inline int32_t map_token(const std::vector<int32_t>& table, size_t id) {
        if (table.empty())
          return static_cast<int32_t>(id);
        if (id < table.size())
          return table[id];
        return -1;
      }

      // On-device argmax over a ``[1, vocab]`` (or ``[.., vocab]``) logits
      // tensor with optional token suppression, mirroring
      // ``forward_step_greedy``'s tail so the prefill / verify argmaxes
      // stay on the GPU.
      inline size_t argmax_suppressed_device(StorageView& logits,
                                             const std::vector<size_t>& suppress_tokens) {
        const Device device = logits.device();
        if (!suppress_tokens.empty()) {
          DisableTokens disable(logits);
          for (const size_t tid : suppress_tokens)
            disable.add(static_cast<dim_t>(tid));
          disable.apply();
        }
        StorageView best_ids(DataType::INT32, device);
        StorageView best_scores(logits.dtype(), device);
        ops::TopK(1)(logits, best_scores, best_ids);
        StorageView best_ids_cpu(DataType::INT32);
        best_ids_cpu.copy_from(best_ids);
        return static_cast<size_t>(best_ids_cpu.scalar_at<int32_t>({0, 0}));
      }
    }

    std::pair<std::vector<std::vector<size_t>>, std::vector<StorageView>>
    WhisperReplica::generate_dual_greedy(
        StorageView features,
        const std::vector<std::vector<size_t>>& prompts,
        size_t max_new_tokens,
        size_t eot_id,
        const std::vector<size_t>& suppress_tokens,
        bool want_attention) {
      PROFILE("WhisperReplica::generate_dual_greedy");
      if (want_attention)
        require_alignment_heads_configured(*_decoder);

      const size_t n_rows = prompts.size();
      if (n_rows == 0)
        return {};
      for (const auto& p : prompts)
        if (p.empty())
          throw std::invalid_argument(
              "generate_dual_greedy: every prompt must be non-empty");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      const DataType dtype = _decoder->output_type();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      // Encode the shared audio once; reused for every row.
      StorageView memory = maybe_encode(std::move(features));  // [1, F, dim]

      // Head-average a single-position attention tensor ``[rows, heads, F]``
      // to a CPU float32 ``[rows, F]``.
      auto attn_to_cpu_rows = [&](const StorageView& attention) -> StorageView {
        StorageView reduced(attention.dtype(), device);
        ops::Mean(1)(attention, reduced);              // [rows, 1, F]
        StorageView fp32 = reduced.to(DataType::FLOAT32);
        StorageView cpu = fp32.to(Device::CPU);        // [rows, 1, F]
        if (cpu.rank() == 3 && cpu.dim(1) == 1)
          cpu.squeeze(1);                              // [rows, F]
        return cpu;
      };

      // Per-row results (indexed by original prompt order).
      std::vector<std::vector<size_t>> gen_ids(n_rows);
      std::vector<std::vector<float>> attn_flat(n_rows);
      std::vector<bool> finished(n_rows, false);
      dim_t F_enc = 0;

      size_t target_len = 0;
      for (const auto& p : prompts)
        target_len = std::max(target_len, p.size());
      const dim_t target_step = static_cast<dim_t>(target_len) - 1;

      // Catch-up: decode every row on its own -- exactly as a single-mode
      // decode would (prompt prefill + incremental greedy steps) -- until its
      // KV cache reaches ``target_step`` positions.  Rows already at
      // ``target_len`` need zero extra steps.  Keeping each row's
      // incrementally-built KV (rather than re-prefilling an extended prompt)
      // is what makes the batched result bit-identical to a per-mode decode:
      // the catch-up tokens' cache is produced by the same forward_step path
      // the reference uses, so no forward_prompt-vs-incremental drift creeps
      // in.  A row that emits EOT during catch-up is finished and excluded.
      std::vector<size_t> active;                          // original indices
      std::vector<layers::DecoderState> states;           // per active row
      std::vector<size_t> next_feed;                       // token at target_step

      for (size_t i = 0; i < n_rows; ++i) {
        const auto& prompt = prompts[i];

        layers::DecoderState state = _decoder->initial_state();
        state.emplace("memory", StorageView(memory));      // own copy of memory
        dim_t cur_step = 0;
        if (prompt.size() > 1) {
          std::vector<std::vector<size_t>> pb = {
            std::vector<size_t>(prompt.begin(), prompt.end() - 1)
          };
          const StorageView inputs = layers::make_sequence_inputs(pb, device);
          _decoder->forward_prompt(inputs, state);
          cur_step = static_cast<dim_t>(prompt.size()) - 1;
        }

        size_t cur_token = prompt.back();
        bool hit_eot = false;
        while (cur_step < target_step) {
          StorageView ids({1}, int32_t(cur_token), device);
          StorageView logits(dtype, device);
          StorageView attention(dtype, device);
          (*_decoder)(cur_step, ids, state, &logits,
                      want_attention ? &attention : nullptr);
          cur_step++;
          const size_t picked = argmax_suppressed_device(logits, suppress_tokens);
          gen_ids[i].push_back(picked);
          if (want_attention) {
            StorageView row_cpu = attn_to_cpu_rows(attention);  // [1, F]
            if (F_enc == 0)
              F_enc = row_cpu.dim(-1);
            const float* ptr = row_cpu.data<float>();
            attn_flat[i].insert(attn_flat[i].end(), ptr, ptr + row_cpu.dim(-1));
          }
          if (picked == eot_id) { hit_eot = true; break; }
          cur_token = picked;
        }

        if (hit_eot) {
          finished[i] = true;            // fully decoded within catch-up
        } else {
          active.push_back(i);
          states.push_back(std::move(state));
          next_feed.push_back(cur_token);
        }
      }

      const size_t A = active.size();
      if (A > 0) {
        // Fuse the per-row caches into one batched decoder state by
        // concatenating every cache tensor along the batch axis.  All rows
        // are at ``target_step`` positions, the audio (memory) is shared, so
        // each key has matching shape across rows.
        layers::DecoderState bstate;
        {
          const ops::Concat concat_op(0);
          for (const auto& kv : states[0]) {
            const std::string& key = kv.first;
            std::vector<const StorageView*> ins;
            ins.reserve(A);
            for (auto& st : states)
              ins.push_back(&st.at(key));
            StorageView out(kv.second.dtype(), device);
            if (A == 1)
              out = *ins[0];
            else
              concat_op(ins, out);
            bstate.emplace(key, std::move(out));
          }
        }

        dim_t cur_step = target_step;

        std::vector<size_t> cur(A);
        std::vector<bool> bfin(A, false);
        for (size_t a = 0; a < A; ++a)
          cur[a] = next_feed[a];

        for (size_t t = 0; t < max_new_tokens; ++t) {
          bool any_active = false;
          for (size_t a = 0; a < A; ++a)
            if (!bfin[a]) { any_active = true; break; }
          if (!any_active)
            break;

          std::vector<int32_t> cur_i32(A);
          for (size_t a = 0; a < A; ++a)
            cur_i32[a] = static_cast<int32_t>(cur[a]);
          StorageView ids_cpu({static_cast<dim_t>(A)}, cur_i32);
          StorageView ids = ids_cpu.to(device);

          StorageView logits(dtype, device);
          StorageView attention(dtype, device);
          (*_decoder)(cur_step, ids, bstate, &logits,
                      want_attention ? &attention : nullptr);
          cur_step++;

          if (!suppress_tokens.empty()) {
            DisableTokens disable(logits);
            for (const size_t tid : suppress_tokens)
              disable.add(static_cast<dim_t>(tid));
            disable.apply();
          }

          StorageView best_ids(DataType::INT32, device);
          StorageView best_scores(logits.dtype(), device);
          ops::TopK(1)(logits, best_scores, best_ids);          // [A, 1]
          StorageView best_cpu(DataType::INT32);
          best_cpu.copy_from(best_ids);

          StorageView attn_cpu;                                 // [A, F]
          if (want_attention) {
            attn_cpu = attn_to_cpu_rows(attention);
            if (F_enc == 0)
              F_enc = attn_cpu.dim(-1);
          }

          for (size_t a = 0; a < A; ++a) {
            if (bfin[a])
              continue;
            const size_t i = active[a];
            const size_t picked = static_cast<size_t>(best_cpu.at<int32_t>(a));
            gen_ids[i].push_back(picked);
            if (want_attention) {
              const float* row = attn_cpu.index<float>({static_cast<dim_t>(a), 0});
              attn_flat[i].insert(attn_flat[i].end(), row, row + F_enc);
            }
            if (picked == eot_id) {
              bfin[a] = true;
            } else {
              cur[a] = picked;
              if (gen_ids[i].size() >= max_new_tokens)
                bfin[a] = true;
            }
          }
        }
      }

      // Assemble per-row attention matrices (or empty views).
      std::vector<StorageView> attn_out(n_rows);
      if (want_attention) {
        for (size_t i = 0; i < n_rows; ++i) {
          const dim_t n = static_cast<dim_t>(gen_ids[i].size());
          if (n == 0 || F_enc == 0)
            continue;
          attn_out[i] = StorageView({n, F_enc}, attn_flat[i]);
        }
      }

      return {std::move(gen_ids), std::move(attn_out)};
    }

    std::vector<size_t>
    WhisperReplica::generate_speculative(
        WhisperReplica& draft,
        StorageView main_features,
        StorageView draft_features,
        const std::vector<size_t>& prompt,
        size_t num_speculative_tokens,
        size_t max_length,
        size_t eot_id,
        const std::vector<size_t>& suppress_tokens,
        const std::vector<int32_t>& d2m,
        const std::vector<int32_t>& m2d,
        size_t min_speculative_tokens,
        size_t max_speculative_tokens,
        bool reset_adaptive_state) {
      PROFILE("WhisperReplica::generate_speculative");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      // Adaptive-K controller: symmetric +1/-1 with *persistent* state.  A
      // round where every drafted token is accepted bumps K up by two; any
      // rejection nudges it down by one (AIMD, as in HF assisted decoding).
      // The equilibrium is therefore the K at which ~1/3 of the rounds fully
      // accept -- a purely acceptance-driven operating point that needs no
      // hand-tuned window.  The up-bias keeps K near the cap when acceptance
      // is high (the wall-time optimum for large-v2+turbo) while still backing
      // off on genuinely low-acceptance audio.  Because K persists
      // across calls (``_spec_k_state``), over a chunked transcription it
      // converges to that equilibrium regardless of the seed, so the seed
      // only matters for the very first chunk (see research/timing
      // experiment_adaptive_k.py).  Disabled (fixed K) unless a usable
      // [min, max] window is supplied.
      const bool adaptive =
          max_speculative_tokens > 0
          && max_speculative_tokens > min_speculative_tokens;
      const size_t k_lo = std::max<size_t>(1, min_speculative_tokens);
      const size_t k_hi = std::max(k_lo, max_speculative_tokens);
      constexpr double K_STEP_UP = 2.0;    // climb on a fully-accepted round
      constexpr double K_STEP_DOWN = 1.0;  // nudge on any rejection
      double k_cur = static_cast<double>(num_speculative_tokens);
      if (adaptive) {
        // Persist across chunks: continue from the stored state unless the
        // caller asked to reset (first chunk of a new audio) or it is still
        // uninitialised.
        if (!reset_adaptive_state && _spec_k_state > 0.0)
          k_cur = _spec_k_state;
        k_cur = std::min(std::max(k_cur, static_cast<double>(k_lo)),
                         static_cast<double>(k_hi));
      }
      const size_t K = num_speculative_tokens;

      // Encode both models once and reuse the encoder output for every
      // (re-)prefill (matches the Python ``_encode_both`` + reuse).  A
      // fresh copy is handed to each ``prefill`` because it takes the
      // features by value; ``maybe_encode`` detects already-encoded
      // input and passes it straight through.
      const StorageView main_enc = maybe_encode(std::move(main_features));
      const StorageView draft_enc = draft.maybe_encode(std::move(draft_features));

      auto to_draft = [&](size_t main_id) -> int32_t {
        return map_token(m2d, main_id);
      };
      auto to_main = [&](size_t draft_id) -> int32_t {
        return map_token(d2m, draft_id);
      };
      auto prompt_to_draft = [&](const std::vector<size_t>& main_prompt) {
        std::vector<size_t> out;
        out.reserve(main_prompt.size());
        for (const size_t t : main_prompt) {
          const int32_t d = to_draft(t);
          out.push_back(d == -1 ? t : static_cast<size_t>(d));
        }
        return out;
      };

      const int32_t draft_eot_i = to_draft(eot_id);
      const size_t draft_eot =
          draft_eot_i == -1 ? eot_id : static_cast<size_t>(draft_eot_i);

      // Prefill both decoders from a main-space prompt and return the
      // main model's (suppressed) first-token argmax.
      auto prefill_both = [&](const std::vector<size_t>& main_prompt,
                              WhisperDecoderState& main_state,
                              WhisperDecoderState& draft_state) -> size_t {
        const std::vector<size_t> draft_prompt = prompt_to_draft(main_prompt);
        auto mres = prefill(StorageView(main_enc), main_prompt);
        main_state = std::move(mres.first);
        StorageView main_logits = std::move(mres.second);
        auto dres = draft.prefill(StorageView(draft_enc), draft_prompt);
        draft_state = std::move(dres.first);
        return argmax_suppressed_device(main_logits, suppress_tokens);
      };

      WhisperDecoderState main_state;
      WhisperDecoderState draft_state;
      size_t main_next = prefill_both(prompt, main_state, draft_state);

      std::vector<size_t> accepted;
      const std::vector<size_t>& full_prompt = prompt;

      auto reprefill_full = [&]() {
        std::vector<size_t> fp(full_prompt);
        fp.insert(fp.end(), accepted.begin(), accepted.end());
        main_next = prefill_both(fp, main_state, draft_state);
      };

      while (accepted.size() < max_length) {
        if (main_next == eot_id) {
          accepted.push_back(main_next);
          break;
        }

        const size_t budget = max_length - accepted.size();
        if (budget <= 1) {  // draft_n = min(K, budget - 1) <= 0
          accepted.push_back(main_next);
          break;
        }
        size_t K_round = K;
        if (adaptive) {
          K_round = static_cast<size_t>(k_cur + 0.5);
          if (K_round < k_lo)
            K_round = k_lo;
          if (K_round > k_hi)
            K_round = k_hi;
        }
        const size_t draft_n = std::min(K_round, budget - 1);

        // --- Draft phase (greedy, no suppression on the draft model) ---
        std::vector<size_t> candidates_main;
        std::vector<size_t> candidates_draft;
        const int32_t seed_draft = to_draft(main_next);

        if (seed_draft == -1) {
          accepted.push_back(main_next);
          reprefill_full();
          continue;
        }

        size_t draft_tok = static_cast<size_t>(seed_draft);
        for (size_t i = 0; i < draft_n; ++i) {
          draft_tok = draft.forward_step_greedy(draft_state, draft_tok, {});
          const int32_t main_tok = to_main(draft_tok);
          if (main_tok == -1)
            break;
          candidates_draft.push_back(draft_tok);
          candidates_main.push_back(static_cast<size_t>(main_tok));
          if (draft_tok == draft_eot)
            break;
        }

        if (candidates_main.empty()) {
          accepted.push_back(main_next);
          reprefill_full();
          continue;
        }

        // --- Verify phase (single batched main pass) ---
        const dim_t main_step_before_verify = main_state.current_step;
        std::vector<size_t> batch_tokens;
        batch_tokens.reserve(candidates_main.size() + 1);
        batch_tokens.push_back(main_next);
        batch_tokens.insert(batch_tokens.end(),
                            candidates_main.begin(), candidates_main.end());

        const std::vector<size_t> verify_preds =
            forward_batch_greedy(main_state, batch_tokens, suppress_tokens);

        // --- Strict acceptance ---
        size_t n_draft_accepted = 0;
        bool has_correction = false;
        size_t correction = 0;
        for (size_t j = 0; j < candidates_main.size(); ++j) {
          if (j >= verify_preds.size())
            break;
          if (verify_preds[j] == candidates_main[j]) {
            ++n_draft_accepted;
            if (candidates_main[j] == eot_id)
              break;
          } else {
            correction = verify_preds[j];
            has_correction = true;
            break;
          }
        }

        accepted.push_back(main_next);
        for (size_t j = 0; j < n_draft_accepted; ++j)
          accepted.push_back(candidates_main[j]);
        if (has_correction)
          accepted.push_back(correction);

        if (!accepted.empty() && accepted.back() == eot_id)
          break;

        // --- State management ---
        const bool all_accepted =
            (n_draft_accepted == candidates_main.size()) && !has_correction;

        // Adapt K for the next round: AIMD +2 on a fully-accepted round, -1 on
        // any rejection (converges to the ~1/3-full-accept equilibrium, biased
        // toward the cap when acceptance is high).
        if (adaptive) {
          if (all_accepted)
            k_cur = std::min(static_cast<double>(k_hi), k_cur + K_STEP_UP);
          else
            k_cur = std::max(static_cast<double>(k_lo), k_cur - K_STEP_DOWN);
        }

        if (all_accepted) {
          if (!candidates_draft.empty())
            draft.forward_step_greedy(draft_state, candidates_draft.back(), {});
          const size_t last_idx = candidates_main.size();
          if (last_idx < verify_preds.size()) {
            main_next = verify_preds[last_idx];
          } else {
            reprefill_full();
          }
        } else {
          const dim_t rollback_to =
              main_step_before_verify + 1 + static_cast<dim_t>(n_draft_accepted);
          main_state.truncate_to_step(rollback_to);
          main_next = forward_step_greedy(main_state, accepted.back(),
                                          suppress_tokens);
          std::vector<size_t> fp(full_prompt);
          fp.insert(fp.end(), accepted.begin(), accepted.end());
          auto dres = draft.prefill(StorageView(draft_enc), prompt_to_draft(fp));
          draft_state = std::move(dres.first);
        }
      }

      // Persist the converged K for the next chunk of this audio.
      if (adaptive)
        _spec_k_state = k_cur;

      return accepted;
    }

    StorageView
    WhisperReplica::collected_attention_to_cpu(const WhisperDecoderState& state,
                                               bool average_heads) const {
      PROFILE("WhisperReplica::collected_attention_to_cpu");
      const auto& rows = state.collected_attention;
      if (rows.empty())
        return StorageView();

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      const DataType dtype = rows.front().dtype();

      // ``Concat`` requires pointers to all inputs.  Each row has shape
      // ``[1, num_heads, F_enc]``, so concatenating along axis 0 yields
      // ``[T, num_heads, F_enc]`` -- the natural per-step stacking.
      std::vector<const StorageView*> ptrs;
      ptrs.reserve(rows.size());
      for (const auto& row : rows)
        ptrs.push_back(&row);

      StorageView stacked(dtype, device);
      ops::Concat(/*axis=*/0)(ptrs, stacked);

      // Optional head-mean -> [T, F_enc].
      StorageView reduced(dtype, device);
      if (average_heads) {
        // Mean over axis=1 (the num_heads axis).  ``ops::Mean`` keeps
        // dims but writes the reduced shape, so the result is
        // ``[T, 1, F_enc]``; we drop the singleton dim below.
        ops::Mean(/*axis=*/1)(stacked, reduced);
      } else {
        reduced = std::move(stacked);
      }

      // Cast to float32 on the device (Python timing code expects fp32).
      StorageView fp32(DataType::FLOAT32, device);
      if (reduced.dtype() == DataType::FLOAT32)
        fp32 = std::move(reduced);
      else
        fp32 = reduced.to(DataType::FLOAT32);

      // Single bulk PCIe transfer to CPU.
      StorageView cpu = fp32.to(Device::CPU);

      // Squeeze the singleton ``num_heads`` axis when averaging so the
      // Python side gets a clean ``[T, F_enc]`` array.
      if (average_heads && cpu.rank() == 3 && cpu.dim(1) == 1)
        cpu.squeeze(1);

      return cpu;
    }

    std::pair<StorageView, StorageView>
    WhisperReplica::forward_batch_with_attention(WhisperDecoderState& wds,
                                                 const std::vector<size_t>& token_ids) {
      PROFILE("WhisperReplica::forward_batch_with_attention");
      require_alignment_heads_configured(*_decoder);

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _decoder->device();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      std::vector<std::vector<size_t>> batch = {token_ids};
      const StorageView ids = layers::make_sequence_inputs(batch, device);

      StorageView logits(_decoder->output_type(), device);
      // For a multi-position (sequence) input the decoder writes the
      // cross-attention as ``[1, num_selected_heads, T, F_enc]``
      // (post-softmax, since set_alignment_heads enabled normalized
      // attention).
      StorageView attention(_decoder->output_type(), device);
      _decoder->forward_with_logits(ids, wds.current_step, wds.state, logits, &attention);
      wds.current_step += static_cast<dim_t>(token_ids.size());

      // Head-average -> ``[1, 1, T, F]`` then drop the head + batch axes
      // to a clean ``[T, F_enc]`` matrix on CPU (single bulk transfer).
      StorageView reduced(attention.dtype(), device);
      ops::Mean(/*axis=*/1)(attention, reduced);

      StorageView fp32(DataType::FLOAT32, device);
      if (reduced.dtype() == DataType::FLOAT32)
        fp32 = std::move(reduced);
      else
        fp32 = reduced.to(DataType::FLOAT32);

      StorageView cpu = fp32.to(Device::CPU);
      // ``cpu`` is ``[1, 1, T, F]``: squeeze the head axis (1) then the
      // batch axis (0).  Guard each squeeze in case a degenerate shape
      // (e.g. T == 1) collapsed an axis earlier.
      if (cpu.rank() == 4 && cpu.dim(1) == 1)
        cpu.squeeze(1);
      if (cpu.rank() == 3 && cpu.dim(0) == 1)
        cpu.squeeze(0);

      return {std::move(logits), std::move(cpu)};
    }


    std::future<std::pair<WhisperDecoderState, StorageView>>
    Whisper::prefill(const StorageView& features,
                     std::vector<size_t> prompt) {
      return post<std::pair<WhisperDecoderState, StorageView>>(
        [features = features.sync_copy(),
         prompt = std::move(prompt)]
        (WhisperReplica& replica) mutable {
          return replica.prefill(std::move(features), prompt);
        });
    }

    std::future<StorageView>
    Whisper::forward_step(WhisperDecoderState& state,
                          size_t token_id) {
      return post<StorageView>(
        [&state, token_id]
        (WhisperReplica& replica) mutable {
          return replica.forward_step(state, token_id);
        });
    }

    std::future<StorageView>
    Whisper::forward_batch(WhisperDecoderState& state,
                           std::vector<size_t> token_ids) {
      return post<StorageView>(
        [&state, token_ids = std::move(token_ids)]
        (WhisperReplica& replica) mutable {
          return replica.forward_batch(state, token_ids);
        });
    }

    std::future<size_t>
    Whisper::forward_step_greedy(WhisperDecoderState& state,
                                 size_t token_id,
                                 std::vector<size_t> suppress_tokens) {
      return post<size_t>(
        [&state, token_id, suppress_tokens = std::move(suppress_tokens)]
        (WhisperReplica& replica) mutable {
          return replica.forward_step_greedy(state, token_id, suppress_tokens);
        });
    }

    std::future<std::vector<size_t>>
    Whisper::forward_batch_greedy(WhisperDecoderState& state,
                                  std::vector<size_t> token_ids,
                                  std::vector<size_t> suppress_tokens) {
      return post<std::vector<size_t>>(
        [&state, token_ids = std::move(token_ids),
         suppress_tokens = std::move(suppress_tokens)]
        (WhisperReplica& replica) mutable {
          return replica.forward_batch_greedy(state, token_ids, suppress_tokens);
        });
    }

    void
    Whisper::set_alignment_heads(std::vector<std::pair<dim_t, dim_t>> heads) {
      std::lock_guard<std::mutex> lock(_alignment_heads_mutex);
      _alignment_heads = std::move(heads);
    }

    std::vector<std::pair<dim_t, dim_t>>
    Whisper::get_alignment_heads_copy() const {
      std::lock_guard<std::mutex> lock(_alignment_heads_mutex);
      return _alignment_heads;
    }

    std::future<std::tuple<WhisperDecoderState, StorageView, StorageView>>
    Whisper::prefill_with_attention(const StorageView& features,
                                    std::vector<size_t> prompt) {
      auto heads = get_alignment_heads_copy();
      return post<std::tuple<WhisperDecoderState, StorageView, StorageView>>(
        [features = features.sync_copy(),
         prompt = std::move(prompt),
         heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          replica.set_alignment_heads(heads);
          return replica.prefill_with_attention(std::move(features), prompt);
        });
    }

    std::future<std::pair<StorageView, StorageView>>
    Whisper::forward_step_with_attention(WhisperDecoderState& state,
                                         size_t token_id) {
      auto heads = get_alignment_heads_copy();
      return post<std::pair<StorageView, StorageView>>(
        [&state, token_id, heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          replica.set_alignment_heads(heads);
          return replica.forward_step_with_attention(state, token_id);
        });
    }

    std::future<std::pair<size_t, StorageView>>
    Whisper::forward_step_greedy_with_attention(WhisperDecoderState& state,
                                                size_t token_id) {
      auto heads = get_alignment_heads_copy();
      return post<std::pair<size_t, StorageView>>(
        [&state, token_id, heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          replica.set_alignment_heads(heads);
          return replica.forward_step_greedy_with_attention(state, token_id);
        });
    }

    std::future<std::pair<StorageView, StorageView>>
    Whisper::forward_batch_with_attention(WhisperDecoderState& state,
                                          std::vector<size_t> token_ids) {
      auto heads = get_alignment_heads_copy();
      return post<std::pair<StorageView, StorageView>>(
        [&state, token_ids = std::move(token_ids), heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          replica.set_alignment_heads(heads);
          return replica.forward_batch_with_attention(state, token_ids);
        });
    }

    std::future<StorageView>
    Whisper::collected_attention_to_cpu(const WhisperDecoderState& state,
                                        bool average_heads) {
      return post<StorageView>(
        [&state, average_heads]
        (WhisperReplica& replica) mutable {
          return replica.collected_attention_to_cpu(state, average_heads);
        });
    }

    std::future<std::pair<WhisperDecoderState, std::vector<size_t>>>
    Whisper::generate_greedy_with_attention(
        StorageView features,
        std::vector<size_t> prompt,
        size_t max_new_tokens,
        size_t eot_id,
        std::vector<size_t> suppress_tokens,
        std::vector<size_t> ban_first_tokens) {
      auto heads = get_alignment_heads_copy();
      return post<std::pair<WhisperDecoderState, std::vector<size_t>>>(
        [features = features.sync_copy(),
         prompt = std::move(prompt),
         max_new_tokens,
         eot_id,
         suppress_tokens = std::move(suppress_tokens),
         ban_first_tokens = std::move(ban_first_tokens),
         heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          replica.set_alignment_heads(heads);
          return replica.generate_greedy_with_attention(
              std::move(features),
              prompt,
              max_new_tokens,
              eot_id,
              suppress_tokens,
              ban_first_tokens);
        });
    }

    std::future<std::pair<std::vector<std::vector<size_t>>, std::vector<StorageView>>>
    Whisper::generate_dual_greedy(
        StorageView features,
        std::vector<std::vector<size_t>> prompts,
        size_t max_new_tokens,
        size_t eot_id,
        std::vector<size_t> suppress_tokens,
        bool want_attention) {
      auto heads = get_alignment_heads_copy();
      return post<std::pair<std::vector<std::vector<size_t>>, std::vector<StorageView>>>(
        [features = features.sync_copy(),
         prompts = std::move(prompts),
         max_new_tokens,
         eot_id,
         suppress_tokens = std::move(suppress_tokens),
         want_attention,
         heads = std::move(heads)]
        (WhisperReplica& replica) mutable {
          if (want_attention)
            replica.set_alignment_heads(heads);
          return replica.generate_dual_greedy(
              std::move(features),
              prompts,
              max_new_tokens,
              eot_id,
              suppress_tokens,
              want_attention);
        });
    }

    std::future<std::vector<size_t>>
    Whisper::generate_speculative(
        Whisper& draft_pool,
        StorageView main_features,
        StorageView draft_features,
        std::vector<size_t> prompt,
        size_t num_speculative_tokens,
        size_t max_length,
        size_t eot_id,
        std::vector<size_t> suppress_tokens,
        std::vector<int32_t> d2m,
        std::vector<int32_t> m2d,
        size_t min_speculative_tokens,
        size_t max_speculative_tokens,
        bool reset_adaptive_state) {
      return post<std::vector<size_t>>(
        [&draft_pool,
         main_features = main_features.sync_copy(),
         draft_features = draft_features.sync_copy(),
         prompt = std::move(prompt),
         num_speculative_tokens,
         max_length,
         eot_id,
         suppress_tokens = std::move(suppress_tokens),
         d2m = std::move(d2m),
         m2d = std::move(m2d),
         min_speculative_tokens,
         max_speculative_tokens,
         reset_adaptive_state]
        (WhisperReplica& replica) mutable {
          // Drive the draft model's replica directly from this (main)
          // worker thread; no job is posted to the draft pool, so both
          // models share one CUDA stream and never race.
          WhisperReplica& draft_replica = draft_pool.get_first_replica_mutable();
          return replica.generate_speculative(
              draft_replica,
              std::move(main_features),
              std::move(draft_features),
              prompt,
              num_speculative_tokens,
              max_length,
              eot_id,
              suppress_tokens,
              d2m,
              m2d,
              min_speculative_tokens,
              max_speculative_tokens,
              reset_adaptive_state);
        });
    }


    class ApplyTimestampRules : public LogitsProcessor {
    private:
      const size_t _eot_id;
      const size_t _no_timestamps_id;
      const size_t _timestamp_begin_id;
      const size_t _timestamp_end_id;
      const size_t _max_initial_timestamp_id;

    public:
      ApplyTimestampRules(const size_t eot_id,
                          const size_t no_timestamps_id,
                          const size_t timestamp_begin_id,
                          const size_t timestamp_end_id,
                          const size_t max_initial_timestamp_id)
        : _eot_id(eot_id)
        , _no_timestamps_id(no_timestamps_id)
        , _timestamp_begin_id(timestamp_begin_id)
        , _timestamp_end_id(timestamp_end_id)
        , _max_initial_timestamp_id(max_initial_timestamp_id)
      {
      }

      void apply(dim_t step,
                 StorageView& logits,
                 DisableTokens& disable_tokens,
                 const StorageView& sequences,
                 const std::vector<dim_t>& batch_offset,
                 const std::vector<std::vector<size_t>>* prefix) override {
        std::vector<dim_t> check_timestamps_prob_for_batch;
        const dim_t batch_size = logits.dim(0);

        for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
          const dim_t sample_begin = get_sample_begin(batch_size, batch_id, batch_offset, prefix);

          // Suppress <|notimestamps|>.
          disable_tokens.add(batch_id, _no_timestamps_id);

          if (step == sample_begin && step == 0) {
            // Suppress non timestamps at the beginning.
            for (size_t i = 0; i < _timestamp_begin_id; ++i)
              disable_tokens.add(batch_id, i);

            // Apply max_initial_timestamp option.
            for (size_t i = _max_initial_timestamp_id + 1; i <= _timestamp_end_id; ++i)
              disable_tokens.add(batch_id, i);

          } else if (step > sample_begin) {
            // Timestamps have to appear in pairs, except directly before EOT.
            const size_t last_token = sequences.at<int32_t>({batch_id, step - 1});

            if (last_token >= _timestamp_begin_id) {
              const size_t penultimate_token = (step - 1 > sample_begin
                                                ? sequences.at<int32_t>({batch_id, step - 2})
                                                : last_token);

              if (penultimate_token >= _timestamp_begin_id) {  // has to be non-timestamp
                for (size_t i = _timestamp_begin_id; i <= _timestamp_end_id; ++i)
                  disable_tokens.add(batch_id, i);
              } else {  // cannot be normal text tokens
                for (size_t i = 0; i < _eot_id; ++i)
                  disable_tokens.add(batch_id, i);
                for (size_t i = _timestamp_begin_id; i < last_token; ++i)
                  disable_tokens.add(batch_id, i);
                check_timestamps_prob_for_batch.push_back(batch_id);
              }
            } else {
              check_timestamps_prob_for_batch.push_back(batch_id);

              // Timestamps shouldn't decrease: forbid timestamp tokens smaller than the last.
              for (dim_t t = step - 1; t >= sample_begin; --t) {
                const size_t token = sequences.at<int32_t>({batch_id, t});

                if (token >= _timestamp_begin_id) {
                  for (size_t i = _timestamp_begin_id; i <= token; ++i)
                    disable_tokens.add(batch_id, i);
                  break;
                }
              }
            }
          }
        }

        if (!check_timestamps_prob_for_batch.empty()) {
          // Apply all changes to the logits before computing the log softmax.
          disable_tokens.apply();

          StorageView log_probs(logits.dtype(), logits.device());
          ops::LogSoftMax()(logits, log_probs);

          for (const dim_t batch_id : check_timestamps_prob_for_batch) {
            bool sample_timestamp = false;

            DEVICE_AND_FLOAT_DISPATCH(
              "ApplyTimestampRules", log_probs.device(), log_probs.dtype(),
              (sample_timestamp = should_sample_timestamp<D, T>(log_probs, batch_id)));

            if (sample_timestamp) {
              for (size_t i = 0; i < _timestamp_begin_id; ++i)
                disable_tokens.add(batch_id, i);
            }
          }
        }
      }

      template <Device D, typename T>
      bool should_sample_timestamp(const StorageView& log_probs, const dim_t batch_id) {
        const dim_t num_text_tokens = _timestamp_begin_id;
        const dim_t num_timestamp_tokens = _timestamp_end_id - _timestamp_begin_id + 1;

        const T* text_log_probs = log_probs.index<T>({batch_id, 0});
        const T* timestamp_log_probs = text_log_probs + num_text_tokens;

        // If sum of probability over timestamps is above any other token, sample timestamp.
        const float max_text_token_log_prob = primitives<D>::max(text_log_probs, num_text_tokens);
        const float timestamp_log_prob = primitives<D>::logsumexp(timestamp_log_probs,
                                                                  num_timestamp_tokens);

        return timestamp_log_prob > max_text_token_log_prob;
      }

    };

  }
}
