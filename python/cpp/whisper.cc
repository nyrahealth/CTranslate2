#include "module.h"

#include <ctranslate2/models/whisper.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class WhisperWrapper : public ReplicaPoolHelper<models::Whisper> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      bool is_multilingual() const {
        return _pool->is_multilingual();
      }

      size_t n_mels() const {
        return _pool->n_mels();
      }

      size_t num_languages() const {
        return _pool->num_languages();
      }

      StorageView encode(const StorageView& features, const bool to_cpu) {
        return _pool->encode(features, to_cpu).get();
      }

      std::pair<std::shared_ptr<models::WhisperDecoderState>, StorageView>
      prefill(const StorageView& features, Ids prompt) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        auto result = _pool->prefill(features, std::move(prompt)).get();
        auto state_ptr = std::make_shared<models::WhisperDecoderState>(
            std::move(result.first));
        return {state_ptr, std::move(result.second)};
      }

      StorageView forward_step(std::shared_ptr<models::WhisperDecoderState> state,
                               size_t token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_step(*state, token_id).get();
      }

      StorageView forward_batch(std::shared_ptr<models::WhisperDecoderState> state,
                                Ids token_ids) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_batch(*state, std::move(token_ids)).get();
      }

      size_t forward_step_greedy(std::shared_ptr<models::WhisperDecoderState> state,
                                 size_t token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_step_greedy(*state, token_id).get();
      }

      std::vector<size_t> forward_batch_greedy(
          std::shared_ptr<models::WhisperDecoderState> state,
          Ids token_ids) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_batch_greedy(*state, std::move(token_ids)).get();
      }

      void set_alignment_heads(const std::vector<std::pair<int64_t, int64_t>>& heads) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        std::vector<std::pair<dim_t, dim_t>> dim_heads;
        dim_heads.reserve(heads.size());
        for (const auto& [layer, head] : heads)
          dim_heads.emplace_back(static_cast<dim_t>(layer), static_cast<dim_t>(head));
        _pool->set_alignment_heads(std::move(dim_heads));
      }

      std::tuple<std::shared_ptr<models::WhisperDecoderState>, StorageView, StorageView>
      prefill_with_attention(const StorageView& features, Ids prompt) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        auto result = _pool->prefill_with_attention(features, std::move(prompt)).get();
        auto state_ptr = std::make_shared<models::WhisperDecoderState>(
            std::move(std::get<0>(result)));
        return {state_ptr, std::move(std::get<1>(result)), std::move(std::get<2>(result))};
      }

      std::pair<StorageView, StorageView>
      forward_step_with_attention(std::shared_ptr<models::WhisperDecoderState> state,
                                  size_t token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_step_with_attention(*state, token_id).get();
      }

      std::pair<size_t, StorageView>
      forward_step_greedy_with_attention(std::shared_ptr<models::WhisperDecoderState> state,
                                         size_t token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->forward_step_greedy_with_attention(*state, token_id).get();
      }

      std::variant<std::vector<models::WhisperGenerationResult>,
                   std::vector<AsyncResult<models::WhisperGenerationResult>>>
      generate(const StorageView& features,
               std::variant<BatchTokens, BatchIds> prompts,
               bool asynchronous,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               bool return_logits_vocab,
               bool return_no_speech_prob,
               size_t max_initial_timestamp_index,
               bool suppress_blank,
               const std::optional<std::vector<int>>& suppress_tokens,
               size_t sampling_topk,
               float sampling_temperature) {
        std::vector<std::future<models::WhisperGenerationResult>> futures;

        models::WhisperOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;
        options.return_logits_vocab = return_logits_vocab;
        options.return_no_speech_prob = return_no_speech_prob;
        options.max_initial_timestamp_index = max_initial_timestamp_index;
        options.suppress_blank = suppress_blank;

        if (suppress_tokens)
          options.suppress_tokens = suppress_tokens.value();
        else
          options.suppress_tokens.clear();
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (prompts.index() == 0)
          futures = _pool->generate(features, std::get<BatchTokens>(prompts), options);
        else
          futures = _pool->generate(features, std::get<BatchIds>(prompts), options);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(const StorageView& features) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        auto futures = _pool->detect_language(features);
        return wait_on_futures(std::move(futures));
      }

      std::vector<models::WhisperAlignmentResult>
      align(const StorageView& features,
            Ids start_sequence,
            BatchIds text_tokens,
            const std::variant<size_t, std::vector<size_t>>& num_frames,
            size_t median_filter_width) {
        const size_t batch_size = text_tokens.size();

        std::vector<size_t> batch_num_frames;
        if (num_frames.index() == 0)
          batch_num_frames.resize(batch_size, std::get<size_t>(num_frames));
        else
          batch_num_frames = std::get<std::vector<size_t>>(num_frames);
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        auto futures = _pool->align(features,
                                    std::move(start_sequence),
                                    std::move(text_tokens),
                                    std::move(batch_num_frames),
                                    median_filter_width);
        return wait_on_futures(std::move(futures));
      }
    };


    void register_whisper(py::module& m) {
      py::class_<models::WhisperDecoderState, std::shared_ptr<models::WhisperDecoderState>>(
        m, "WhisperDecoderState",
        "Opaque container for Whisper decoder KV-cache state.")

        .def_readonly("current_step", &models::WhisperDecoderState::current_step,
                      "Number of tokens processed so far (prompt + generated).")

        .def("clone", [](const models::WhisperDecoderState& self) {
          return std::make_shared<models::WhisperDecoderState>(self.deep_copy());
        },
             py::call_guard<py::gil_scoped_release>(),
             "Return a deep copy of this state (GPU tensors are duplicated).")

        .def("truncate_to_step",
             [](std::shared_ptr<models::WhisperDecoderState> self, int64_t target_step) {
               self->truncate_to_step(static_cast<dim_t>(target_step));
             },
             py::arg("target_step"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Truncate the KV cache to keep only the first ``target_step`` decode steps.

                 Slices self-attention key/value tensors along the time dimension,
                 trims the collected cross-attention buffer to match, and resets
                 the internal step counter.  Much cheaper than a full re-prefill
                 when rolling back after a speculative-decoding rejection or a
                 hallucination-repair rewind.

                 Arguments:
                   target_step: Number of decode steps to retain.
             )pbdoc")

        .def_property_readonly(
            "collected_attention",
            [](const models::WhisperDecoderState& self) {
              return self.collected_attention;
            },
            R"pbdoc(
                Per-step cross-attention rows captured by the ``*_with_attention``
                APIs.

                Each entry is a :class:`StorageView` of shape
                ``[1, num_selected_heads, F_enc]`` (post-softmax — each
                ``(batch, head, :)`` row sums to 1 over encoder frames).  Index
                ``k`` corresponds to the attention emitted while predicting the
                ``k``-th newly-generated token (i.e. the row associated with
                that token's identity).

                The list is empty if attention extraction was never enabled or
                if it has been truncated all the way back by
                :meth:`truncate_to_step`.
            )pbdoc")

        .def("__repr__", [](const models::WhisperDecoderState& s) {
          return "WhisperDecoderState(current_step=" + std::to_string(s.current_step)
            + ", num_entries=" + std::to_string(s.state.size())
            + ", attention_steps=" + std::to_string(s.collected_attention.size()) + ")";
        })
        ;

      py::class_<models::WhisperGenerationResult>(m, "WhisperGenerationResult",
                                                  "A generation result from the Whisper model.")

        .def_readonly("sequences", &models::WhisperGenerationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &models::WhisperGenerationResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &models::WhisperGenerationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")
        .def_readonly("logits", &models::WhisperGenerationResult::logits,
                      "logits in each sequence (empty if :obj:`return_logits_vocab` was disabled).")
        .def_readonly("no_speech_prob", &models::WhisperGenerationResult::no_speech_prob,
                      "Probability of the no speech token (0 if :obj:`return_no_speech_prob` was disabled).")

        .def("__repr__", [](const models::WhisperGenerationResult& result) {
          return "WhisperGenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", logits=" + std::string(py::repr(py::cast(result.logits)))
            + ", no_speech_prob=" + std::string(py::repr(py::cast(result.no_speech_prob)))
            + ")";
        })
        ;

      declare_async_wrapper<models::WhisperGenerationResult>(m, "WhisperGenerationResultAsync");

      py::class_<models::WhisperAlignmentResult>(m, "WhisperAlignmentResult",
                                                 "An alignment result from the Whisper model.")

        .def_readonly("alignments", &models::WhisperAlignmentResult::alignments,
                      "List of aligned text and time indices.")
        .def_readonly("text_token_probs", &models::WhisperAlignmentResult::text_token_probs,
                      "Probabilities of text tokens.")

        .def("__repr__", [](const models::WhisperAlignmentResult& result) {
          return "WhisperAlignmentResult(alignments=" + std::string(py::repr(py::cast(result.alignments)))
            + ", text_token_probs=" + std::string(py::repr(py::cast(result.text_token_probs)))
            + ")";
        })
        ;

      py::class_<WhisperWrapper>(
        m, "Whisper",
        R"pbdoc(
            Implements the Whisper speech recognition model published by OpenAI.

            See Also:
               https://github.com/openai/whisper
        )pbdoc")

        .def_property_readonly("is_multilingual", &WhisperWrapper::is_multilingual,
                               "Returns ``True`` if this model is multilingual.")

        .def_property_readonly("n_mels", &WhisperWrapper::n_mels,
                               "Returns dimension of mel input features.")

        .def_property_readonly("num_languages", &WhisperWrapper::num_languages,
                               "Returns the number of languages supported.")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Whisper model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &WhisperWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &WhisperWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &WhisperWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &WhisperWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &WhisperWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &WhisperWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &WhisperWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &WhisperWrapper::encode,
             py::arg("features"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``.
                   to_cpu: Copy the encoder output to the CPU before returning the value.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("generate", &WhisperWrapper::generate,
             py::arg("features"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_logits_vocab")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("max_initial_timestamp_index")=50,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features and generates from the given prompt.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.
                   prompts: Batch of initial string tokens or token IDs.
                   asynchronous: Run the model asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
                   return_logits_vocab: Include the log probs in the output
                   return_no_speech_prob: Include the probability of the no speech token in the
                     result.
                   max_initial_timestamp_index: Maximum index of the first predicted timestamp.
                   suppress_blank: Suppress blank outputs at the beginning of the sampling.
                   suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                     of symbols as defined in the model ``config.json`` file.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.

                 Returns:
                   A list of generation results.
             )pbdoc")

        .def("detect_language", &WhisperWrapper::detect_language,
             py::arg("features"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Returns the probability of each language.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.

                 Returns:
                   For each batch, a list of pairs (language, probability) ordered from
                   best to worst probability.

                 Raises:
                   RuntimeError: if the model is not multilingual.
             )pbdoc")

        .def("align", &WhisperWrapper::align,
             py::arg("features"),
             py::arg("start_sequence"),
             py::arg("text_tokens"),
             py::arg("num_frames"),
             py::kw_only(),
             py::arg("median_filter_width")=7,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Computes the alignments between the text tokens and the audio.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.
                   start_sequence: The start sequence tokens.
                   text_tokens: Batch of text tokens to align.
                   num_frames: Number of non padding frames in the features.
                   median_filter_width: Width of the median filter kernel.

                 Returns:
                   A list of alignment results.
             )pbdoc")

        .def("prefill", &WhisperWrapper::prefill,
             py::arg("features"),
             py::arg("prompt"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encode features and prefill the decoder with a prompt.

                 Processes the full prompt and returns a decoder state containing the
                 KV-cache, plus logits for the first generation position.  Use this
                 with :meth:`forward_step` and :meth:`forward_batch` for incremental
                 decoding with KV-cache persistence.

                 Arguments:
                   features: Mel spectrogram with shape ``[1, n_mels, chunk_length]``
                     or pre-encoded features from :meth:`encode`.
                   prompt: Token IDs for the full decoder prompt.

                 Returns:
                   A tuple ``(state, logits)`` where ``state`` is a
                   :class:`WhisperDecoderState` and ``logits`` has shape
                   ``[1, vocab_size]``.
             )pbdoc")

        .def("forward_step", &WhisperWrapper::forward_step,
             py::arg("state"),
             py::arg("token_id"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Run one decoder step with KV-cache reuse.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_id: The token ID to feed at the current step.

                 Returns:
                   Logits with shape ``[1, vocab_size]``.
             )pbdoc")

        .def("forward_batch", &WhisperWrapper::forward_batch,
             py::arg("state"),
             py::arg("token_ids"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Process multiple tokens in parallel with KV-cache reuse.

                 Extends the KV-cache with all provided tokens and returns
                 logits at every position.  Useful for speculative decoding
                 verification where the main model checks K draft tokens
                 in a single forward pass.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_ids: List of token IDs to process.

                 Returns:
                   Logits with shape ``[1, num_tokens, vocab_size]``.
             )pbdoc")

        .def("forward_step_greedy", &WhisperWrapper::forward_step_greedy,
             py::arg("state"),
             py::arg("token_id"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Run one decoder step and return the greedy (argmax) token ID.

                 Like :meth:`forward_step` but performs TopK(1) on the GPU and
                 returns a single integer instead of the full logits tensor.
                 Avoids the large GPU-to-CPU transfer of the vocabulary logits.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_id: The token ID to feed at the current step.

                 Returns:
                   The greedy next-token ID.
             )pbdoc")

        .def("forward_batch_greedy", &WhisperWrapper::forward_batch_greedy,
             py::arg("state"),
             py::arg("token_ids"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Process multiple tokens and return greedy predictions at each position.

                 Like :meth:`forward_batch` but performs TopK(1) on the GPU and
                 returns a list of token IDs instead of the full logits tensor.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_ids: List of token IDs to process.

                 Returns:
                   List of greedy next-token IDs (one per input position).
             )pbdoc")

        .def("set_alignment_heads", &WhisperWrapper::set_alignment_heads,
             py::arg("heads"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Configure which cross-attention heads to collect when the
                 ``*_with_attention`` APIs are used.

                 The same heads are applied to whichever pool replica handles
                 each request, so this works with multi-replica pools.  Pass an
                 empty list to disable collection.

                 Enabling a non-empty selection also switches the decoder to
                 emit **post-softmax** cross-attention (rows that sum to 1 over
                 encoder frames), which is what timing extractors like
                 ``viterbi_align_words_with_blanks`` expect.

                 Arguments:
                   heads: List of ``(layer_index, head_index)`` pairs.
             )pbdoc")

        .def("prefill_with_attention", &WhisperWrapper::prefill_with_attention,
             py::arg("features"),
             py::arg("prompt"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Like :meth:`prefill`, but also returns the cross-attention row
                 for the first generation position and appends it to
                 ``state.collected_attention``.

                 Arguments:
                   features: Mel spectrogram with shape ``[1, n_mels, chunk_length]``
                     or pre-encoded features from :meth:`encode`.
                   prompt: Token IDs for the full decoder prompt.

                 Returns:
                   A tuple ``(state, logits, attention)`` where ``attention`` has
                   shape ``[1, num_selected_heads, F_enc]``.
             )pbdoc")

        .def("forward_step_with_attention", &WhisperWrapper::forward_step_with_attention,
             py::arg("state"),
             py::arg("token_id"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Like :meth:`forward_step`, but also returns the cross-attention
                 row for this step and appends it to ``state.collected_attention``.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_id: The token ID to feed at the current step.

                 Returns:
                   A tuple ``(logits, attention)`` with shapes
                   ``[1, vocab_size]`` and ``[1, num_selected_heads, F_enc]``.
             )pbdoc")

        .def("forward_step_greedy_with_attention",
             &WhisperWrapper::forward_step_greedy_with_attention,
             py::arg("state"),
             py::arg("token_id"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Like :meth:`forward_step_greedy`, but also returns the
                 cross-attention row for this step and appends it to
                 ``state.collected_attention``.

                 Argmax stays on the GPU; only the attention row and the picked
                 token id leave the device.

                 Arguments:
                   state: A :class:`WhisperDecoderState` (mutated in-place).
                   token_id: The token ID to feed at the current step.

                 Returns:
                   A tuple ``(picked_token_id, attention)`` where ``attention``
                   has shape ``[1, num_selected_heads, F_enc]``.
             )pbdoc")

        .def("unload_model", &WhisperWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this whisper but keep enough runtime context
                 to quickly resume whisper on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &WhisperWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &WhisperWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
