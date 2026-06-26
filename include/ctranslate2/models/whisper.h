#pragma once

#include <mutex>
#include <tuple>
#include <utility>

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Include log probs of each token in the result
      bool return_logits_vocab = false;

      // Include the probability of the no speech token in the result.
      bool return_no_speech_prob = false;

      // Maximum index of the first predicted timestamp.
      size_t max_initial_timestamp_index = 50;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};
    };

    struct WhisperGenerationResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;
      std::vector<std::vector<StorageView>> logits;
      float no_speech_prob = 0;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    struct WhisperAlignmentResult {
      std::vector<std::pair<dim_t, dim_t>> alignments;
      std::vector<float> text_token_probs;
    };

    struct WhisperDecoderState {
      layers::DecoderState state;
      dim_t current_step = 0;
      // Per-step cross-attention rows captured during incremental
      // decoding.  Each entry has shape ``[1, num_selected_heads,
      // F_enc]`` (post-softmax, so each (batch, head, :) row sums to 1
      // over encoder frames) and lives on the model's device.
      // Populated only by the ``*_with_attention`` APIs.  Index ``k``
      // corresponds to the attention emitted when predicting the
      // (prompt_len+k)-th token, i.e. the row matching the k-th
      // generated token (0-indexed).
      std::vector<StorageView> collected_attention;

      WhisperDecoderState() = default;
      WhisperDecoderState(WhisperDecoderState&&) = default;
      WhisperDecoderState& operator=(WhisperDecoderState&&) = default;

      WhisperDecoderState deep_copy() const {
        WhisperDecoderState copy;
        copy.current_step = current_step;
        for (const auto& kv : state)
          copy.state.emplace(kv.first, StorageView(kv.second));
        copy.collected_attention.reserve(collected_attention.size());
        for (const auto& a : collected_attention)
          copy.collected_attention.emplace_back(a);
        return copy;
      }

      void truncate_to_step(dim_t target_step);
    };

    class WhisperModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class WhisperReplica : public ModelReplica {
    public:
      static std::unique_ptr<WhisperReplica> create_from_model(const Model& model);

      WhisperReplica(const std::shared_ptr<const WhisperModel>& model);

      bool is_multilingual() const {
        return _is_multilingual;
      }

      size_t n_mels() const {
        return _n_mels;
      }

      size_t num_languages() const {
        return _num_languages;
      }

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<std::string>>& prompts,
               const WhisperOptions& options);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<size_t>>& prompts,
               const WhisperOptions& options);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageView features);

      std::vector<WhisperAlignmentResult>
      align(StorageView features,
            const std::vector<size_t>& start_sequence,
            const std::vector<std::vector<size_t>>& text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

      std::pair<WhisperDecoderState, StorageView>
      prefill(StorageView features,
              const std::vector<size_t>& prompt);

      StorageView forward_step(WhisperDecoderState& state,
                               size_t token_id);

      StorageView forward_batch(WhisperDecoderState& state,
                                const std::vector<size_t>& token_ids);

      // ``suppress_tokens`` (main-vocab IDs) are masked to -inf on the
      // device before the argmax, so token suppression stays on the fast
      // greedy path (no full-vocab logits transfer to Python).  An empty
      // list means no suppression.
      size_t forward_step_greedy(WhisperDecoderState& state,
                                 size_t token_id,
                                 const std::vector<size_t>& suppress_tokens = {});

      std::vector<size_t> forward_batch_greedy(WhisperDecoderState& state,
                                               const std::vector<size_t>& token_ids,
                                               const std::vector<size_t>& suppress_tokens = {});

      // Like ``forward_batch`` but also returns the per-position
      // post-softmax cross-attention over encoder frames for the selected
      // alignment heads.  Used by speculative decoding's word-timing path
      // to recover main-model attention for the always-verified token and
      // for verifier-correction tokens in the same batched pass (no extra
      // compute).  ``attention`` is head-averaged and returned on CPU with
      // shape ``[T, F_enc]`` (T = ``token_ids.size()``); ``attention[i]``
      // is the row used to predict the token following ``token_ids[i]``.
      // Requires ``set_alignment_heads([...])`` first.
      std::pair<StorageView, StorageView>
      forward_batch_with_attention(WhisperDecoderState& state,
                                   const std::vector<size_t>& token_ids);

      // Configure which (layer, head) cross-attention rows to collect on
      // subsequent ``*_with_attention`` calls.  Also enables post-softmax
      // attention output on the decoder.  Pass an empty list to disable
      // collection (and revert the decoder to its default raw-attention
      // mode used by ``align()``).
      void set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& heads);

      // Like ``prefill`` but also returns the post-softmax cross-attention
      // row at the first generation position (i.e. the attention emitted
      // when the last prompt token is fed and the model predicts the
      // first new token).  The row is also appended to
      // ``state.collected_attention``.
      std::tuple<WhisperDecoderState, StorageView, StorageView>
      prefill_with_attention(StorageView features,
                             const std::vector<size_t>& prompt);

      // Like ``forward_step`` but also returns the cross-attention row
      // for this step and appends it to ``state.collected_attention``.
      std::pair<StorageView, StorageView>
      forward_step_with_attention(WhisperDecoderState& state,
                                  size_t token_id);

      // Like ``forward_step_greedy`` but also captures the cross-attention
      // row.  The greedy argmax stays on the GPU; only the (small)
      // attention row plus the picked token id leave the device.
      std::pair<size_t, StorageView>
      forward_step_greedy_with_attention(WhisperDecoderState& state,
                                         size_t token_id);

      // Concatenate ``state.collected_attention`` along the time axis,
      // optionally average over the heads dimension, cast to float32,
      // and transfer the result to CPU in a single bulk PCIe copy.
      //
      // Output shape:
      //   * ``average_heads = true``  -> ``[T, F_enc]``
      //   * ``average_heads = false`` -> ``[T, num_heads, F_enc]``
      //
      // Returns an empty StorageView (size 0) when no rows are buffered.
      // ``state`` is not mutated.
      StorageView collected_attention_to_cpu(const WhisperDecoderState& state,
                                             bool average_heads = true) const;

      // Runs a full greedy decode loop with on-GPU argmax and per-step
      // cross-attention capture, all inside a single thread-pool job
      // (no Python round-trip per step).  Used by CrisperWhisper's
      // word-timing pipeline to avoid the dispatch / logits-to-CPU
      // overhead of doing the same thing from Python.
      //
      // ``suppress_tokens`` are masked at every step.
      // ``ban_first_tokens`` are masked **only** on the first step --
      // this is how the hallucination-repair "escape" round forces
      // the model away from the loop-starting token.
      //
      // Stops on the first ``eot_id`` emission or after
      // ``max_new_tokens`` steps.  Returns the new state (with
      // ``collected_attention`` populated) and the generated token
      // list (which may end with ``eot_id`` when EOT was emitted).
      std::pair<WhisperDecoderState, std::vector<size_t>>
      generate_greedy_with_attention(StorageView features,
                                     const std::vector<size_t>& prompt,
                                     size_t max_new_tokens,
                                     size_t eot_id,
                                     const std::vector<size_t>& suppress_tokens,
                                     const std::vector<size_t>& ban_first_tokens);

      // Runs the entire *strict* speculative-decoding loop natively inside
      // a single thread-pool job: this replica is the verifier ("main"),
      // ``draft`` proposes ``num_speculative_tokens`` tokens per round,
      // and the main model verifies them in one batched pass.  KV-caches
      // for both models are kept on the device and only the picked token
      // ids cross to the host -- no per-token Python dispatch.
      //
      // Both models run on *this* worker thread (one CUDA stream), so the
      // draft state and main state never race.  ``draft`` may alias
      // ``*this`` (same-model draft): the two decoder states are
      // independent so this is safe.
      //
      // Token-space translation between the two vocabularies is supplied
      // as lookup tables: ``d2m[draft_id] -> main_id`` and
      // ``m2d[main_id] -> draft_id`` (value -1 means "unmapped").  An
      // *empty* table means the identity mapping (same vocabulary).
      //
      // ``suppress_tokens`` (main-vocab ids) are masked on the device
      // before every main-model argmax (prefill, verify, rollback),
      // exactly matching a non-speculative suppressed decode.  The
      // returned ids are in main-vocab space and include the trailing
      // ``eot_id`` when generation stopped on EOT.
      //
      // Adaptive K: when ``max_speculative_tokens > min_speculative_tokens``
      // the number of tokens drafted per round self-tunes to the draft's
      // acceptance with a symmetric +1/-1 controller -- a round where every
      // drafted token is accepted bumps K up by one, any rejection nudges it
      // down by one -- clamped to ``[min_speculative_tokens,
      // max_speculative_tokens]``.  The controller's K *persists across
      // calls* (stored on the replica) so that, over a chunked transcription,
      // it converges to the per-audio equilibrium (where ~half the rounds
      // fully accept) regardless of the seed.  Pass ``reset_adaptive_state``
      // on the first chunk of a new audio to re-seed K from
      // ``num_speculative_tokens``; pass it false on subsequent chunks to
      // keep converging.  Pass ``max_speculative_tokens == 0`` (the default)
      // to keep K fixed at ``num_speculative_tokens``.
      std::vector<size_t>
      generate_speculative(WhisperReplica& draft,
                           StorageView main_features,
                           StorageView draft_features,
                           const std::vector<size_t>& prompt,
                           size_t num_speculative_tokens,
                           size_t max_length,
                           size_t eot_id,
                           const std::vector<size_t>& suppress_tokens,
                           const std::vector<int32_t>& d2m,
                           const std::vector<int32_t>& m2d,
                           size_t min_speculative_tokens = 0,
                           size_t max_speculative_tokens = 0,
                           bool reset_adaptive_state = true);

    private:
      const std::shared_ptr<const WhisperModel> _model;
      const std::unique_ptr<layers::WhisperEncoder> _encoder;
      const std::unique_ptr<layers::WhisperDecoder> _decoder;

      size_t _sot_id;
      size_t _eot_id;
      size_t _no_timestamps_id;
      size_t _no_speech_id;
      size_t _n_mels;
      size_t _num_languages;
      bool _is_multilingual;

      // Persistent adaptive-K controller state (see generate_speculative).
      // <= 0 means "uninitialised": the next adaptive call seeds it.
      double _spec_k_state = 0.0;

      StorageView maybe_encode(StorageView features);
    };

    class Whisper : public ReplicaPool<WhisperReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      bool is_multilingual() const;
      size_t n_mels() const;
      size_t num_languages() const;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<std::string>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<std::vector<std::pair<std::string, float>>>>
      detect_language(const StorageView& features);

      std::vector<std::future<WhisperAlignmentResult>>
      align(const StorageView& features,
            std::vector<size_t> start_sequence,
            std::vector<std::vector<size_t>> text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

      std::future<std::pair<WhisperDecoderState, StorageView>>
      prefill(const StorageView& features,
              std::vector<size_t> prompt);

      std::future<StorageView>
      forward_step(WhisperDecoderState& state,
                   size_t token_id);

      std::future<StorageView>
      forward_batch(WhisperDecoderState& state,
                    std::vector<size_t> token_ids);

      std::future<size_t>
      forward_step_greedy(WhisperDecoderState& state,
                          size_t token_id,
                          std::vector<size_t> suppress_tokens = {});

      std::future<std::vector<size_t>>
      forward_batch_greedy(WhisperDecoderState& state,
                           std::vector<size_t> token_ids,
                           std::vector<size_t> suppress_tokens = {});

      // Configure the (layer, head) cross-attention rows to collect on
      // subsequent ``*_with_attention`` calls.  The configuration is
      // stored on the pool and re-applied to whichever replica handles
      // each request, so it works correctly with multiple replicas.
      // Pass an empty vector to disable collection.
      void set_alignment_heads(std::vector<std::pair<dim_t, dim_t>> heads);

      std::future<std::tuple<WhisperDecoderState, StorageView, StorageView>>
      prefill_with_attention(const StorageView& features,
                             std::vector<size_t> prompt);

      std::future<std::pair<StorageView, StorageView>>
      forward_step_with_attention(WhisperDecoderState& state,
                                  size_t token_id);

      std::future<std::pair<size_t, StorageView>>
      forward_step_greedy_with_attention(WhisperDecoderState& state,
                                         size_t token_id);

      // Async wrapper around ``WhisperReplica::forward_batch_with_attention``.
      // Returns ``(logits, attention)`` where ``attention`` is the
      // head-averaged per-position cross-attention on CPU ``[T, F_enc]``.
      std::future<std::pair<StorageView, StorageView>>
      forward_batch_with_attention(WhisperDecoderState& state,
                                   std::vector<size_t> token_ids);

      // Async wrapper around ``WhisperReplica::generate_greedy_with_attention``.
      // The whole greedy loop runs inside one replica job, so the Python
      // caller pays a single round-trip per generation segment instead of
      // one per token.  ``ban_first_tokens`` is applied only on the
      // first step (used to break loop-starting tokens during repair).
      std::future<std::pair<WhisperDecoderState, std::vector<size_t>>>
      generate_greedy_with_attention(StorageView features,
                                     std::vector<size_t> prompt,
                                     size_t max_new_tokens,
                                     size_t eot_id,
                                     std::vector<size_t> suppress_tokens,
                                     std::vector<size_t> ban_first_tokens);

      // Async wrapper around ``WhisperReplica::generate_speculative``.
      // The whole strict speculative loop runs inside one job on *this*
      // pool's worker thread; ``draft_pool``'s first replica is driven
      // directly from that same thread (no job is posted to it), so the
      // Python caller pays a single round-trip per generated segment.
      std::future<std::vector<size_t>>
      generate_speculative(Whisper& draft_pool,
                           StorageView main_features,
                           StorageView draft_features,
                           std::vector<size_t> prompt,
                           size_t num_speculative_tokens,
                           size_t max_length,
                           size_t eot_id,
                           std::vector<size_t> suppress_tokens,
                           std::vector<int32_t> d2m,
                           std::vector<int32_t> m2d,
                           size_t min_speculative_tokens = 0,
                           size_t max_speculative_tokens = 0,
                           bool reset_adaptive_state = true);

      // Async wrapper for the bulk attention transfer.  Performs the
      // concat + (optional) head-mean on the device that owns the state,
      // then copies once to CPU.
      std::future<StorageView>
      collected_attention_to_cpu(const WhisperDecoderState& state,
                                 bool average_heads = true);

    private:
      std::vector<std::pair<dim_t, dim_t>> _alignment_heads;
      mutable std::mutex _alignment_heads_mutex;

      std::vector<std::pair<dim_t, dim_t>> get_alignment_heads_copy() const;
    };

  }
}
