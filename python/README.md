# CTranslate2 — CrisperWhisper Fork

Fork of [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) v4.7.1 with
additional Whisper decoder APIs for **speculative decoding** in
[CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper).

## What's different?

This fork adds five methods to the Whisper model that expose KV-cache
management for speculative decoding:

| Method | Purpose |
|--------|---------|
| `prefill(features, prompt)` | Encode audio + prefill decoder, returning a persistent `WhisperDecoderState` and initial logits |
| `forward_step_greedy(state, token)` | Feed one token, run decoder, return argmax token ID (GPU-side TopK) |
| `forward_batch_greedy(state, tokens)` | Feed N tokens, run decoder in one pass, return N predicted token IDs |
| `forward_step(state, token)` | Feed one token, return full logits (for suppress-token support) |
| `forward_batch(state, tokens)` | Feed N tokens, return full logits matrix |
| `WhisperDecoderState.truncate_to_step(step)` | Roll back KV-cache to a given step (~0.1 ms, avoids full re-prefill) |

These enable O(1)-per-token speculative decoding with KV-cache persistence,
yielding **1.25-1.4x speedup** over standard greedy decoding on a single GPU.

## Installation

```bash
pip install ctranslate2-crisperwhisper
```

The Python import name remains `ctranslate2` — this is a **drop-in replacement**.

> **Note:** If you have the upstream `ctranslate2` package installed, uninstall
> it first: `pip uninstall ctranslate2`

## Building from source

```bash
# 1. Build the C++ library
cd /path/to/CTranslate2
mkdir build && cd build
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# 2. Install the Python package
cd ../python
pip install -e .
```

## License

MIT — same as upstream CTranslate2.
