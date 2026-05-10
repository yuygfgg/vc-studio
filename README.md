# VC Studio

CosyVoice3-based realtime and offline voice conversion studio.

## Setup

```bash
git submodule update --init --recursive
python3 -m pip install -r requirements.txt
```

## Model

Download the CosyVoice3 model files into a local model directory:

```bash
HF_TOKEN=your_token_here huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --include cosyvoice3.yaml flow.pt hift.pt campplus.onnx speech_tokenizer_v3.onnx
```

The model directory must contain:

- `flow.pt`
- `hift.pt`
- `campplus.onnx`
- `speech_tokenizer_v3.onnx`

## Launch

Start the GUI with a model directory:

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512
```

Prefill a voice package:

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --voice-package voices/example.cvvoice
```

Prefill legacy prompt WAV input:

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --prompt prompt.wav
```

Prefill an offline job:

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --voice-package voices/example.cvvoice \
  --source source.wav \
  --output out/vc_streaming.wav \
  --csv out/vc_report.csv
```

Prefill device/runtime choices:

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --device auto \
  --ort-provider auto \
  --prompt-cache-max-mb 1024 \
  --prompt-cache-max-seconds 0 \
  --prompt-cache-dtype auto \
  --prompt-cache-storage device
```

Command-line arguments only prefill GUI fields. See all available options with:

```bash
python3 vc_studio.py --help
```

## Near-Streaming CosyVoice Runtime

VC Studio adapts CosyVoice3 voice conversion to a near-streaming runtime instead of
waiting for a full source utterance. Source audio is converted into speech tokens at
25 tokens per second by `AsyncSourceTokenizer` for files or `MicrophoneSourceTokenizer`
for live input. Tokenization runs in its own thread and may use separate left and
right tokenizer context, then crops the result back to the current chunk.

`run_window_stream` consumes those tokens as soon as enough context is available.
Each inference window contains left history, the current commit region, optional
delayed-commit tokens, mel overlap, and the right context required by the flow
pre-lookahead and HiFT vocoder. Only the current commit region is emitted, so
latency is bounded mostly by `--chunk-sec` plus the configured lookahead rather
than by the full input length.

## Cache Hierarchy

The repository uses several cache layers because CosyVoice3 spends time in
different subsystems:

- Model/session cache: `VCStudioBackend` keeps the loaded `VCOnlyModel` for the
  active model directory, Torch device, ONNX Runtime provider, and CoreML cache
  directory. ONNX Runtime sessions use graph optimizations; CoreML can persist its
  compiled model under `--coreml-cache-dir`.
- Voice package feature cache: `.cvvoice` packages store pre-extracted prompt
  tokens, prompt mel features, speaker embeddings, source metadata, model hashes,
  and fusion weights. This avoids repeating reference feature extraction on every
  run.
- Prompt KV cache: in streaming flow mode, `prepare_stream_context` can call
  `model.flow.prepare_prompt_cache()` once for the target prompt. The cache stores
  per-diffusion-step DiT prompt key/value tensors, input embedding convolution
  tails, prompt inputs, and the final prompt diffusion state. Later chunks run only
  the source side against that prepared prompt cache.
- History KV cache: after a chunk finishes, the flow can keep a bounded KV cache
  for the most recent source history and splice it onto the base prompt cache for
  the next chunk. This avoids recomputing the left-history attention window when
  the next window starts exactly at the previous chunk boundary.
- Vocoder state cache: `--hift-mode stateful` carries HiFT F0 predictor state,
  convolution tails, source excitation phase, sample offset, and ISTFT boundary
  state across chunks. Window mode can also preserve harmonic source phase by
  sample offset.
- DSP/runtime caches: mel filter bases, Hann windows, Hugging Face files, Numba
  artifacts, source token chunks, and realtime playback queues are kept outside
  the core flow cache so repeated work is minimized.

Prompt caches are quality-preserving: VC Studio only enables a prepared prompt KV
cache when the full selected prompt fits the DiT static chunk grid
(`static_chunk_size=50` mel frames) and the configured cache budget. If it does
not fit, the runtime disables that cache and keeps the same prompt quality path
without truncating the prompt. The budget estimates memory from branch count,
diffusion steps, transformer layers, K/V tensors, CFG batch size, attention heads,
frame count, and storage dtype. `--prompt-cache-max-mb`,
`--prompt-cache-max-seconds`, `--prompt-cache-dtype`, and
`--prompt-cache-storage` control whether the cache is allowed, how it is stored,
and whether KV tensors stay on the device or are offloaded to CPU.

## Chunk Consistency

Chunked inference can drift or click if each window is treated as an unrelated
utterance. VC Studio uses several mechanisms to keep neighboring chunks coherent:

- Stable flow coordinates: diffusion noise is sliced by absolute source mel
  offset, and cached source positions keep RoPE positions aligned to source time
  even when a history cache is reused.
- Explicit history and lookahead: each flow window includes left history plus the
  right context required by flow pre-lookahead and HiFT. This gives the current
  commit region enough context without emitting future frames.
- Mel overlap blending: future mel frames from one chunk are saved and blended
  into the start of the next chunk with a cosine crossfade.
- Emitted-history anchoring: regenerated history mel can be replaced by the
  already emitted mel tail, reducing feedback drift between windows.
- Vocoder continuity: stateful HiFT keeps convolution, F0, source phase, and ISTFT
  boundary state. Windowed HiFT uses absolute sample offsets and phase handoff to
  keep harmonic excitation aligned.
- Audio boundary repair: optional waveform crossfade and de-click correction can
  smooth remaining sample-level discontinuities. Optional VAD gating applies
  fades when muting nonspeech regions.

The main quality/latency knobs are `--history-sec`, `--mel-overlap-sec`,
`--delayed-commit-sec`, `--audio-blend-ms`, and `--audio-declick-ms`. Larger
values usually improve joins and stability, while smaller values reduce latency.

## Multi-Prompt Timbre Fusion

Voice packages can contain multiple prompt references for one target voice. During
package creation, each accepted WAV becomes a branch with its own prompt tokens,
prompt mel features, speaker embedding, source hash, duration, and weight metadata.
Fusion weights can be equal, duration based, or manual. Positive raw weights are
normalized, optionally sharpened by `branch_weight_gamma`, and stored with the
package.

Speaker embedding fusion is independent from prompt attention. Each branch
embedding is L2-normalized, combined by the normalized weights, then L2-normalized
again into `fused_speaker_embedding`. In the default runtime path, the dominant
branch supplies prompt tokens and prompt mel features, while the fused embedding
provides the target speaker condition.

For stronger multi-reference prompt behavior, `--enable-grouped-prompt` enables
grouped branch attention. The DiT source path keeps references separate in
attention space: branch prompt KV tensors are stacked with masks, branch weights,
and `attention_temperature`, then attention outputs are mixed per branch instead
of concatenating references into one long prompt.

`--enable-grouped-prompt-cache` is a separate performance option. When it is on
and the full active branch set fits the cache budget, VC Studio prepares grouped
prompt KV once and reuses it. When it is off, or when the cache cannot fit, the
runtime still uses grouped prompt attention and recomputes temporary prompt KV
per diffusion step and transformer layer. This means automatic cache fallback can
make inference slower, but it does not silently switch the quality path back to
dominant-branch prompting or allocate the full reusable grouped cache.

## Performance Tuning

Start with `--flow-context streaming`, `--hift-mode stateful`, prompt KV cache on,
and history KV cache on. Enable `--enable-grouped-prompt` when you want the
multi-reference grouped attention quality path; add `--enable-grouped-prompt-cache`
when you also want its cache and have enough memory. Use `--device cuda` when
available, `--device mps` on Apple Silicon, or `--device cpu` for compatibility.
`--ort-provider auto` selects CUDA, CoreML, or CPU for the speech tokenizer
depending on the installed ONNX Runtime build.

For lower latency, reduce `--chunk-sec`, `--tokenizer-chunk-sec`, and lookahead
settings. For smoother output, increase history, mel overlap, delayed commit, or
audio repair settings. For lower memory use, reduce `--prompt-cache-max-mb`, set
`--prompt-cache-max-seconds`, choose fp16/bf16 prompt cache storage where supported,
or use `--prompt-cache-storage cpu_offload`; if the cache no longer fits, the
runtime keeps quality and runs without that cache.

## Notes

Realtime audio playback and capture require `sounddevice` and a working PortAudio installation.
