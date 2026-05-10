# CosyVoice VC Probe

This workspace is reduced to a CosyVoice voice-conversion-only feasibility probe.
It does not load or run the CosyVoice LLM/TTS path.

## Recommended Model

Use `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` for the current probe. The VC path only
needs the flow, vocoder, speaker embedding, and speech tokenizer files.

```bash
git submodule update --init --recursive

HF_TOKEN=your_token_here huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --include cosyvoice3.yaml flow.pt hift.pt campplus.onnx speech_tokenizer_v3.onnx
```

## Run Streaming Window Probe

```bash
python3 vc_streaming_benchmark.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --source asset/cross_lingual_prompt.wav \
  --prompt asset/zero_shot_prompt.wav \
  --output out/vc_streaming.wav \
  --chunk-sec 2.0 \
  --tokenizer-chunk-sec 2.0 \
  --tokenizer-left-context-sec 0.5 \
  --tokenizer-right-context-sec 0.2 \
  --history-sec 1.0 \
  --mel-overlap-sec 0.4 \
  --delayed-commit-sec 0.0 \
  --hift-mode window \
  --audio-declick-ms 0.0 \
  --audio-blend-ms 0.0 \
  --flow-context streaming \
  --device auto \
  --ort-provider auto
```

`--source` is the content speech to convert. `--prompt` is the target speaker
reference. A 5-15 second clean prompt is a practical starting point.
The flow path runs fixed-size windows. Each window prepends a bounded left
history context, discards the generated history audio, and appends only the new
audio region. `--mel-overlap-sec` makes each window generate a short future mel
overlap; the next window blends that cached overlap with its own opening mel
before vocoding. `--delayed-commit-sec` waits for additional future source
tokens before emitting the current chunk. Those delayed frames are used as model
and vocoder context but are not committed from the current window, which can
improve prosody at the cost of real output latency.

`--hift-mode` selects how the HiFT vocoder runs:

- `window` is the default quality-safe mode. It vocodes each emitted window with
  bounded mel history and discards the generated history audio.
- `stateful` advances fixed-size HiFT caches across chunks instead of
  recomputing vocoder history. It keeps f0 left-conv caches, NSF source phase,
  source STFT tail, up/down/resblock/conv_post caches, and a two-frame ISTFT
  boundary buffer. Cache size is fixed by the model architecture and does not
  grow with stream duration; for the CosyVoice3 0.5B HiFT used here it is about
  0.6 MiB per stream in fp32.

`--audio-declick-ms` and `--audio-blend-ms` default to `0.0`. They are optional
waveform-domain diagnostics and are not required for the stateful HiFT boundary
alignment path.
Flow sampling uses source-global deterministic noise and source-global DiT
positions, so the same absolute source frames receive stable diffusion noise and
RoPE positions across adjacent windows. `--flow-context streaming` keeps the DiT
causal chunk attention mask and enables prompt/history KV caches. If quality is
more important than compute, `--flow-context window-full` disables those KV
caches and uses full attention inside each emitted window while still generating
the audio chunk by chunk.

The source speech tokenizer is also run in a streaming-like background thread.
`--tokenizer-chunk-sec` controls the current source audio span tokenized per
step; it defaults to `--chunk-sec` when omitted. Each tokenizer step can include
extra audio context with `--tokenizer-left-context-sec` and
`--tokenizer-right-context-sec`. The context is used only to stabilize tokenizer
boundaries; only the current chunk's global 25 Hz token range is submitted to
the flow model. Increasing right context can reduce consonant boundary artifacts
but adds real lookahead latency. The per-run summary reports both the original
source duration and the larger tokenizer window duration as
`source_tokenize_audio_seconds` and `source_tokenize_window_audio_seconds`.

`--device auto` selects `cuda`, then `mps`, then `cpu`.
`--ort-provider auto` selects `CUDAExecutionProvider`, then `CoreMLExecutionProvider`, then CPU.
On Apple Silicon, use `--device mps --ort-provider coreml` when both are available.

When CoreML is used, ONNX Runtime writes compiled model artifacts to
`<model-dir>/.ort_coreml_cache` by default. Override it with
`--coreml-cache-dir /path/to/cache`. Delete that cache directory after changing
the ONNX tokenizer file.

## Model Files

The CLI expects these CosyVoice3 files under `--model-dir`:

- `flow.pt`
- `hift.pt`
- `campplus.onnx`
- `speech_tokenizer_v3.onnx`

The LLM weights are not required for this VC probe. The current public MLX
conversion is not used here because it does not provide the VC-only flow,
vocoder, and ONNX tokenizer files needed by this path.
