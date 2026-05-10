# VC Studio

High performance CosyVoice3 based Realtime Voice Conversion studio.

## Model

```bash
git submodule update --init --recursive

HF_TOKEN=your_token_here huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --include cosyvoice3.yaml flow.pt hift.pt campplus.onnx speech_tokenizer_v3.onnx
```

## Launch VC Studio

```bash
python3 vc_studio.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  --prompt asset/zero_shot_prompt.wav
```

`vc_studio.py` launches the unified GUI. Command-line options only prefill the
GUI fields.

The **Realtime** tab captures the microphone at 16 kHz, converts the speech in
chunks, and plays the generated 24 kHz audio through the selected output device.
Use headphones to avoid feeding the converted voice back into the microphone.
Realtime audio uses `sounddevice`; on systems without PortAudio, install the
system PortAudio package before installing Python requirements.

The **Offline Benchmark** tab runs the same streaming window pipeline on a source
WAV, writes the converted WAV, and can optionally export the per-chunk timing
CSV. This replaces the old command-line benchmark path.

The **Parameters** tab controls shared realtime and offline settings:

- `chunk-sec` controls how much source speech is committed per inference step.
- `tokenizer-chunk-sec` controls the speech-tokenizer step size. Set it to `0`
  to follow `chunk-sec`.
- `tokenizer-left-context-sec` and `tokenizer-right-context-sec` stabilize source
  tokenizer boundaries.
- `history-sec` controls left context passed into the flow window.
- `mel-overlap-sec` blends future mel context across adjacent chunks.
- `delayed-commit-sec` waits for extra source context before emitting a chunk.
- `audio-declick-ms` and `audio-blend-ms` are optional waveform boundary tools.
- `flow-context` selects streaming causal attention or full attention inside the
  current window.
- `hift-mode` selects windowed vocoding or stateful vocoder caches.

`--prompt` is the target speaker reference. A 5-15 second clean prompt is a
practical starting point.

The flow path runs fixed-size windows. Each window prepends a bounded left
history context, discards the generated history audio, and appends only the new
audio region. `mel-overlap-sec` makes each window generate a short future mel
overlap; the next window blends that cached overlap with its own opening mel
before vocoding. `delayed-commit-sec` waits for additional future source tokens
before emitting the current chunk. Those delayed frames are used as model and
vocoder context but are not committed from the current window, which can improve
prosody at the cost of real output latency.

`hift-mode` selects how the HiFT vocoder runs:

- `window` is the quality-safe mode. It vocodes each emitted window with bounded
  mel history and discards the generated history audio.
- `stateful` advances fixed-size HiFT caches across chunks instead of
  recomputing vocoder history. It keeps f0 left-conv caches, NSF source phase,
  source STFT tail, up/down/resblock/conv_post caches, and a two-frame ISTFT
  boundary buffer. Cache size is fixed by the model architecture and does not
  grow with stream duration; for the CosyVoice3 0.5B HiFT used here it is about
  0.6 MiB per stream in fp32.

`audio-declick-ms` and `audio-blend-ms` default to `0.0`. They are optional
waveform-domain diagnostics and are not required for the stateful HiFT boundary
alignment path.
Flow sampling uses source-global deterministic noise and source-global DiT
positions, so the same absolute source frames receive stable diffusion noise and
RoPE positions across adjacent windows. `flow-context=streaming` keeps the DiT
causal chunk attention mask and enables prompt/history KV caches. If quality is
more important than compute, `flow-context=window-full` disables those KV
caches and uses full attention inside each emitted window while still generating
the audio chunk by chunk.

The source speech tokenizer is also run in a streaming-like background thread.
`tokenizer-chunk-sec` controls the current source audio span tokenized per step;
set it to `0` to follow `chunk-sec`. Each tokenizer step can include extra audio
context with `tokenizer-left-context-sec` and `tokenizer-right-context-sec`. The
context is used only to stabilize tokenizer boundaries; only the current chunk's
global 25 Hz token range is submitted to the flow model. Increasing right
context can reduce consonant boundary artifacts but adds real lookahead latency.
The offline summary reports both the original source duration and the larger
tokenizer window duration as
`source_tokenize_audio_seconds` and `source_tokenize_window_audio_seconds`.

`--device auto` selects `cuda`, then `mps`, then `cpu`.
`--ort-provider auto` selects `CUDAExecutionProvider`, then `CoreMLExecutionProvider`, then CPU.
On Apple Silicon, use `--device mps --ort-provider coreml` when both are available.

## Model Files

The app expects these CosyVoice3 files under the model directory:

- `flow.pt`
- `hift.pt`
- `campplus.onnx`
- `speech_tokenizer_v3.onnx`
