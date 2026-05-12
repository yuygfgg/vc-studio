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
  --lavasr-lowpass-hz 7800 \
  --prompt-cache-max-mb 1024 \
  --prompt-cache-max-seconds 0 \
  --prompt-cache-dtype auto \
  --prompt-cache-storage device \
  --prompt-runtime-policy auto
```

Command-line arguments only prefill GUI fields. See all available options with:

```bash
python3 vc_studio.py --help
```

## Near-Streaming CosyVoice Runtime

VC Studio runs CosyVoice3 voice conversion as a bounded-window streaming
approximation. Let the speech tokenizer emit source tokens at rate
$r_{\mathrm{tok}} = 25$ tokens per second, and let $q$ denote the model
token-to-mel expansion ratio. For chunk index $k$, define the committed token
interval as

$$
C_k = [s_k, s_k + n_c),
$$

where $n_c$ is derived from `--chunk-sec`. The flow inference window is extended
with left history $n_h$, delayed-commit lookahead $n_d$, and tokenizer/right
context $n_r$:

$$
W_k = [s_k - n_h,\ s_k + n_c + n_d + n_r).
$$

Only $C_k$ is emitted. The remaining context is used to stabilize token
boundaries, flow attention, and HiFT synthesis, and is then discarded or retained
only as cache state. The absolute source mel offset for the first emitted frame is

$$
o_k = q\,s_k.
$$

This offset is propagated into diffusion noise indexing, RoPE position indexing,
and source-history cache construction. Consequently, each chunk is evaluated in a
local computational window while preserving global source-time coordinates.

## Cache Hierarchy

The runtime separates immutable model state, package-level reference state, DiT
attention state, and waveform synthesis state.

- Model/session cache: `VCStudioBackend` retains one loaded `VCOnlyModel` per
  model directory, Torch device, ONNX Runtime provider, and CoreML cache
  directory. ONNX Runtime sessions are reused with provider-specific graph
  optimizations; CoreML may persist compiled artifacts under `--coreml-cache-dir`.
- Voice package cache: `.cvvoice` stores reference tokens, prompt mel features,
  speaker embeddings, branch weights, source metadata, model hashes, and optional
  soft prompt tensors. This makes package loading independent of reference WAV
  feature extraction.
- Prompt KV cache: for a prepared prompt of length $T_p$, each diffusion step
  stores DiT prompt key/value tensors and the input-embedding cache required to
  continue the source side without recomputing the prompt prefix.
- History KV cache: after a chunk has been synthesized, the most recent source
  history of length $T_h$ can be represented as attention KV state and concatenated
  with the base prompt cache for the next chunk.
- Vocoder state cache: stateful HiFT carries predictor state, convolution tails,
  harmonic source phase, sample offset, and ISTFT boundary state across chunks.
- DSP/runtime caches: mel bases, analysis windows, source-token buffers, Numba
  artifacts, downloaded model files, and playback queues are maintained outside
  the DiT cache hierarchy.

Let $S$ be the number of diffusion steps, $L$ the number of DiT transformer
blocks, $C$ the classifier-free-guidance batch factor, $H$ the number of attention
heads, $d_h$ the per-head dimension, $b$ the number of bytes per stored scalar,
and $T$ the cached prompt length in mel frames. A single-prompt KV cache has
approximate storage

$$
M_{\mathrm{single}}
\approx
2SLCHTd_hb,
$$

where the factor $2$ accounts for keys and values. For grouped prompt attention
with active branch lengths $\{T_i\}_{i=1}^{B}$, the corresponding upper bound is

$$
M_{\mathrm{grouped}}
\approx
2SLCHd_hb \sum_{i=1}^{B} T_i.
$$

For a soft prompt with configured length $T_s$,

$$
M_{\mathrm{soft}}
\approx
2SLCHT_s d_hb.
$$

Prompt-cache admission is quality preserving. A cache is prepared only when the
complete selected prompt satisfies the static DiT cache grid and the configured
memory policy. If the cache is not admissible, VC Studio disables that cache and
executes the same conditioning path without truncating the prompt. The relevant
controls are `--prompt-cache-max-mb`, `--prompt-cache-max-seconds`,
`--prompt-cache-dtype`, and `--prompt-cache-storage`.

## Chunk Consistency

The streaming approximation introduces two consistency constraints. First, a
chunk must be evaluated at the same positional coordinates it would have occupied
in a full-utterance run. Second, the waveform boundary between adjacent emitted
chunks must be locally smooth.

For flow diffusion, the random noise tensor is indexed by absolute source mel
position:

$$
\epsilon_k[j] = \epsilon[o_k + j],
\qquad 0 \le j < |W_k|q.
$$

The same absolute offset determines RoPE positions for source frames:

$$
p_k[j] = o_k + j.
$$

When history KV is reused, these positions are reconstructed relative to the
prompt and history prefix so that cached and newly computed states remain
coordinate-compatible.

For mel overlap, let $M$ be the overlap length, $u_j$ the saved tail from the
previous chunk, and $v_j$ the regenerated head of the current chunk. VC Studio
uses a cosine interpolation weight

$$
\alpha_j =
\frac{1}{2}\left(1 - \cos\frac{\pi j}{M - 1}\right),
\qquad 0 \le j < M,
$$

and emits

$$
\hat{v}_j = (1 - \alpha_j)u_j + \alpha_j v_j.
$$

Emitted-history anchoring can replace regenerated history frames with the
previously emitted mel tail before the commit region is selected. This constrains
the current window to the already audible trajectory and reduces autoregressive
drift across chunk boundaries.

HiFT continuity is handled by carrying a synthesis state $s_k$:

$$
s_{k+1} = F_{\mathrm{HiFT}}(s_k,\ \hat{y}_k),
$$

where $s_k$ contains F0 predictor state, convolution tails, excitation phase,
sample offset, and ISTFT boundary state. Optional waveform de-clicking and
crossfade operate after vocoder synthesis. Optional VAD gating applies a smooth
amplitude envelope to nonspeech regions. LavaSR bandwidth extension is applied as
a post-processor: the converted waveform is lowpass-filtered, resampled to the
LavaSR input rate, enhanced, and emitted at the higher output sample rate.

## Multi-Prompt Timbre Fusion

Assume a voice package contains $B$ accepted references. Reference $i$ provides
prompt tokens $z_i$, prompt mel features $y_i$, a speaker embedding $e_i$, and a
nonnegative raw fusion weight $a_i$. Raw weights are normalized and optionally
sharpened by `branch_weight_gamma` $\gamma$:

$$
\pi_i =
\frac{a_i}{\sum_{j=1}^{B} a_j},
\qquad
w_i =
\frac{\pi_i^\gamma}{\sum_{j=1}^{B} \pi_j^\gamma}.
$$

The package-level speaker embedding is computed independently from prompt
attention:

$$
\bar{e} =
\mathrm{normalize}
\left(
  \sum_{i=1}^{B} w_i\mathrm{normalize}(e_i)
\right).
$$

This common representation supports three runtime conditioning policies.

### Dominant Branch Mode

Dominant mode selects a single prompt branch

$$
d = \arg\max_i w_i,
$$

with deterministic lowest-index tie breaking. Runtime conditioning uses
$(z_d, y_d, \bar{e})$. This mode has the lowest prompt-attention cost, but the
prompt token and prompt mel context are sampled from one reference only. Its
conditioning operator can be summarized as

$$
\mathcal{C}_{\mathrm{dom}} =
\left(z_d,\ y_d,\ \bar{e}\right).
$$

Thus additional references influence the speaker embedding but not the prompt
sequence attended by the DiT source path.

### Grouped Branch Attention Mode

Grouped mode retains branch identity in attention space. For transformer layer
$\ell$, each branch produces prompt key/value tensors
$(K_{i,\ell}, V_{i,\ell})$. For a source query matrix $Q_\ell$, branch-local
attention is evaluated as

$$
A_{i,\ell} =
\mathrm{softmax}
\left(
  \frac{Q_\ell K_{i,\ell}^{\top}}{\sqrt{d_h}\,\tau} + M_i
\right)V_{i,\ell},
$$

where $M_i$ is the branch mask, $d_h$ is the attention head dimension, and
$\tau$ is `attention_temperature`. The grouped prompt contribution is the
weighted mixture

$$
A_{\ell}^{\mathrm{grouped}} =
\sum_{i=1}^{B} w_i A_{i,\ell}.
$$

This avoids concatenating references into a single prompt and preserves
per-branch masks and weights. Its runtime and cache cost are proportional to the
total active prompt length $\sum_{i=1}^{B} T_i$. If a grouped KV cache is not
admissible, VC Studio recomputes grouped prompt KV online while preserving the
grouped conditioning operator.

### Soft Prompt Mode

Soft prompt mode adds a package-specific distillation step and stores the result
as `soft_prompt_v1`. Soft prompt training is enabled by default in the GUI. The
package stores:

- `soft_prompt_mu`: the token-derived decoder conditioning after expansion to
  mel-frame rate.
- `soft_prompt_feat`: a fixed-length mel prompt condition.
- `soft_speaker_embedding`: the normalized weighted speaker embedding.

Let $C_T(\cdot)$ denote center-cropping references longer than the soft prompt
length $T$ while leaving shorter references at their native length. The token
path maps each reference token sequence to a mel-rate decoder conditioning
sequence:

$$
m_i =
\mathrm{repeat\_interleave}
\left(
  \mathrm{pre\_lookahead}
  \left(
    \mathrm{input\_embedding}(z_i)
  \right),
  q
\right).
$$

The soft prompt initialization uses only valid native frames from each reference
and normalizes by the branch weights that cover each frame:

$$
\mu_0[\tau] =
\frac{\sum_{i:\tau < |C_T(m_i)|} w_i C_T(m_i)[\tau]}
     {\sum_{i:\tau < |C_T(m_i)|} w_i},
\qquad
\quad
f_0[\tau] =
\frac{\sum_{i:\tau < |C_T(y_i)|} w_i C_T(y_i)[\tau]}
     {\sum_{i:\tau < |C_T(y_i)|} w_i},
\qquad
e_0 = \bar{e}.
$$

Over-length references are croppedi into the canonical interval of the soft prompt.
Shorter references keep their native frame sequence and simply stop contributing after
their real length. No single reference is selected as the sole prompt carrier.

Distillation optimizes additive residuals on both decoder-conditioning prompt
streams:

$$
\mu_{\mathrm{soft}} = \mu_0 + \Delta_\mu,
\qquad
f_{\mathrm{soft}} = f_0 + \Delta_f.
$$

All CosyVoice model parameters are frozen. The speaker embedding $e_0$ is also
frozen in the current implementation. For a sampled source token window $x$, a
sampled diffusion time $t$, and a distillation layer $\ell$, the teacher is
grouped branch attention and the student is the fixed soft prompt path:

$$
h_T =
H_{\ell}^{\mathrm{grouped}}
\left(
  x,\ t;\ \{\mathrm{window}_T(m_i),\mathrm{window}_T(y_i),\ e_i,\ w_i\}_{i=1}^{B}
\right),
$$

$$
h_S =
H_{\ell}^{\mathrm{soft}}
\left(
  x,\ t;\ \mu_0 + \Delta_\mu,\ f_0 + \Delta_f,\ e_0
\right).
$$

The optimization target is

$$
\min_{\Delta_\mu}
\mathbb{E}_{x,t}
\left[
  \left\lVert h_S^{\mathrm{src}} - h_T^{\mathrm{src}} \right\rVert_2^2 +
  \lambda \left(\left\lVert \Delta_\mu \right\rVert_2^2 + \left\lVert \Delta_f \right\rVert_2^2\right) +
  \beta \left(\left\lVert D\Delta_\mu \right\rVert_2^2 + \left\lVert D\Delta_f \right\rVert_2^2\right)
\right],
$$

where $D$ is the first-order finite-difference operator along the prompt-time
axis. The superscript $\mathrm{src}$ denotes the source-window slice of the
hidden state; the prompt prefix is not used as a reconstruction target. Source
training windows are sampled from the weighted reference set, and teacher prompt
windows preserve native reference length unless a reference exceeds $T$, in which
case a crop is sampled. Therefore, adding reference audio increases the empirical
support of the speaker adaptation problem while the inference-time prompt length
remains bounded by $T$.

The resulting runtime complexities are

$$
\mathcal{O}_{\mathrm{dom}} = \mathcal{O}(T_d),
\qquad
\mathcal{O}_{\mathrm{grouped}} = \mathcal{O}\left(\sum_{i=1}^{B} T_i\right),
\qquad
\mathcal{O}_{\mathrm{soft}} = \mathcal{O}(T),
$$

where $T_d$ is the dominant-branch prompt length and $T_i$ are per-branch prompt
lengths. Under the intended operating regime, `soft_prompt_v1` provides the most
expressive target-speaker conditioning: it uses multi-reference evidence during
offline optimization, avoids the prompt-length bottleneck of a single selected
reference, and exposes a fixed-cost prompt during inference.

## Performance Tuning

The default operating point is `--flow-context streaming`, `--hift-mode stateful`,
prompt KV caching enabled, history KV caching enabled, and
`--prompt-runtime-policy auto`. In this configuration, packages with
`soft_prompt_v1` use the fixed-cost soft prompt path; legacy packages use the
configured dominant or grouped fallback policy.

Let $n_c$ be the committed token count, $n_d$ the delayed-commit token count,
$n_r$ the right-context token count, and $r_{\mathrm{tok}}$ the tokenizer rate.
Ignoring hardware queueing and model execution time, the algorithmic lookahead is
approximately

$$
L_{\mathrm{lookahead}}
\approx
\frac{n_d + n_r}{r_{\mathrm{tok}}}.
$$

Reducing `--chunk-sec`, `--tokenizer-chunk-sec`, `--delayed-commit-sec`, and
right-context settings decreases latency, but it also reduces the context
available to the tokenizer, flow pre-lookahead, and vocoder. Increasing
`--history-sec`, `--mel-overlap-sec`, `--delayed-commit-sec`,
`--audio-blend-ms`, or `--audio-declick-ms` increases boundary stability at the
cost of additional compute, memory, or latency.

The prompt runtime policy controls the speaker-conditioning operator:

- `auto`: select `soft_prompt_v1` when present; otherwise use the configured
  legacy package behavior.
- `soft`: require the distilled soft prompt tensors.
- `grouped`: evaluate grouped branch attention.
- `dominant`: evaluate the dominant branch with the fused speaker embedding.

Prompt-cache memory is controlled by `--prompt-cache-max-mb`,
`--prompt-cache-max-seconds`, `--prompt-cache-dtype`, and
`--prompt-cache-storage`. Reducing the budget may disable a cache, but cache
rejection does not alter the selected conditioning operator. It only changes
whether the corresponding prompt state is precomputed or recomputed.

## Notes

Realtime audio playback and capture require `sounddevice` and a working PortAudio installation.
