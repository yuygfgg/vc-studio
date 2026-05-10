#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import threading
import time
import warnings
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/tmp/cosyvoice_hf_cache")
warnings.filterwarnings(
    "ignore",
    message=".*LoRACompatibleLinear.*PEFT backend.*",
    category=FutureWarning,
)

from cosyvoice.vc.audio import mel_spectrogram_24000, read_mono, write_wav
from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 chunked VC streaming benchmark")
    parser.add_argument("--model-dir", required=True, help="Directory containing CosyVoice3 VC model files")
    parser.add_argument("--source", required=True, help="Source speech wav to convert")
    parser.add_argument("--prompt", required=True, help="Target speaker reference wav")
    parser.add_argument("--output", default="out/vc_streaming.wav", help="Output wav path")
    parser.add_argument("--csv", default=None, help="Optional CSV path for per-chunk timings")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument(
        "--ort-provider",
        default="cpu",
        choices=["auto", "cpu", "cuda", "coreml"],
        help="ONNX Runtime provider for speech tokenizer",
    )
    parser.add_argument("--coreml-cache-dir", default=None, help="Directory for ONNX Runtime CoreML compiled model cache")
    parser.add_argument("--chunk-sec", type=float, default=2.0, help="Source chunk size in seconds")
    parser.add_argument("--tokenizer-chunk-sec", type=float, default=None, help="Source tokenizer input chunk size in seconds")
    parser.add_argument(
        "--tokenizer-left-context-sec",
        type=float,
        default=0.5,
        help="Past audio context prepended to each source tokenizer chunk",
    )
    parser.add_argument(
        "--tokenizer-right-context-sec",
        type=float,
        default=0.2,
        help="Future audio lookahead appended to each source tokenizer chunk",
    )
    parser.add_argument("--history-sec", type=float, default=1.0, help="Left history context size in seconds")
    parser.add_argument(
        "--mel-overlap-sec",
        type=float,
        default=0.4,
        help="Future mel overlap used for cosine blending between windows",
    )
    parser.add_argument(
        "--delayed-commit-sec",
        type=float,
        default=0.0,
        help="Extra future source context to wait for before emitting each chunk",
    )
    parser.add_argument(
        "--audio-declick-ms",
        type=float,
        default=0.0,
        help="Boundary de-click smoothing in milliseconds; set to 0 to disable",
    )
    parser.add_argument(
        "--audio-blend-ms",
        type=float,
        default=0.0,
        help="Optional waveform crossfade at chunk boundaries in milliseconds; keep short to avoid smearing consonants",
    )
    parser.add_argument(
        "--flow-context",
        default="streaming",
        choices=["streaming", "window-full"],
        help="DiT context mode: streaming uses causal chunk attention and KV caches; window-full uses full attention inside each window",
    )
    parser.add_argument(
        "--hift-mode",
        default="window",
        choices=["window", "stateful"],
        help="HiFT vocoder mode: window recomputes bounded mel history; stateful advances fixed-size vocoder caches",
    )
    parser.add_argument(
        "--disable-prompt-kv-cache",
        action="store_true",
        help="Disable DiT prompt KV cache in streaming context and use the original full estimator path",
    )
    parser.add_argument(
        "--disable-history-kv-cache",
        action="store_true",
        help="Disable bounded DiT KV cache for the left history context",
    )
    args = parser.parse_args()

    model = VCOnlyModel(
        args.model_dir,
        device=args.device,
        ort_provider=args.ort_provider,
        coreml_cache_dir=args.coreml_cache_dir,
    )
    chunk_tokens = max(1, round(args.chunk_sec * 25))
    tokenizer_chunk_sec = args.tokenizer_chunk_sec if args.tokenizer_chunk_sec is not None else args.chunk_sec
    history_tokens = align_history_tokens(args.history_sec, model)
    overlap_tokens = align_overlap_tokens(args.mel_overlap_sec)
    delayed_commit_tokens = align_delayed_commit_tokens(args.delayed_commit_sec)
    audio_declick_samples = align_audio_declick_samples(args.audio_declick_ms, model.sample_rate)
    max_audio_blend_samples = align_audio_blend_samples(args.audio_blend_ms, model.sample_rate)
    flow_streaming = args.flow_context == "streaming"
    use_prompt_kv_cache = flow_streaming and not args.disable_prompt_kv_cache
    history_cache_enabled = (
        flow_streaming
        and not args.disable_history_kv_cache
        and use_prompt_kv_cache
        and is_static_cache_aligned(history_tokens, model)
    )

    prompt_prepare_start = time.perf_counter()
    prompt_token, prompt_feat, embedding = prepare_prompt_inputs(model, args.prompt)
    if use_prompt_kv_cache:
        prompt_token, prompt_feat = trim_prompt_to_static_cache(prompt_token, prompt_feat, model)
    sync_device(model.device)
    prompt_prepare_seconds = time.perf_counter() - prompt_prepare_start

    source_tokenizer = AsyncSourceTokenizer(
        model,
        args.source,
        tokenizer_chunk_sec,
        left_context_sec=args.tokenizer_left_context_sec,
        right_context_sec=args.tokenizer_right_context_sec,
    )
    source_tokenizer.start()

    prompt_cache_len = 0
    prompt_cache_steps = None
    prompt_cache_prepare_seconds = 0.0
    if use_prompt_kv_cache:
        cache_start = time.perf_counter()
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=model.device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=model.device)
        prompt_cache_len, prompt_cache_steps = model.flow.prepare_prompt_cache(
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=flow_streaming,
        )
        sync_device(model.device)
        prompt_cache_prepare_seconds = time.perf_counter() - cache_start

    try:
        speech, rows = run_window_stream(
            model=model,
            source_stream=source_tokenizer,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            chunk_tokens=chunk_tokens,
            history_tokens=history_tokens,
            overlap_tokens=overlap_tokens,
            delayed_commit_tokens=delayed_commit_tokens,
            audio_declick_samples=audio_declick_samples,
            max_audio_blend_samples=max_audio_blend_samples,
            flow_streaming=flow_streaming,
            hift_mode=args.hift_mode,
            use_prompt_kv_cache=use_prompt_kv_cache,
            use_history_kv_cache=history_cache_enabled,
            prompt_cache_len=prompt_cache_len,
            prompt_cache_steps=prompt_cache_steps,
        )
    finally:
        source_tokenizer.join()
    source_stats = source_tokenizer.stats()

    write_wav(args.output, speech, model.sample_rate)
    if args.csv is not None:
        write_rows(args.csv, rows)

    source_tokens = source_stats["tokens"]
    source_seconds = source_tokens / 25.0
    infer_compute_seconds = sum(row["compute_seconds"] for row in rows)
    pipeline_compute_seconds = infer_compute_seconds + source_stats["tokenize_seconds"]
    token_wait_seconds = sum(row["token_wait_seconds"] for row in rows)
    wall_pipeline_seconds = rows[-1]["wall_end_seconds"] if rows else source_stats["wall_seconds"]
    max_chunk_seconds = max((row["compute_seconds"] for row in rows), default=0.0)
    avg_infer_rtf = infer_compute_seconds / source_seconds if source_seconds > 0 else 0.0
    avg_pipeline_rtf = pipeline_compute_seconds / source_seconds if source_seconds > 0 else 0.0
    wall_rtf = wall_pipeline_seconds / source_seconds if source_seconds > 0 else 0.0
    print(f"device={model.device}")
    print(f"flow_context={args.flow_context}")
    print(f"hift_mode={args.hift_mode}")
    print(f"prompt_kv_cache={use_prompt_kv_cache}")
    print(f"history_kv_cache={history_cache_enabled}")
    print(f"source_tokens={source_tokens} source_seconds={source_seconds:.3f}")
    print(f"prompt_tokens={prompt_token.shape[1]} prompt_seconds={prompt_token.shape[1] / 25.0:.3f}")
    print(f"prompt_cache_mel={prompt_cache_len} prompt_cache_seconds={prompt_cache_len / 50.0:.3f}")
    print(f"chunk_tokens={chunk_tokens} chunk_seconds={chunk_tokens / 25.0:.3f}")
    print(f"tokenizer_chunk_seconds={tokenizer_chunk_sec:.3f}")
    print(f"tokenizer_left_context_seconds={args.tokenizer_left_context_sec:.3f}")
    print(f"tokenizer_right_context_seconds={args.tokenizer_right_context_sec:.3f}")
    print(f"history_tokens={history_tokens} history_seconds={history_tokens / 25.0:.3f}")
    print(f"mel_overlap_tokens={overlap_tokens} mel_overlap_seconds={overlap_tokens / 25.0:.3f}")
    print(f"delayed_commit_tokens={delayed_commit_tokens} delayed_commit_seconds={delayed_commit_tokens / 25.0:.3f}")
    print(f"audio_declick_samples={audio_declick_samples} audio_declick_ms={audio_declick_samples / model.sample_rate * 1000:.3f}")
    print(f"max_audio_blend_samples={max_audio_blend_samples} audio_blend_ms={max_audio_blend_samples / model.sample_rate * 1000:.3f}")
    print(f"prompt_prepare_seconds={prompt_prepare_seconds:.3f}")
    print(f"prompt_cache_prepare_seconds={prompt_cache_prepare_seconds:.3f}")
    print(f"source_tokenize_chunks={source_stats['chunks']}")
    print(f"source_tokenize_audio_seconds={source_stats['audio_seconds']:.3f}")
    print(f"source_tokenize_window_audio_seconds={source_stats['window_audio_seconds']:.3f}")
    print(f"source_tokenize_read_seconds={source_stats['read_seconds']:.3f}")
    print(f"source_tokenize_compute_seconds={source_stats['tokenize_seconds']:.3f}")
    print(f"source_tokenize_wall_seconds={source_stats['wall_seconds']:.3f}")
    print(f"stream_token_wait_seconds={token_wait_seconds:.3f}")
    print(f"stream_infer_compute_seconds={infer_compute_seconds:.3f}")
    print(f"stream_pipeline_compute_seconds={pipeline_compute_seconds:.3f}")
    print(f"stream_wall_seconds={wall_pipeline_seconds:.3f}")
    print(f"avg_infer_compute_rtf={avg_infer_rtf:.3f}")
    print(f"avg_pipeline_compute_rtf={avg_pipeline_rtf:.3f}")
    print(f"wall_stream_rtf={wall_rtf:.3f}")
    print(f"max_chunk_compute_seconds={max_chunk_seconds:.3f}")
    print(f"chunks={len(rows)}")
    print(Path(args.output).resolve())


def prepare_prompt_inputs(
    model: VCOnlyModel,
    prompt_wav: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_16k = read_mono(prompt_wav, 16000)
    prompt_24k = read_mono(prompt_wav, model.sample_rate)

    prompt_token = model.features.speech_token(prompt_16k).to(model.device)
    prompt_feat = mel_spectrogram_24000(prompt_24k).squeeze(0).transpose(0, 1).unsqueeze(0).to(model.device)
    prompt_token, prompt_feat = align_prompt_token_feat(prompt_token, prompt_feat, model.token_mel_ratio)
    embedding = model.features.speaker_embedding(prompt_16k).to(model.device)
    return prompt_token, prompt_feat, embedding


def trim_prompt_to_static_cache(
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    model: VCOnlyModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_chunk_mel = model.flow.decoder.estimator.static_chunk_size
    token_multiple = max(1, static_chunk_mel // model.token_mel_ratio)
    token_frames = (prompt_token.shape[1] // token_multiple) * token_multiple
    if token_frames <= 0:
        return prompt_token, prompt_feat
    mel_frames = token_frames * model.token_mel_ratio
    return prompt_token[:, :token_frames], prompt_feat[:, :mel_frames]


def align_history_tokens(history_sec: float, model: VCOnlyModel) -> int:
    if history_sec <= 0:
        return 0
    return max(1, round(history_sec * 25))


def align_overlap_tokens(overlap_sec: float) -> int:
    if overlap_sec <= 0:
        return 0
    return max(1, round(overlap_sec * 25))


def align_delayed_commit_tokens(delayed_commit_sec: float) -> int:
    if delayed_commit_sec <= 0:
        return 0
    return max(1, round(delayed_commit_sec * 25))


def align_audio_declick_samples(audio_declick_ms: float, sample_rate: int) -> int:
    if audio_declick_ms <= 0:
        return 0
    return max(1, round(audio_declick_ms * sample_rate / 1000))


def align_audio_blend_samples(audio_blend_ms: float, sample_rate: int) -> int:
    if audio_blend_ms <= 0:
        return 0
    return max(1, round(audio_blend_ms * sample_rate / 1000))


def is_static_cache_aligned(history_tokens: int, model: VCOnlyModel) -> bool:
    if history_tokens <= 0:
        return False
    static_chunk_mel = model.flow.decoder.estimator.static_chunk_size
    return (history_tokens * model.token_mel_ratio) % static_chunk_mel == 0


class AsyncSourceTokenizer:
    token_rate = 25.0

    def __init__(
        self,
        model: VCOnlyModel,
        source_wav: str | Path,
        chunk_sec: float,
        left_context_sec: float = 0.0,
        right_context_sec: float = 0.0,
    ):
        if chunk_sec <= 0:
            raise ValueError("tokenizer chunk size must be positive")
        if left_context_sec < 0 or right_context_sec < 0:
            raise ValueError("tokenizer context sizes must be non-negative")
        if chunk_sec + left_context_sec + right_context_sec > 30:
            raise ValueError("tokenizer chunk plus context must be <= 30 seconds")
        self.model = model
        self.source_wav = source_wav
        self.chunk_samples = max(1, round(chunk_sec * 16000))
        self.left_context_samples = round(left_context_sec * 16000)
        self.right_context_samples = round(right_context_sec * 16000)
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._token_chunks: list[torch.Tensor] = []
        self._token_count = 0
        self._done = False
        self._error: BaseException | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._read_seconds = 0.0
        self._tokenize_seconds = 0.0
        self._audio_samples = 0
        self._window_audio_samples = 0
        self._chunks = 0

    @property
    def started_at(self) -> float:
        if self._started_at is None:
            raise RuntimeError("source tokenizer has not been started")
        return self._started_at

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("source tokenizer has already been started")
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="source-tokenizer", daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
        self._raise_if_error()

    def wait_until(self, token_count: int) -> float:
        wait_start = time.perf_counter()
        with self._condition:
            while self._token_count < token_count and not self._done and self._error is None:
                self._condition.wait(timeout=0.1)
            self._raise_if_error_locked()
        return time.perf_counter() - wait_start

    def token_count(self) -> int:
        with self._condition:
            return self._token_count

    def is_done(self) -> bool:
        with self._condition:
            return self._done

    def get(self, start_token: int, end_token: int) -> torch.Tensor:
        if start_token < 0 or end_token < start_token:
            raise ValueError(f"invalid token range: {start_token}:{end_token}")
        self.wait_until(end_token)
        with self._condition:
            if end_token > self._token_count:
                raise RuntimeError(
                    f"requested token range {start_token}:{end_token}, "
                    f"but only {self._token_count} source tokens are available"
                )
            pieces = []
            offset = 0
            for chunk in self._token_chunks:
                chunk_end = offset + chunk.shape[1]
                if chunk_end > start_token and offset < end_token:
                    left = max(0, start_token - offset)
                    right = min(chunk.shape[1], end_token - offset)
                    pieces.append(chunk[:, left:right])
                offset = chunk_end
                if offset >= end_token:
                    break
        if not pieces:
            return torch.empty(1, 0, dtype=torch.int32)
        return torch.cat(pieces, dim=1)

    def stats(self) -> dict:
        with self._condition:
            finished_at = self._finished_at if self._finished_at is not None else time.perf_counter()
            started_at = self._started_at if self._started_at is not None else finished_at
            return {
                "tokens": self._token_count,
                "chunks": self._chunks,
                "audio_seconds": self._audio_samples / 16000.0,
                "window_audio_seconds": self._window_audio_samples / 16000.0,
                "read_seconds": self._read_seconds,
                "tokenize_seconds": self._tokenize_seconds,
                "wall_seconds": finished_at - started_at,
            }

    def _run(self) -> None:
        try:
            read_start = time.perf_counter()
            wav_16k = read_mono(self.source_wav, 16000)
            read_seconds = time.perf_counter() - read_start
            with self._condition:
                self._read_seconds = read_seconds

            total_samples = wav_16k.shape[1]
            for start in range(0, total_samples, self.chunk_samples):
                current_end = min(start + self.chunk_samples, total_samples)
                if current_end <= start:
                    continue
                window_start = max(0, start - self.left_context_samples)
                window_end = min(total_samples, current_end + self.right_context_samples)
                chunk = wav_16k[:, window_start:window_end]
                tokenize_start = time.perf_counter()
                token = self.model.features.speech_token(chunk).cpu()
                tokenize_seconds = time.perf_counter() - tokenize_start
                token = token.to(dtype=torch.int32)
                token = self._crop_current_tokens(
                    token=token,
                    window_start_sample=window_start,
                    current_start_sample=start,
                    current_end_sample=current_end,
                )
                with self._condition:
                    self._chunks += 1
                    self._audio_samples += current_end - start
                    self._window_audio_samples += window_end - window_start
                    self._tokenize_seconds += tokenize_seconds
                    if token.shape[1] > 0:
                        self._token_chunks.append(token)
                        self._token_count += token.shape[1]
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._done = True
                self._finished_at = time.perf_counter()
                self._condition.notify_all()
        else:
            with self._condition:
                self._done = True
                self._finished_at = time.perf_counter()
                self._condition.notify_all()

    def _raise_if_error(self) -> None:
        with self._condition:
            self._raise_if_error_locked()

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("source tokenizer failed") from self._error

    def _crop_current_tokens(
        self,
        token: torch.Tensor,
        window_start_sample: int,
        current_start_sample: int,
        current_end_sample: int,
    ) -> torch.Tensor:
        if token.shape[1] == 0:
            return token
        window_token_start = self._sample_to_token(window_start_sample)
        current_token_start = self._sample_to_token(current_start_sample)
        current_token_end = self._sample_to_token(current_end_sample)
        left = max(0, current_token_start - window_token_start)
        right = max(left, current_token_end - window_token_start)
        left = min(left, token.shape[1])
        right = min(right, token.shape[1])
        if right <= left and current_end_sample > current_start_sample and left < token.shape[1]:
            right = left + 1
        return token[:, left:right]

    def _sample_to_token(self, sample: int) -> int:
        return round(sample * self.token_rate / 16000)


def run_window_stream(
    model: VCOnlyModel,
    source_stream: AsyncSourceTokenizer,
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    embedding: torch.Tensor,
    chunk_tokens: int,
    history_tokens: int,
    overlap_tokens: int,
    delayed_commit_tokens: int,
    audio_declick_samples: int,
    max_audio_blend_samples: int,
    flow_streaming: bool,
    hift_mode: str,
    use_prompt_kv_cache: bool,
    use_history_kv_cache: bool,
    prompt_cache_len: int,
    prompt_cache_steps,
) -> tuple[torch.Tensor, list[dict]]:
    right_context_tokens = model.flow.pre_lookahead_len
    hop_samples = model.sample_rate // (25 * model.token_mel_ratio)
    hift_right_context_mel = hift_required_right_context_mel(model, hop_samples)
    chunks = []
    rows = []
    stream_start = source_stream.started_at
    history_cache = None
    history_cache_start = 0
    history_cache_end = 0
    history_target_mel = history_tokens * model.token_mel_ratio
    overlap_target_mel = overlap_tokens * model.token_mel_ratio
    pending_overlap_mel = None
    pending_boundary_speech = None
    emitted_mel_tail = None
    hift_stream_state = None
    hift_window_phase_state = None
    hift_window_phase_state_start = 0

    index = 0
    start_token = 0
    while True:
        chunk_start_wall = time.perf_counter()
        target_end = start_token + chunk_tokens + delayed_commit_tokens + overlap_tokens + right_context_tokens
        token_wait_seconds = source_stream.wait_until(target_end)
        available_tokens = source_stream.token_count()
        source_done = source_stream.is_done()
        if available_tokens <= start_token:
            if source_done:
                break
            continue

        chunk_end = min(start_token + chunk_tokens, available_tokens)
        history_start = max(0, start_token - history_tokens)
        window_end = min(target_end, available_tokens)
        finalize = source_done and window_end >= available_tokens
        current_tokens = chunk_end - start_token
        current_mel = current_tokens * model.token_mel_ratio
        history_mel = (start_token - history_start) * model.token_mel_ratio
        cache_hit = (
            use_history_kv_cache
            and history_cache is not None
            and history_cache_start == history_start
            and history_cache_end == start_token
            and history_cache.get("history_cache_len", 0) == history_mel
        )
        active_cache_steps = history_cache if cache_hit else prompt_cache_steps
        active_cache_len = active_cache_steps["cache_len"] if active_cache_steps is not None else prompt_cache_len
        source_cache_len = min(history_target_mel, current_mel) if use_history_kv_cache and active_cache_steps is not None else 0
        source_cache_end = (0 if cache_hit else history_mel) + current_mel
        source_mel_offset = (start_token if cache_hit else history_start) * model.token_mel_ratio

        flow_start = time.perf_counter()
        mel_window, updated_cache = infer_flow_window(
            model=model,
            token=source_stream.get(history_start, window_end).to(model.device),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            streaming=flow_streaming,
            finalize=finalize,
            use_prompt_cache=use_prompt_kv_cache,
            prompt_cache_len=active_cache_len,
            prompt_cache_steps=active_cache_steps,
            source_cache_len=source_cache_len,
            source_cache_end=source_cache_end,
            source_mel_offset=source_mel_offset,
        )
        sync_device(model.device)
        flow_seconds = time.perf_counter() - flow_start

        history_mel_chunk = mel_window[:, :, :history_mel]
        current_mel_chunk = mel_window[:, :, history_mel:history_mel + current_mel].clone()
        future_overlap_mel = mel_window[:, :, history_mel + current_mel:]
        blend_mel = 0
        if pending_overlap_mel is not None:
            blend_mel = min(pending_overlap_mel.shape[2], current_mel_chunk.shape[2], overlap_target_mel)
            if blend_mel > 0:
                current_mel_chunk[:, :, :blend_mel] = blend_mel_chunks(
                    pending_overlap_mel[:, :, :blend_mel],
                    current_mel_chunk[:, :, :blend_mel],
                )

        if emitted_mel_tail is not None and history_mel > 0:
            cached_history_mel = min(history_mel, emitted_mel_tail.shape[2])
            if cached_history_mel > 0:
                history_mel_chunk = history_mel_chunk.clone()
                history_mel_chunk[:, :, -cached_history_mel:] = emitted_mel_tail[:, :, -cached_history_mel:]

        vocoder_finalize = finalize or future_overlap_mel.shape[2] < hift_right_context_mel
        hift_start = time.perf_counter()
        if hift_mode == "stateful":
            mel_for_vocoder = torch.cat([current_mel_chunk, future_overlap_mel], dim=2)
            speech_chunk, hift_stream_state = model.hift.inference_stateful(
                speech_feat=mel_for_vocoder,
                current_frames=current_mel,
                state=hift_stream_state,
                finalize=vocoder_finalize,
            )
        else:
            mel_for_vocoder = torch.cat([history_mel_chunk, current_mel_chunk, future_overlap_mel], dim=2)
            source_sample_offset = history_start * model.token_mel_ratio * hop_samples
            source_phase = None
            if hift_window_phase_state is not None and hift_window_phase_state_start == source_sample_offset:
                source_phase = hift_window_phase_state
            state_sample_at = max(0, (history_mel + current_mel - history_target_mel) * hop_samples)
            speech_window, hift_state = model.hift.inference(
                speech_feat=mel_for_vocoder,
                finalize=vocoder_finalize,
                source_sample_offset=source_sample_offset,
                source_phase=source_phase,
                return_source_state_at_sample=state_sample_at,
            )
            left_samples = history_mel * hop_samples
            right_samples = left_samples + current_mel * hop_samples
            if speech_window.shape[1] < right_samples:
                raise RuntimeError(
                    f"vocoder output is too short for current chunk: need {right_samples} samples, "
                    f"got {speech_window.shape[1]}; increase mel overlap or delayed commit"
                )
            speech_chunk = speech_window[:, left_samples:right_samples]
            if isinstance(hift_state, dict) and hift_state.get("source_phase") is not None:
                hift_window_phase_state = hift_state["source_phase"].detach()
                hift_window_phase_state_start = max(0, (chunk_end - history_tokens) * model.token_mel_ratio * hop_samples)
        sync_device(model.device)
        hift_seconds = time.perf_counter() - hift_start
        target_samples = current_mel * hop_samples
        if speech_chunk.shape[1] != target_samples:
            raise RuntimeError(
                f"stateful vocoder output has unexpected length: need {target_samples} samples, "
                f"got {speech_chunk.shape[1]}"
            )
        audio_blend_samples = 0
        if pending_boundary_speech is not None:
            audio_blend_samples = min(
                pending_boundary_speech.shape[1],
                speech_chunk.shape[1],
                max_audio_blend_samples,
            )
            if audio_blend_samples > 0:
                speech_chunk = speech_chunk.clone()
                speech_chunk[:, :audio_blend_samples] = blend_speech_chunks(
                    pending_boundary_speech[:, :audio_blend_samples].to(speech_chunk.device, dtype=speech_chunk.dtype),
                    speech_chunk[:, :audio_blend_samples],
                )
        declick_samples = 0
        if chunks and audio_declick_samples > 0:
            speech_chunk, declick_samples = declick_speech_boundary(
                previous=chunks[-1].to(speech_chunk.device, dtype=speech_chunk.dtype),
                current=speech_chunk,
                max_samples=audio_declick_samples,
            )
        chunks.append(speech_chunk.cpu())
        emitted_mel_tail = keep_mel_tail(
            previous_tail=emitted_mel_tail,
            new_mel=current_mel_chunk,
            max_frames=history_target_mel,
        )
        pending_overlap_mel = future_overlap_mel[:, :, :overlap_target_mel].detach() if overlap_target_mel > 0 else None
        pending_boundary_speech = None

        output_seconds = speech_chunk.shape[1] / model.sample_rate
        compute_seconds = flow_seconds + hift_seconds
        rows.append(
            make_row(
                index=index,
                start_token=start_token,
                end_token=chunk_end,
                window_tokens=window_end - history_start,
                history_tokens=start_token - history_start,
                delayed_commit_tokens=min(delayed_commit_tokens, max(0, window_end - chunk_end)),
                history_cache_hit=cache_hit,
                source_mel_offset=source_mel_offset,
                mel_blend_frames=blend_mel,
                audio_blend_samples=audio_blend_samples,
                audio_declick_samples=declick_samples,
                output_seconds=output_seconds,
                flow_seconds=flow_seconds,
                hift_seconds=hift_seconds,
                compute_seconds=compute_seconds,
                token_wait_seconds=token_wait_seconds,
                wall_end_seconds=time.perf_counter() - stream_start,
                finalize=finalize,
            )
        )
        print_chunk(rows[-1], chunk_start_wall)

        if updated_cache is not None and source_cache_len > 0:
            cached_tokens = updated_cache.get("history_cache_len", 0) // model.token_mel_ratio
            history_cache = updated_cache
            history_cache_start = chunk_end - cached_tokens
            history_cache_end = chunk_end

        start_token = chunk_end
        index += 1

    return concat_chunks(chunks), rows


def infer_flow_window(
    model: VCOnlyModel,
    token: torch.Tensor,
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    embedding: torch.Tensor,
    streaming: bool,
    finalize: bool,
    use_prompt_cache: bool,
    prompt_cache_len: int,
    prompt_cache_steps,
    source_cache_len: int,
    source_cache_end: int,
    source_mel_offset: int,
) -> tuple[torch.Tensor, dict | None]:
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=model.device)
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=model.device)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=model.device)
    with torch.inference_mode():
        mel, updated_cache = model.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
            use_prompt_cache=use_prompt_cache,
            prompt_cache_len=prompt_cache_len,
            prompt_cache_steps=prompt_cache_steps,
            source_cache_len=source_cache_len,
            source_cache_end=source_cache_end,
            source_mel_offset=source_mel_offset,
        )
    return mel, updated_cache


def blend_mel_chunks(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    frames = previous.shape[2]
    if frames <= 0:
        return current
    ramp = torch.linspace(0.0, 1.0, frames, device=current.device, dtype=current.dtype)
    fade_in = 0.5 - 0.5 * torch.cos(torch.pi * ramp)
    fade_in = fade_in.view(1, 1, frames)
    fade_out = 1.0 - fade_in
    return previous.to(current.device, dtype=current.dtype) * fade_out + current * fade_in


def blend_speech_chunks(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    samples = previous.shape[1]
    if samples <= 0:
        return current
    ramp = torch.linspace(0.0, 1.0, samples, device=current.device, dtype=current.dtype)
    fade_in = 0.5 - 0.5 * torch.cos(torch.pi * ramp)
    fade_in = fade_in.view(1, samples)
    fade_out = 1.0 - fade_in
    return previous.to(current.device, dtype=current.dtype) * fade_out + current * fade_in


def hift_required_right_context_mel(model: VCOnlyModel, hop_samples: int) -> int:
    f0_right = getattr(model.hift.f0_predictor.condnet[0], "causal_padding", 0)
    conv_pre_right = getattr(model.hift, "conv_pre_look_right", 0)
    upsample_samples = math.prod(getattr(model.hift, "upsample_rates", [1]))
    istft_trim_samples = upsample_samples * model.hift.istft_params["hop_len"]
    istft_right = math.ceil(istft_trim_samples / hop_samples)
    return int(f0_right + conv_pre_right + istft_right)


def declick_speech_boundary(
    previous: torch.Tensor,
    current: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, int]:
    if max_samples <= 0 or previous.shape[1] == 0 or current.shape[1] == 0:
        return current, 0
    samples = min(max_samples, current.shape[1])
    discontinuity = current[:, :1] - previous[:, -1:]
    if torch.max(torch.abs(discontinuity)).item() == 0:
        return current, 0

    ramp = torch.linspace(0.0, 1.0, samples, device=current.device, dtype=current.dtype)
    correction = 0.5 + 0.5 * torch.cos(torch.pi * ramp)
    current = current.clone()
    current[:, :samples] = current[:, :samples] - discontinuity * correction.view(1, -1)
    return current, samples


def keep_mel_tail(previous_tail: torch.Tensor | None, new_mel: torch.Tensor, max_frames: int) -> torch.Tensor | None:
    if max_frames <= 0:
        return None
    if previous_tail is None:
        merged = new_mel.detach()
    else:
        merged = torch.cat([previous_tail.to(new_mel.device), new_mel.detach()], dim=2)
    return merged[:, :, -max_frames:]


def make_row(
    index: int,
    start_token: int,
    end_token: int,
    window_tokens: int,
    history_tokens: int,
    delayed_commit_tokens: int,
    history_cache_hit: bool,
    source_mel_offset: int,
    mel_blend_frames: int,
    audio_blend_samples: int,
    audio_declick_samples: int,
    output_seconds: float,
    flow_seconds: float,
    hift_seconds: float,
    compute_seconds: float,
    token_wait_seconds: float,
    wall_end_seconds: float,
    finalize: bool,
) -> dict:
    input_seconds = (end_token - start_token) / 25.0
    return {
        "chunk": index,
        "start_token": start_token,
        "end_token": end_token,
        "window_tokens": window_tokens,
        "history_tokens": history_tokens,
        "delayed_commit_tokens": delayed_commit_tokens,
        "history_cache_hit": history_cache_hit,
        "source_mel_offset": source_mel_offset,
        "mel_blend_frames": mel_blend_frames,
        "audio_blend_samples": audio_blend_samples,
        "audio_declick_samples": audio_declick_samples,
        "input_seconds": input_seconds,
        "output_seconds": output_seconds,
        "flow_seconds": flow_seconds,
        "hift_seconds": hift_seconds,
        "compute_seconds": compute_seconds,
        "token_wait_seconds": token_wait_seconds,
        "chunk_rtf": compute_seconds / input_seconds if input_seconds > 0 else 0.0,
        "wall_end_seconds": wall_end_seconds,
        "finalize": finalize,
    }


def print_chunk(row: dict, chunk_start_wall: float) -> None:
    wall_seconds = time.perf_counter() - chunk_start_wall
    print(
        "chunk={chunk} input={input_seconds:.3f}s output={output_seconds:.3f}s "
        "window_tokens={window_tokens} history_tokens={history_tokens} delayed_commit_tokens={delayed_commit_tokens} "
        "history_cache_hit={history_cache_hit} "
        "source_mel_offset={source_mel_offset} mel_blend_frames={mel_blend_frames} "
        "audio_blend_samples={audio_blend_samples} audio_declick_samples={audio_declick_samples} "
        "token_wait={token_wait_seconds:.3f}s flow={flow_seconds:.3f}s hift={hift_seconds:.3f}s compute={compute_seconds:.3f}s "
        "rtf={chunk_rtf:.3f} wall={wall:.3f}s finalize={finalize}".format(
            **row,
            wall=wall_seconds,
        ),
        flush=True,
    )


def concat_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.zeros(1, 0)
    return torch.cat(chunks, dim=1)


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
