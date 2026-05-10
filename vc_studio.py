#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import math
import os
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
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
    parser = argparse.ArgumentParser(description="CosyVoice VC Studio GUI")
    parser.add_argument("--model-dir", default="", help="Prefill model directory")
    parser.add_argument("--source", default="", help="Prefill offline source wav")
    parser.add_argument("--prompt", default="", help="Prefill target speaker reference wav")
    parser.add_argument("--output", default="out/vc_streaming.wav", help="Prefill offline output wav path")
    parser.add_argument("--csv", default="", help="Prefill optional offline CSV path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Prefill torch device")
    parser.add_argument(
        "--ort-provider",
        default="auto",
        choices=["auto", "cpu", "cuda", "coreml"],
        help="Prefill ONNX Runtime provider",
    )
    parser.add_argument("--coreml-cache-dir", default="", help="Prefill CoreML cache directory")
    parser.add_argument("--chunk-sec", type=float, default=1.0, help="Prefill source chunk size")
    parser.add_argument("--tokenizer-chunk-sec", type=float, default=0.0, help="Prefill tokenizer chunk size; 0 follows chunk size")
    parser.add_argument("--tokenizer-left-context-sec", type=float, default=0.5, help="Prefill tokenizer left context")
    parser.add_argument("--tokenizer-right-context-sec", type=float, default=0.2, help="Prefill tokenizer right context")
    parser.add_argument("--history-sec", type=float, default=1.0, help="Prefill flow left history")
    parser.add_argument("--mel-overlap-sec", type=float, default=0.25, help="Prefill mel overlap")
    parser.add_argument("--delayed-commit-sec", type=float, default=0.0, help="Prefill delayed commit")
    parser.add_argument("--audio-declick-ms", type=float, default=0.0, help="Prefill waveform de-click")
    parser.add_argument("--audio-blend-ms", type=float, default=0.0, help="Prefill waveform crossfade")
    parser.add_argument("--flow-context", default="streaming", choices=["streaming", "window-full"], help="Prefill flow context")
    parser.add_argument("--hift-mode", default="stateful", choices=["window", "stateful"], help="Prefill HiFT mode")
    parser.add_argument("--disable-prompt-kv-cache", action="store_true", help="Prefill prompt cache off")
    parser.add_argument("--disable-history-kv-cache", action="store_true", help="Prefill history cache off")
    args = parser.parse_args()
    launch_gui(args)


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


@dataclass(frozen=True)
class StreamSettings:
    chunk_sec: float
    tokenizer_chunk_sec: float | None
    tokenizer_left_context_sec: float
    tokenizer_right_context_sec: float
    history_sec: float
    mel_overlap_sec: float
    delayed_commit_sec: float
    audio_declick_ms: float
    audio_blend_ms: float
    flow_context: str
    hift_mode: str
    disable_prompt_kv_cache: bool
    disable_history_kv_cache: bool


@dataclass
class PreparedStreamContext:
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    chunk_tokens: int
    tokenizer_chunk_sec: float
    history_tokens: int
    overlap_tokens: int
    delayed_commit_tokens: int
    audio_declick_samples: int
    max_audio_blend_samples: int
    flow_streaming: bool
    use_prompt_kv_cache: bool
    use_history_kv_cache: bool
    prompt_cache_len: int
    prompt_cache_steps: object | None
    prompt_prepare_seconds: float
    prompt_cache_prepare_seconds: float


def prepare_stream_context(
    model: VCOnlyModel,
    prompt_wav: str | Path,
    settings: StreamSettings,
) -> PreparedStreamContext:
    chunk_tokens = max(1, round(settings.chunk_sec * 25))
    tokenizer_chunk_sec = settings.tokenizer_chunk_sec if settings.tokenizer_chunk_sec else settings.chunk_sec
    history_tokens = align_history_tokens(settings.history_sec, model)
    overlap_tokens = align_overlap_tokens(settings.mel_overlap_sec)
    delayed_commit_tokens = align_delayed_commit_tokens(settings.delayed_commit_sec)
    audio_declick_samples = align_audio_declick_samples(settings.audio_declick_ms, model.sample_rate)
    max_audio_blend_samples = align_audio_blend_samples(settings.audio_blend_ms, model.sample_rate)
    flow_streaming = settings.flow_context == "streaming"
    use_prompt_kv_cache = flow_streaming and not settings.disable_prompt_kv_cache
    history_cache_enabled = (
        flow_streaming
        and not settings.disable_history_kv_cache
        and use_prompt_kv_cache
        and is_static_cache_aligned(history_tokens, model)
    )

    prompt_prepare_start = time.perf_counter()
    prompt_token, prompt_feat, embedding = prepare_prompt_inputs(model, prompt_wav)
    if use_prompt_kv_cache:
        prompt_token, prompt_feat = trim_prompt_to_static_cache(prompt_token, prompt_feat, model)
    sync_device(model.device)
    prompt_prepare_seconds = time.perf_counter() - prompt_prepare_start

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

    return PreparedStreamContext(
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        embedding=embedding,
        chunk_tokens=chunk_tokens,
        tokenizer_chunk_sec=tokenizer_chunk_sec,
        history_tokens=history_tokens,
        overlap_tokens=overlap_tokens,
        delayed_commit_tokens=delayed_commit_tokens,
        audio_declick_samples=audio_declick_samples,
        max_audio_blend_samples=max_audio_blend_samples,
        flow_streaming=flow_streaming,
        use_prompt_kv_cache=use_prompt_kv_cache,
        use_history_kv_cache=history_cache_enabled,
        prompt_cache_len=prompt_cache_len,
        prompt_cache_steps=prompt_cache_steps,
        prompt_prepare_seconds=prompt_prepare_seconds,
        prompt_cache_prepare_seconds=prompt_cache_prepare_seconds,
    )


def stream_context_log_lines(
    model: VCOnlyModel,
    settings: StreamSettings,
    context: PreparedStreamContext,
) -> list[str]:
    return [
        f"device={model.device}",
        f"flow_context={settings.flow_context}",
        f"hift_mode={settings.hift_mode}",
        f"prompt_kv_cache={context.use_prompt_kv_cache}",
        f"history_kv_cache={context.use_history_kv_cache}",
        f"prompt_tokens={context.prompt_token.shape[1]} prompt_seconds={context.prompt_token.shape[1] / 25.0:.3f}",
        f"prompt_cache_mel={context.prompt_cache_len} prompt_cache_seconds={context.prompt_cache_len / 50.0:.3f}",
        f"chunk_tokens={context.chunk_tokens} chunk_seconds={context.chunk_tokens / 25.0:.3f}",
        f"tokenizer_chunk_seconds={context.tokenizer_chunk_sec:.3f}",
        f"tokenizer_left_context_seconds={settings.tokenizer_left_context_sec:.3f}",
        f"tokenizer_right_context_seconds={settings.tokenizer_right_context_sec:.3f}",
        f"history_tokens={context.history_tokens} history_seconds={context.history_tokens / 25.0:.3f}",
        f"mel_overlap_tokens={context.overlap_tokens} mel_overlap_seconds={context.overlap_tokens / 25.0:.3f}",
        f"delayed_commit_tokens={context.delayed_commit_tokens} delayed_commit_seconds={context.delayed_commit_tokens / 25.0:.3f}",
        f"audio_declick_samples={context.audio_declick_samples} audio_declick_ms={context.audio_declick_samples / model.sample_rate * 1000:.3f}",
        f"max_audio_blend_samples={context.max_audio_blend_samples} audio_blend_ms={context.max_audio_blend_samples / model.sample_rate * 1000:.3f}",
        f"prompt_prepare_seconds={context.prompt_prepare_seconds:.3f}",
        f"prompt_cache_prepare_seconds={context.prompt_cache_prepare_seconds:.3f}",
    ]


def offline_summary_lines(
    model: VCOnlyModel,
    source_stats: dict,
    rows: list[dict],
    output: str | Path,
) -> list[str]:
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
    return [
        f"source_tokens={source_tokens} source_seconds={source_seconds:.3f}",
        f"source_tokenize_chunks={source_stats['chunks']}",
        f"source_tokenize_audio_seconds={source_stats['audio_seconds']:.3f}",
        f"source_tokenize_window_audio_seconds={source_stats['window_audio_seconds']:.3f}",
        f"source_tokenize_read_seconds={source_stats['read_seconds']:.3f}",
        f"source_tokenize_compute_seconds={source_stats['tokenize_seconds']:.3f}",
        f"source_tokenize_wall_seconds={source_stats['wall_seconds']:.3f}",
        f"stream_token_wait_seconds={token_wait_seconds:.3f}",
        f"stream_infer_compute_seconds={infer_compute_seconds:.3f}",
        f"stream_pipeline_compute_seconds={pipeline_compute_seconds:.3f}",
        f"stream_wall_seconds={wall_pipeline_seconds:.3f}",
        f"avg_infer_compute_rtf={avg_infer_rtf:.3f}",
        f"avg_pipeline_compute_rtf={avg_pipeline_rtf:.3f}",
        f"wall_stream_rtf={wall_rtf:.3f}",
        f"max_chunk_compute_seconds={max_chunk_seconds:.3f}",
        f"chunks={len(rows)}",
        f"output={Path(output).resolve()}",
    ]


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


class MicrophoneSourceTokenizer:
    token_rate = 25.0

    def __init__(
        self,
        model: VCOnlyModel,
        chunk_sec: float,
        input_device: int | None = None,
        left_context_sec: float = 0.0,
        right_context_sec: float = 0.0,
        log_fn: Callable[[str], None] | None = None,
    ):
        if chunk_sec <= 0:
            raise ValueError("tokenizer chunk size must be positive")
        if left_context_sec < 0 or right_context_sec < 0:
            raise ValueError("tokenizer context sizes must be non-negative")
        if chunk_sec + left_context_sec + right_context_sec > 30:
            raise ValueError("tokenizer chunk plus context must be <= 30 seconds")
        self.model = model
        self.input_device = input_device
        self.chunk_samples = max(1, round(chunk_sec * 16000))
        self.left_context_samples = round(left_context_sec * 16000)
        self.right_context_samples = round(right_context_sec * 16000)
        self.log_fn = log_fn
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._token_chunks: list[torch.Tensor] = []
        self._audio = np.zeros(0, dtype=np.float32)
        self._next_start_sample = 0
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
            raise RuntimeError("microphone tokenizer has not been started")
        return self._started_at

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("microphone tokenizer has already been started")
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="microphone-tokenizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

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
            try:
                import sounddevice as sd
            except ImportError as error:
                raise RuntimeError("sounddevice is required for microphone streaming") from error

            def callback(indata, frames, time_info, status) -> None:
                if status and self.log_fn is not None:
                    self.log_fn(f"input status: {status}")
                mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
                self._audio_queue.put(mono)

            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=0,
                device=self.input_device,
                callback=callback,
            ):
                while not self._stop_event.is_set():
                    self._drain_audio_queue(timeout=0.05)
                    self._process_available(final=False)
                self._drain_audio_queue(timeout=0.0)
                self._process_available(final=True)
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

    def _drain_audio_queue(self, timeout: float) -> None:
        blocks = []
        read_start = time.perf_counter()
        try:
            first = self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return
        blocks.append(first)
        while True:
            try:
                blocks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        if blocks:
            self._audio = np.concatenate([self._audio, *blocks])
            with self._condition:
                self._read_seconds += time.perf_counter() - read_start
                self._condition.notify_all()

    def _process_available(self, final: bool) -> None:
        total_samples = self._audio.shape[0]
        while self._next_start_sample < total_samples:
            start = self._next_start_sample
            target_current_end = start + self.chunk_samples
            if not final and total_samples < target_current_end + self.right_context_samples:
                break
            current_end = min(target_current_end, total_samples)
            if current_end <= start:
                break
            window_start = max(0, start - self.left_context_samples)
            window_end = min(total_samples, current_end + self.right_context_samples)
            chunk = torch.from_numpy(self._audio[window_start:window_end].copy()).float().unsqueeze(0)
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
            self._next_start_sample = current_end

    def _raise_if_error(self) -> None:
        with self._condition:
            self._raise_if_error_locked()

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("microphone tokenizer failed") from self._error

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


class RealtimeAudioPlayer:
    def __init__(
        self,
        sample_rate: int,
        output_device: int | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.log_fn = log_fn
        self._stream = None
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._queued_samples = 0
        self._played_samples = 0
        self._underflows = 0
        self._has_written = False

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required for realtime playback") from error
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=0,
            device=self.output_device,
            callback=self._callback,
        )
        self._stream.start()

    def write(self, speech: torch.Tensor) -> None:
        audio = speech.detach().cpu().squeeze(0).numpy().astype(np.float32)
        audio = np.clip(audio, -0.999, 0.999)
        if audio.size == 0:
            return
        with self._lock:
            self._chunks.append(audio)
            self._queued_samples += int(audio.size)
            self._has_written = True

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def stats(self) -> dict:
        with self._lock:
            buffered = self._queued_samples - self._played_samples
            return {
                "buffer_seconds": max(0, buffered) / self.sample_rate,
                "played_seconds": self._played_samples / self.sample_rate,
                "underflows": self._underflows,
            }

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status and self.log_fn is not None:
            self.log_fn(f"output status: {status}")
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                available = chunk.shape[0] - self._offset
                take = min(frames - filled, available)
                out[filled:filled + take] = chunk[self._offset:self._offset + take]
                filled += take
                self._offset += take
                if self._offset >= chunk.shape[0]:
                    self._chunks.popleft()
                    self._offset = 0
            self._played_samples += filled
            if filled < frames and self._has_written:
                self._underflows += 1
        outdata[:, 0] = out


def run_window_stream(
    model: VCOnlyModel,
    source_stream,
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
    on_audio_chunk: Callable[[torch.Tensor, dict], None] | None = None,
    log_fn: Callable[[str], None] | None = print,
    stop_event: threading.Event | None = None,
    collect_chunks: bool = True,
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
    previous_speech_chunk = None
    emitted_mel_tail = None
    hift_stream_state = None
    hift_window_phase_state = None
    hift_window_phase_state_start = 0

    index = 0
    start_token = 0
    while True:
        if (
            stop_event is not None
            and stop_event.is_set()
            and source_stream.is_done()
            and source_stream.token_count() <= start_token
        ):
            break
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
        if previous_speech_chunk is not None and audio_declick_samples > 0:
            speech_chunk, declick_samples = declick_speech_boundary(
                previous=previous_speech_chunk.to(speech_chunk.device, dtype=speech_chunk.dtype),
                current=speech_chunk,
                max_samples=audio_declick_samples,
            )
        speech_chunk_cpu = speech_chunk.cpu()
        if collect_chunks:
            chunks.append(speech_chunk_cpu)
        previous_speech_chunk = speech_chunk_cpu
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
        if on_audio_chunk is not None:
            on_audio_chunk(speech_chunk_cpu, rows[-1])
        if log_fn is not None:
            print_chunk(rows[-1], chunk_start_wall, log_fn=log_fn)

        if updated_cache is not None and source_cache_len > 0:
            cached_tokens = updated_cache.get("history_cache_len", 0) // model.token_mel_ratio
            history_cache = updated_cache
            history_cache_start = chunk_end - cached_tokens
            history_cache_end = chunk_end

        start_token = chunk_end
        index += 1
        if stop_event is not None and stop_event.is_set():
            source_done = source_stream.is_done()
            if source_done:
                break

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


def print_chunk(row: dict, chunk_start_wall: float, log_fn: Callable[[str], None] = print) -> None:
    wall_seconds = time.perf_counter() - chunk_start_wall
    log_fn(
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


def rounded_rect(canvas, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs) -> int:
    radius = min(radius, max(0, (x2 - x1) // 2), max(0, (y2 - y1) // 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=18, **kwargs)


class RoundedButton:
    def __init__(
        self,
        parent,
        text: str,
        command: Callable[[], None],
        *,
        width: int = 128,
        height: int = 38,
        radius: int = 16,
        fill: str,
        hover_fill: str,
        disabled_fill: str,
        foreground: str,
        background: str,
        font: tuple[str, int, str] = ("Helvetica", 11, "bold"),
    ):
        import tkinter as tk

        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=background,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.radius = radius
        self.fill = fill
        self.hover_fill = hover_fill
        self.disabled_fill = disabled_fill
        self.foreground = foreground
        self.background = background
        self.font = font
        self.state = "normal"
        self.hover = False
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Button-1>", self._on_click)
        self._draw()

    def grid(self, *args, **kwargs):
        return self.canvas.grid(*args, **kwargs)

    def pack(self, *args, **kwargs):
        return self.canvas.pack(*args, **kwargs)

    def configure(self, cnf=None, **kwargs) -> None:
        if cnf:
            kwargs.update(cnf)
        if "state" in kwargs:
            self.state = kwargs.pop("state")
            self.canvas.configure(cursor="" if self.state == "disabled" else "hand2")
            self._draw()
        if "text" in kwargs:
            self.text = kwargs.pop("text")
            self._draw()
        if kwargs:
            self.canvas.configure(**kwargs)

    config = configure

    def _draw(self) -> None:
        self.canvas.delete("all")
        fill = self.disabled_fill if self.state == "disabled" else self.hover_fill if self.hover else self.fill
        text_fill = "#f8faf8" if self.state != "disabled" else "#d5d3cb"
        rounded_rect(self.canvas, 1, 1, self.width - 1, self.height - 1, self.radius, fill=fill, outline="")
        self.canvas.create_text(
            self.width // 2,
            self.height // 2,
            text=self.text,
            fill=text_fill if self.foreground == "#ffffff" else self.foreground,
            font=self.font,
        )

    def _on_enter(self, event) -> None:
        self.hover = True
        self._draw()

    def _on_leave(self, event) -> None:
        self.hover = False
        self._draw()

    def _on_click(self, event) -> None:
        if self.state != "disabled":
            self.command()


class MetricCard:
    def __init__(
        self,
        parent,
        name: str,
        variable,
        *,
        colors: dict[str, str],
        width: int = 142,
        height: int = 78,
    ):
        import tkinter as tk

        self.variable = variable
        self.name = name
        self.colors = colors
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=colors["panel"],
            highlightthickness=0,
            bd=0,
        )
        self.variable.trace_add("write", lambda *_: self._draw())
        self.canvas.bind("<Configure>", self._on_configure)
        self._draw()

    def grid(self, *args, **kwargs):
        return self.canvas.grid(*args, **kwargs)

    def _on_configure(self, event) -> None:
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        rounded_rect(
            self.canvas,
            1,
            1,
            self.width - 1,
            self.height - 1,
            14,
            fill=self.colors["card"],
            outline=self.colors["line"],
            width=1,
        )
        self.canvas.create_text(
            16,
            18,
            text=self.name,
            fill=self.colors["muted"],
            anchor="w",
            font=("Helvetica", 10),
        )
        self.canvas.create_text(
            16,
            48,
            text=self.variable.get(),
            fill=self.colors["text"],
            anchor="w",
            font=("Helvetica", 19, "bold"),
        )


def launch_gui(args: argparse.Namespace) -> None:
    app = VCStudioApp(args)
    app.mainloop()


class VCStudioApp:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("CosyVoice VC Studio")
        self.root.geometry("1180x820")
        self.root.minsize(1040, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.active_mode: str | None = None
        self.offline_stop_event: threading.Event | None = None
        self.live_stop_event: threading.Event | None = None
        self.live_source: MicrophoneSourceTokenizer | None = None
        self.live_player: RealtimeAudioPlayer | None = None
        self.model_lock = threading.Lock()
        self.model_key: tuple[str, str, str, str] | None = None
        self.model: VCOnlyModel | None = None
        self.input_device_map: dict[str, int | None] = {"Default": None}
        self.output_device_map: dict[str, int | None] = {"Default": None}

        self._create_variables(args)
        self._configure_style()
        self._build_ui()
        self._refresh_audio_devices()
        self.root.after(100, self._poll_ui_queue)

    def mainloop(self) -> None:
        self.root.mainloop()

    def _create_variables(self, args: argparse.Namespace) -> None:
        tk = self.tk
        self.model_dir_var = tk.StringVar(value=args.model_dir)
        self.prompt_var = tk.StringVar(value=args.prompt)
        self.source_var = tk.StringVar(value=args.source)
        self.output_var = tk.StringVar(value=args.output)
        self.csv_var = tk.StringVar(value=args.csv)
        self.device_var = tk.StringVar(value=args.device)
        self.ort_provider_var = tk.StringVar(value=args.ort_provider)
        self.coreml_cache_var = tk.StringVar(value=args.coreml_cache_dir)
        self.chunk_sec_var = tk.StringVar(value=f"{args.chunk_sec:g}")
        self.tokenizer_chunk_sec_var = tk.StringVar(value=f"{args.tokenizer_chunk_sec:g}")
        self.tokenizer_left_context_sec_var = tk.StringVar(value=f"{args.tokenizer_left_context_sec:g}")
        self.tokenizer_right_context_sec_var = tk.StringVar(value=f"{args.tokenizer_right_context_sec:g}")
        self.history_sec_var = tk.StringVar(value=f"{args.history_sec:g}")
        self.mel_overlap_sec_var = tk.StringVar(value=f"{args.mel_overlap_sec:g}")
        self.delayed_commit_sec_var = tk.StringVar(value=f"{args.delayed_commit_sec:g}")
        self.audio_declick_ms_var = tk.StringVar(value=f"{args.audio_declick_ms:g}")
        self.audio_blend_ms_var = tk.StringVar(value=f"{args.audio_blend_ms:g}")
        self.flow_context_var = tk.StringVar(value=args.flow_context)
        self.hift_mode_var = tk.StringVar(value=args.hift_mode)
        self.prompt_cache_var = tk.BooleanVar(value=not args.disable_prompt_kv_cache)
        self.history_cache_var = tk.BooleanVar(value=not args.disable_history_kv_cache)
        self.input_device_var = tk.StringVar(value="Default")
        self.output_device_var = tk.StringVar(value="Default")
        self.status_var = tk.StringVar(value="Ready")
        self.metric_chunk_var = tk.StringVar(value="-")
        self.metric_rtf_var = tk.StringVar(value="-")
        self.metric_lag_var = tk.StringVar(value="-")
        self.metric_buffer_var = tk.StringVar(value="-")
        self.metric_underflow_var = tk.StringVar(value="-")

    def _configure_style(self) -> None:
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        bg = "#f4f1ea"
        panel = "#fbf7ef"
        card = "#ffffff"
        text = "#252a31"
        muted = "#747b72"
        accent = "#2f7d75"
        accent_hover = "#398f86"
        danger = "#b85852"
        danger_hover = "#c8645e"
        line = "#ded6c9"
        field = "#fffdf8"
        tab = "#ebe4d7"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=text, fieldbackground=field, bordercolor=line)
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("Panel.TLabel", background=panel, foreground=text)
        style.configure("Card.TLabel", background=card, foreground=text)
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Helvetica", 22, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Helvetica", 12))
        style.configure("Metric.TLabel", background=card, foreground=text, font=("Helvetica", 18, "bold"))
        style.configure("MetricName.TLabel", background=card, foreground=muted, font=("Helvetica", 10))
        style.configure("TButton", padding=(12, 7), background="#e8dfd0", foreground=text, bordercolor=line)
        style.map("TButton", background=[("active", "#efe7da"), ("disabled", "#ded8ce")])
        style.configure("Accent.TButton", background=accent, foreground="#ffffff", bordercolor=accent)
        style.map("Accent.TButton", background=[("active", accent_hover), ("disabled", "#b9c7c2")])
        style.configure("Danger.TButton", background=danger, foreground="#ffffff", bordercolor=danger)
        style.map("Danger.TButton", background=[("active", danger_hover), ("disabled", "#d8c0bb")])
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), background=tab, foreground=muted)
        style.map("TNotebook.Tab", background=[("selected", card)], foreground=[("selected", text)])
        self.colors = {
            "bg": bg,
            "panel": panel,
            "card": card,
            "text": text,
            "muted": muted,
            "accent": accent,
            "accent_hover": accent_hover,
            "danger": danger,
            "danger_hover": danger_hover,
            "line": line,
            "field": field,
            "disabled": "#cfc7bb",
        }

    def _build_ui(self) -> None:
        ttk = self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="CosyVoice VC Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Realtime microphone conversion and offline benchmark in one control room.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Subtitle.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(root, padding=(18, 6, 18, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        left.columnconfigure(1, weight=1)
        self._build_model_panel(left)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._build_realtime_tab(notebook)
        self._build_offline_tab(notebook)
        self._build_parameters_tab(notebook)

        bottom = ttk.Frame(root, padding=(18, 0, 18, 16))
        bottom.grid(row=2, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)
        ttk.Label(bottom, text="Run Log", style="Subtitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log_text = self.tk.Text(
            bottom,
            height=9,
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["line"],
            highlightcolor=self.colors["accent"],
            padx=12,
            pady=10,
            wrap="word",
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")

    def _build_model_panel(self, parent) -> None:
        ttk = self.ttk
        row = 0
        ttk.Label(parent, text="Model", style="Panel.TLabel", font=("Helvetica", 14, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 12))
        row += 1
        row = self._path_row(parent, row, "Model dir", self.model_dir_var, "directory")
        row = self._path_row(parent, row, "Prompt wav", self.prompt_var, "open_wav")
        ttk.Label(parent, text="Torch device", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=self.device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=12).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1
        ttk.Label(parent, text="ORT provider", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=self.ort_provider_var, values=["auto", "cpu", "cuda", "coreml"], state="readonly", width=12).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1
        row = self._path_row(parent, row, "CoreML cache", self.coreml_cache_var, "directory", optional=True)
        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=14)
        row += 1
        ttk.Label(parent, text="Metrics", style="Panel.TLabel", font=("Helvetica", 14, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1
        metrics = ttk.Frame(parent, style="Panel.TFrame")
        metrics.grid(row=row, column=0, columnspan=3, sticky="ew")
        metrics.columnconfigure((0, 1), weight=1)
        self._metric_card(metrics, 0, 0, "Chunk", self.metric_chunk_var)
        self._metric_card(metrics, 0, 1, "RTF", self.metric_rtf_var)
        self._metric_card(metrics, 1, 0, "Lag", self.metric_lag_var)
        self._metric_card(metrics, 1, 1, "Buffer", self.metric_buffer_var)
        self._metric_card(metrics, 2, 0, "Underflows", self.metric_underflow_var)

    def _build_realtime_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure(0, weight=1)
        notebook.add(tab, text="Realtime")
        ttk.Label(tab, text="Audio I/O", font=("Helvetica", 15, "bold")).grid(row=0, column=0, sticky="w")
        device_frame = ttk.Frame(tab)
        device_frame.grid(row=1, column=0, sticky="ew", pady=(12, 16))
        device_frame.columnconfigure(1, weight=1)
        ttk.Label(device_frame, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, state="readonly")
        self.input_device_combo.grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(device_frame, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var, state="readonly")
        self.output_device_combo.grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(device_frame, text="Refresh", command=self._refresh_audio_devices).grid(row=0, column=2, rowspan=2, sticky="ns", padx=(10, 0), pady=5)

        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, sticky="w", pady=(4, 20))
        self.start_live_button = RoundedButton(
            buttons,
            "Start Live",
            self._start_realtime,
            width=132,
            fill=self.colors["accent"],
            hover_fill=self.colors["accent_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.start_live_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_live_button = RoundedButton(
            buttons,
            "Stop",
            self._stop_realtime,
            width=92,
            fill=self.colors["danger"],
            hover_fill=self.colors["danger_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.stop_live_button.configure(state="disabled")
        self.stop_live_button.grid(row=0, column=1)

        hint = (
            "Use headphones to avoid feeding generated audio back into the microphone. "
            "Lower chunk and overlap values reduce latency; larger context usually improves stability."
        )
        ttk.Label(tab, text=hint, style="Muted.TLabel", wraplength=680).grid(row=3, column=0, sticky="w")

    def _build_offline_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text="Offline Benchmark")
        ttk.Label(tab, text="Benchmark Job", font=("Helvetica", 15, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        row = 1
        row = self._path_row(tab, row, "Source wav", self.source_var, "open_wav")
        row = self._path_row(tab, row, "Output wav", self.output_var, "save_wav")
        row = self._path_row(tab, row, "CSV report", self.csv_var, "save_csv", optional=True)
        buttons = ttk.Frame(tab)
        buttons.grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 0))
        self.run_offline_button = RoundedButton(
            buttons,
            "Run Benchmark",
            self._start_offline,
            width=158,
            fill=self.colors["accent"],
            hover_fill=self.colors["accent_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.run_offline_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_offline_button = RoundedButton(
            buttons,
            "Stop",
            self._stop_offline,
            width=92,
            fill=self.colors["danger"],
            hover_fill=self.colors["danger_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.stop_offline_button.configure(state="disabled")
        self.stop_offline_button.grid(row=0, column=1)

    def _build_parameters_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure((0, 1), weight=1)
        notebook.add(tab, text="Parameters")

        left = ttk.Frame(tab, style="Card.TFrame", padding=14)
        right = ttk.Frame(tab, style="Card.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ttk.Label(left, text="Timing", style="Card.TLabel", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        timing = [
            ("Chunk sec", self.chunk_sec_var),
            ("Tokenizer chunk", self.tokenizer_chunk_sec_var),
            ("Tokenizer left ctx", self.tokenizer_left_context_sec_var),
            ("Tokenizer right ctx", self.tokenizer_right_context_sec_var),
            ("History sec", self.history_sec_var),
            ("Mel overlap sec", self.mel_overlap_sec_var),
            ("Delayed commit sec", self.delayed_commit_sec_var),
        ]
        for index, (label, var) in enumerate(timing, start=1):
            self._number_row(left, index, label, var)

        ttk.Label(right, text="Quality / Runtime", style="Card.TLabel", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._number_row(right, 1, "De-click ms", self.audio_declick_ms_var)
        self._number_row(right, 2, "Audio blend ms", self.audio_blend_ms_var)
        ttk.Label(right, text="Flow context", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(right, textvariable=self.flow_context_var, values=["streaming", "window-full"], state="readonly").grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Label(right, text="HiFT mode", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(right, textvariable=self.hift_mode_var, values=["stateful", "window"], state="readonly").grid(row=4, column=1, sticky="ew", pady=6)
        ttk.Checkbutton(right, text="Prompt KV cache", variable=self.prompt_cache_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 2))
        ttk.Checkbutton(right, text="History KV cache", variable=self.history_cache_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        left.columnconfigure(1, weight=1)
        right.columnconfigure(1, weight=1)

    def _number_row(self, parent, row: int, label: str, variable) -> None:
        self.ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        self.ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", pady=6)

    def _path_row(self, parent, row: int, label: str, variable, kind: str, optional: bool = False) -> int:
        ttk = self.ttk
        style = "Panel.TLabel" if str(parent.cget("style")) == "Panel.TFrame" else "TLabel"
        ttk.Label(parent, text=label, style=style).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 8))
        ttk.Entry(parent, textvariable=variable, width=36).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_path(variable, kind)).grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=5)
        if optional:
            variable.set(variable.get())
        return row + 1

    def _metric_card(self, parent, row: int, column: int, name: str, variable) -> None:
        card = MetricCard(parent, name, variable, colors=self.colors)
        card.grid(row=row, column=column, sticky="ew", padx=4, pady=4)

    def _browse_path(self, variable, kind: str) -> None:
        if kind == "directory":
            value = self.filedialog.askdirectory(initialdir=self._initial_dir(variable.get()))
        elif kind == "save_wav":
            value = self.filedialog.asksaveasfilename(
                initialdir=self._initial_dir(variable.get()),
                defaultextension=".wav",
                filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            )
        elif kind == "save_csv":
            value = self.filedialog.asksaveasfilename(
                initialdir=self._initial_dir(variable.get()),
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
        else:
            value = self.filedialog.askopenfilename(
                initialdir=self._initial_dir(variable.get()),
                filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            )
        if value:
            variable.set(value)

    def _initial_dir(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.is_dir():
            return str(path)
        if path.parent.exists():
            return str(path.parent)
        return str(Path.cwd())

    def _refresh_audio_devices(self) -> None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as error:
            self._log(f"Audio device refresh failed: {error}")
            self.input_device_map = {"Default": None}
            self.output_device_map = {"Default": None}
        else:
            self.input_device_map = {"Default": None}
            self.output_device_map = {"Default": None}
            for index, device in enumerate(devices):
                hostapi = hostapis[device["hostapi"]]["name"] if hostapis else ""
                label = f"{index}: {device['name']} [{hostapi}]"
                if device.get("max_input_channels", 0) > 0:
                    self.input_device_map[label] = index
                if device.get("max_output_channels", 0) > 0:
                    self.output_device_map[label] = index
        self.input_device_combo["values"] = list(self.input_device_map.keys())
        self.output_device_combo["values"] = list(self.output_device_map.keys())
        if self.input_device_var.get() not in self.input_device_map:
            self.input_device_var.set("Default")
        if self.output_device_var.get() not in self.output_device_map:
            self.output_device_var.set("Default")

    def _start_offline(self) -> None:
        if self._is_running():
            self.messagebox.showinfo("Busy", "A job is already running.")
            return
        try:
            config = self._snapshot_config(require_source=True)
        except ValueError as error:
            self.messagebox.showerror("Invalid settings", str(error))
            return
        self.offline_stop_event = threading.Event()
        self._set_running("offline", True)
        self.worker_thread = threading.Thread(target=self._offline_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop_offline(self) -> None:
        if self.offline_stop_event is not None:
            self.offline_stop_event.set()
        self._post("status", "Stopping offline job after the current chunk...")

    def _start_realtime(self) -> None:
        if self._is_running():
            self.messagebox.showinfo("Busy", "A job is already running.")
            return
        import importlib.util

        if importlib.util.find_spec("sounddevice") is None:
            self.messagebox.showerror(
                "Missing dependency",
                "Realtime audio requires sounddevice. Install requirements.txt, then restart the GUI.",
            )
            return
        try:
            config = self._snapshot_config(require_source=False)
        except ValueError as error:
            self.messagebox.showerror("Invalid settings", str(error))
            return
        config["input_device"] = self.input_device_map.get(self.input_device_var.get())
        config["output_device"] = self.output_device_map.get(self.output_device_var.get())
        self.live_stop_event = threading.Event()
        self._set_running("realtime", True)
        self.worker_thread = threading.Thread(target=self._realtime_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop_realtime(self) -> None:
        if self.live_stop_event is not None:
            self.live_stop_event.set()
        if self.live_source is not None:
            self.live_source.stop()
        self._post("status", "Stopping live stream...")

    def _snapshot_config(self, require_source: bool) -> dict:
        model_dir = self.model_dir_var.get().strip()
        prompt = self.prompt_var.get().strip()
        if not model_dir:
            raise ValueError("Model directory is required.")
        if not prompt:
            raise ValueError("Prompt wav is required.")
        if not Path(model_dir).expanduser().is_dir():
            raise ValueError("Model directory does not exist.")
        if not Path(prompt).expanduser().is_file():
            raise ValueError("Prompt wav does not exist.")
        if require_source and not self.source_var.get().strip():
            raise ValueError("Source wav is required for offline benchmark.")
        if require_source and not Path(self.source_var.get().strip()).expanduser().is_file():
            raise ValueError("Source wav does not exist.")
        settings = self._settings_from_form()
        source = self.source_var.get().strip()
        output = self.output_var.get().strip() or "out/vc_streaming.wav"
        csv_path = self.csv_var.get().strip()
        return {
            "model_dir": str(Path(model_dir).expanduser()),
            "prompt": str(Path(prompt).expanduser()),
            "source": str(Path(source).expanduser()) if source else "",
            "output": str(Path(output).expanduser()),
            "csv": str(Path(csv_path).expanduser()) if csv_path else "",
            "device": self.device_var.get(),
            "ort_provider": self.ort_provider_var.get(),
            "coreml_cache_dir": str(Path(self.coreml_cache_var.get().strip()).expanduser()) if self.coreml_cache_var.get().strip() else None,
            "settings": settings,
        }

    def _settings_from_form(self) -> StreamSettings:
        chunk_sec = self._positive_float(self.chunk_sec_var, "Chunk sec")
        tokenizer_chunk_sec = self._nonnegative_float(self.tokenizer_chunk_sec_var, "Tokenizer chunk")
        effective_tokenizer_chunk_sec = tokenizer_chunk_sec if tokenizer_chunk_sec > 0 else chunk_sec
        tokenizer_left_context_sec = self._nonnegative_float(self.tokenizer_left_context_sec_var, "Tokenizer left context")
        tokenizer_right_context_sec = self._nonnegative_float(self.tokenizer_right_context_sec_var, "Tokenizer right context")
        if effective_tokenizer_chunk_sec + tokenizer_left_context_sec + tokenizer_right_context_sec > 30:
            raise ValueError("Tokenizer chunk plus left/right context must be 30 seconds or less.")
        return StreamSettings(
            chunk_sec=chunk_sec,
            tokenizer_chunk_sec=tokenizer_chunk_sec if tokenizer_chunk_sec > 0 else None,
            tokenizer_left_context_sec=tokenizer_left_context_sec,
            tokenizer_right_context_sec=tokenizer_right_context_sec,
            history_sec=self._nonnegative_float(self.history_sec_var, "History sec"),
            mel_overlap_sec=self._nonnegative_float(self.mel_overlap_sec_var, "Mel overlap sec"),
            delayed_commit_sec=self._nonnegative_float(self.delayed_commit_sec_var, "Delayed commit sec"),
            audio_declick_ms=self._nonnegative_float(self.audio_declick_ms_var, "De-click ms"),
            audio_blend_ms=self._nonnegative_float(self.audio_blend_ms_var, "Audio blend ms"),
            flow_context=self.flow_context_var.get(),
            hift_mode=self.hift_mode_var.get(),
            disable_prompt_kv_cache=not self.prompt_cache_var.get(),
            disable_history_kv_cache=not self.history_cache_var.get(),
        )

    def _positive_float(self, variable, name: str) -> float:
        value = self._float(variable, name)
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")
        return value

    def _nonnegative_float(self, variable, name: str) -> float:
        value = self._float(variable, name)
        if value < 0:
            raise ValueError(f"{name} must be 0 or greater.")
        return value

    def _float(self, variable, name: str) -> float:
        try:
            return float(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error

    def _offline_worker(self, config: dict) -> None:
        try:
            self._post("status", "Loading model...")
            model = self._get_model(config)
            self._post("status", "Preparing prompt...")
            context = prepare_stream_context(model, config["prompt"], config["settings"])
            for line in stream_context_log_lines(model, config["settings"], context):
                self._post("log", line)
            source_tokenizer = AsyncSourceTokenizer(
                model,
                config["source"],
                context.tokenizer_chunk_sec,
                left_context_sec=config["settings"].tokenizer_left_context_sec,
                right_context_sec=config["settings"].tokenizer_right_context_sec,
            )
            source_tokenizer.start()
            try:
                speech, rows = run_window_stream(
                    model=model,
                    source_stream=source_tokenizer,
                    prompt_token=context.prompt_token,
                    prompt_feat=context.prompt_feat,
                    embedding=context.embedding,
                    chunk_tokens=context.chunk_tokens,
                    history_tokens=context.history_tokens,
                    overlap_tokens=context.overlap_tokens,
                    delayed_commit_tokens=context.delayed_commit_tokens,
                    audio_declick_samples=context.audio_declick_samples,
                    max_audio_blend_samples=context.max_audio_blend_samples,
                    flow_streaming=context.flow_streaming,
                    hift_mode=config["settings"].hift_mode,
                    use_prompt_kv_cache=context.use_prompt_kv_cache,
                    use_history_kv_cache=context.use_history_kv_cache,
                    prompt_cache_len=context.prompt_cache_len,
                    prompt_cache_steps=context.prompt_cache_steps,
                    log_fn=lambda message: self._post("log", message),
                    stop_event=self.offline_stop_event,
                    on_audio_chunk=lambda speech_chunk, row: self._post_metrics(row),
                )
            finally:
                source_tokenizer.join()
            source_stats = source_tokenizer.stats()
            if self.offline_stop_event is not None and self.offline_stop_event.is_set():
                self._post("log", "Offline job stopped by user.")
            write_wav(config["output"], speech, model.sample_rate)
            if config["csv"]:
                write_rows(config["csv"], rows)
            for line in offline_summary_lines(model, source_stats, rows, config["output"]):
                self._post("log", line)
            self._post("status", "Offline benchmark finished.")
        except Exception as error:
            self._post_exception("Offline benchmark failed", error)
        finally:
            self._post("finished", "offline")

    def _realtime_worker(self, config: dict) -> None:
        source = None
        player = None
        try:
            self._post("status", "Loading model...")
            model = self._get_model(config)
            self._post("status", "Preparing prompt...")
            context = prepare_stream_context(model, config["prompt"], config["settings"])
            for line in stream_context_log_lines(model, config["settings"], context):
                self._post("log", line)
            player = RealtimeAudioPlayer(
                sample_rate=model.sample_rate,
                output_device=config.get("output_device"),
                log_fn=lambda message: self._post("log", message),
            )
            player.start()
            source = MicrophoneSourceTokenizer(
                model=model,
                chunk_sec=context.tokenizer_chunk_sec,
                input_device=config.get("input_device"),
                left_context_sec=config["settings"].tokenizer_left_context_sec,
                right_context_sec=config["settings"].tokenizer_right_context_sec,
                log_fn=lambda message: self._post("log", message),
            )
            self.live_source = source
            self.live_player = player
            source.start()
            self._post("status", "Live stream is running.")

            def handle_chunk(speech_chunk: torch.Tensor, row: dict) -> None:
                if player is not None:
                    player.write(speech_chunk)
                    self._post_metrics(row, player.stats())

            run_window_stream(
                model=model,
                source_stream=source,
                prompt_token=context.prompt_token,
                prompt_feat=context.prompt_feat,
                embedding=context.embedding,
                chunk_tokens=context.chunk_tokens,
                history_tokens=context.history_tokens,
                overlap_tokens=context.overlap_tokens,
                delayed_commit_tokens=context.delayed_commit_tokens,
                audio_declick_samples=context.audio_declick_samples,
                max_audio_blend_samples=context.max_audio_blend_samples,
                flow_streaming=context.flow_streaming,
                hift_mode=config["settings"].hift_mode,
                use_prompt_kv_cache=context.use_prompt_kv_cache,
                use_history_kv_cache=context.use_history_kv_cache,
                prompt_cache_len=context.prompt_cache_len,
                prompt_cache_steps=context.prompt_cache_steps,
                on_audio_chunk=handle_chunk,
                log_fn=lambda message: self._post("log", message),
                stop_event=self.live_stop_event,
                collect_chunks=False,
            )
            self._post("log", f"live_source_stats={source.stats()}")
            self._post("status", "Live stream stopped.")
        except Exception as error:
            self._post_exception("Live stream failed", error)
        finally:
            if source is not None:
                source.stop()
                try:
                    source.join()
                except Exception:
                    pass
            if player is not None:
                player.stop()
            self.live_source = None
            self.live_player = None
            self._post("finished", "realtime")

    def _get_model(self, config: dict) -> VCOnlyModel:
        key = (
            str(Path(config["model_dir"]).expanduser()),
            config["device"],
            config["ort_provider"],
            str(Path(config["coreml_cache_dir"]).expanduser()) if config["coreml_cache_dir"] else "",
        )
        with self.model_lock:
            if self.model is None or self.model_key != key:
                self._post("log", f"Loading model from {key[0]}")
                self.model = VCOnlyModel(
                    key[0],
                    device=config["device"],
                    ort_provider=config["ort_provider"],
                    coreml_cache_dir=config["coreml_cache_dir"],
                )
                self.model_key = key
                self._post("log", "Model loaded.")
            return self.model

    def _post_metrics(self, row: dict, player_stats: dict | None = None) -> None:
        input_clock = row["end_token"] / 25.0
        lag = row["wall_end_seconds"] - input_clock
        payload = {
            "chunk": str(row["chunk"]),
            "rtf": f"{row['chunk_rtf']:.2f}",
            "lag": f"{lag:.2f}s",
            "buffer": "-",
            "underflows": "-",
        }
        if player_stats is not None:
            payload["buffer"] = f"{player_stats['buffer_seconds']:.2f}s"
            payload["underflows"] = str(player_stats["underflows"])
        self._post("metrics", payload)

    def _post_exception(self, title: str, error: Exception) -> None:
        import traceback

        self._post("log", f"{title}: {error}")
        self._post("log", traceback.format_exc())
        self._post("status", title)

    def _is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _set_running(self, mode: str, running: bool) -> None:
        self.active_mode = mode if running else None
        self.start_live_button.configure(state="disabled" if running else "normal")
        self.run_offline_button.configure(state="disabled" if running else "normal")
        self.stop_live_button.configure(state="normal" if running and mode == "realtime" else "disabled")
        self.stop_offline_button.configure(state="normal" if running and mode == "offline" else "disabled")
        self.status_var.set("Starting..." if running else "Ready")

    def _post(self, kind: str, payload: object) -> None:
        self.ui_queue.put((kind, payload))

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._log(str(payload))
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "metrics":
                metrics = payload
                self.metric_chunk_var.set(metrics["chunk"])
                self.metric_rtf_var.set(metrics["rtf"])
                self.metric_lag_var.set(metrics["lag"])
                self.metric_buffer_var.set(metrics["buffer"])
                self.metric_underflow_var.set(metrics["underflows"])
            elif kind == "finished":
                self._set_running(str(payload), False)
        self.root.after(100, self._poll_ui_queue)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _on_close(self) -> None:
        if self.live_stop_event is not None:
            self.live_stop_event.set()
        if self.offline_stop_event is not None:
            self.offline_stop_event.set()
        if self.live_source is not None:
            self.live_source.stop()
        if self.live_player is not None:
            self.live_player.stop()
        self.root.destroy()


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
