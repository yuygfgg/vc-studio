from __future__ import annotations

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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/tmp/cosyvoice_hf_cache")
warnings.filterwarnings(
    "ignore",
    message=".*LoRACompatibleLinear.*PEFT backend.*",
    category=FutureWarning,
)

import numpy as np
import torch

from cosyvoice.vc.audio import mel_spectrogram_24000, read_mono, write_wav
from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat


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
    vad_enabled: bool
    vad_threshold: float
    vad_min_speech_ms: float
    vad_min_silence_ms: float
    vad_speech_pad_ms: float
    flow_context: str
    hift_mode: str
    disable_prompt_kv_cache: bool
    disable_history_kv_cache: bool


@dataclass(frozen=True)
class BackendConfig:
    model_dir: str
    prompt: str
    settings: StreamSettings
    source: str = ""
    output: str = "out/vc_streaming.wav"
    csv: str = ""
    device: str = "auto"
    ort_provider: str = "auto"
    coreml_cache_dir: str | None = None
    input_device: int | None = None
    output_device: int | None = None


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
        f"vad_enabled={settings.vad_enabled}",
        f"vad_threshold={settings.vad_threshold:.3f}",
        f"vad_min_speech_ms={settings.vad_min_speech_ms:.1f}",
        f"vad_min_silence_ms={settings.vad_min_silence_ms:.1f}",
        f"vad_speech_pad_ms={settings.vad_speech_pad_ms:.1f}",
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
        f"source_vad_enabled={source_stats['vad_enabled']}",
        f"source_vad_speech_chunks={source_stats['vad_speech_chunks']}",
        f"source_vad_silence_chunks={source_stats['vad_silence_chunks']}",
        f"source_vad_silence_seconds={source_stats['vad_silence_seconds']:.3f}",
        f"source_vad_compute_seconds={source_stats['vad_compute_seconds']:.3f}",
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


class SileroVADGate:
    sample_rate = 16000

    def __init__(
        self,
        threshold: float,
        min_speech_ms: float,
        min_silence_ms: float,
        speech_pad_ms: float,
    ):
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as error:
            raise RuntimeError(
                "Silero VAD is enabled but the optional silero-vad package is not installed. "
                "Install requirements.txt or run `pip install silero-vad`."
            ) from error
        self.threshold = threshold
        self.min_speech_ms = int(round(min_speech_ms))
        self.min_silence_ms = int(round(min_silence_ms))
        self.speech_pad_ms = int(round(speech_pad_ms))
        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps
        if hasattr(self.model, "eval"):
            self.model.eval()

    def is_speech(self, audio: torch.Tensor | np.ndarray) -> bool:
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio)
        else:
            wav = audio.detach().cpu()
        wav = wav.flatten().to(dtype=torch.float32)
        if wav.numel() == 0:
            return False
        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
        return bool(speech_timestamps)


def create_vad_gate(settings: StreamSettings) -> SileroVADGate | None:
    if not settings.vad_enabled:
        return None
    return SileroVADGate(
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )


def is_static_cache_aligned(history_tokens: int, model: VCOnlyModel) -> bool:
    if history_tokens <= 0:
        return False
    static_chunk_mel = model.flow.decoder.estimator.static_chunk_size
    return (history_tokens * model.token_mel_ratio) % static_chunk_mel == 0


def build_token_speech_mask(
    speech_spans: list[tuple[int, int, bool]],
    start_token: int,
    end_token: int,
) -> torch.Tensor:
    if end_token <= start_token:
        return torch.zeros(0, dtype=torch.bool)
    mask = torch.ones(end_token - start_token, dtype=torch.bool)
    for span_start, span_end, is_speech in speech_spans:
        if span_end <= start_token:
            continue
        if span_start >= end_token:
            break
        left = max(span_start, start_token) - start_token
        right = min(span_end, end_token) - start_token
        if right > left:
            mask[left:right] = is_speech
    return mask


class AsyncSourceTokenizer:
    token_rate = 25.0

    def __init__(
        self,
        model: VCOnlyModel,
        source_wav: str | Path,
        chunk_sec: float,
        left_context_sec: float = 0.0,
        right_context_sec: float = 0.0,
        vad_gate: SileroVADGate | None = None,
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
        self.vad_gate = vad_gate
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._token_chunks: list[torch.Tensor] = []
        self._speech_spans: list[tuple[int, int, bool]] = []
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
        self._vad_compute_seconds = 0.0
        self._vad_speech_chunks = 0
        self._vad_silence_chunks = 0
        self._vad_silence_samples = 0

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

    def speech_mask(self, start_token: int, end_token: int) -> torch.Tensor:
        if start_token < 0 or end_token < start_token:
            raise ValueError(f"invalid token range: {start_token}:{end_token}")
        self.wait_until(end_token)
        with self._condition:
            if end_token > self._token_count:
                raise RuntimeError(
                    f"requested speech mask {start_token}:{end_token}, "
                    f"but only {self._token_count} source tokens are available"
                )
            return build_token_speech_mask(self._speech_spans, start_token, end_token)

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
                "vad_enabled": self.vad_gate is not None,
                "vad_compute_seconds": self._vad_compute_seconds,
                "vad_speech_chunks": self._vad_speech_chunks,
                "vad_silence_chunks": self._vad_silence_chunks,
                "vad_silence_seconds": self._vad_silence_samples / 16000.0,
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
                vad_is_speech = True
                vad_seconds = 0.0
                if self.vad_gate is not None:
                    vad_start = time.perf_counter()
                    vad_is_speech = self.vad_gate.is_speech(wav_16k[:, start:current_end])
                    vad_seconds = time.perf_counter() - vad_start
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
                    self._vad_compute_seconds += vad_seconds
                    if vad_is_speech:
                        self._vad_speech_chunks += 1
                    else:
                        self._vad_silence_chunks += 1
                        self._vad_silence_samples += current_end - start
                    if token.shape[1] > 0:
                        token_start = self._token_count
                        self._token_chunks.append(token)
                        self._token_count += token.shape[1]
                        self._speech_spans.append((token_start, self._token_count, vad_is_speech))
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
        vad_gate: SileroVADGate | None = None,
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
        self.vad_gate = vad_gate
        self.log_fn = log_fn
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._token_chunks: list[torch.Tensor] = []
        self._speech_spans: list[tuple[int, int, bool]] = []
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
        self._vad_compute_seconds = 0.0
        self._vad_speech_chunks = 0
        self._vad_silence_chunks = 0
        self._vad_silence_samples = 0

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

    def speech_mask(self, start_token: int, end_token: int) -> torch.Tensor:
        if start_token < 0 or end_token < start_token:
            raise ValueError(f"invalid token range: {start_token}:{end_token}")
        self.wait_until(end_token)
        with self._condition:
            if end_token > self._token_count:
                raise RuntimeError(
                    f"requested speech mask {start_token}:{end_token}, "
                    f"but only {self._token_count} source tokens are available"
                )
            return build_token_speech_mask(self._speech_spans, start_token, end_token)

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
                "vad_enabled": self.vad_gate is not None,
                "vad_compute_seconds": self._vad_compute_seconds,
                "vad_speech_chunks": self._vad_speech_chunks,
                "vad_silence_chunks": self._vad_silence_chunks,
                "vad_silence_seconds": self._vad_silence_samples / 16000.0,
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
            vad_is_speech = True
            vad_seconds = 0.0
            if self.vad_gate is not None:
                vad_start = time.perf_counter()
                vad_is_speech = self.vad_gate.is_speech(self._audio[start:current_end])
                vad_seconds = time.perf_counter() - vad_start
            window_audio = self._audio[window_start:window_end].copy()
            chunk = torch.from_numpy(window_audio).float().unsqueeze(0)
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
                self._vad_compute_seconds += vad_seconds
                if vad_is_speech:
                    self._vad_speech_chunks += 1
                else:
                    self._vad_silence_chunks += 1
                    self._vad_silence_samples += current_end - start
                if token.shape[1] > 0:
                    token_start = self._token_count
                    self._token_chunks.append(token)
                    self._token_count += token.shape[1]
                    self._speech_spans.append((token_start, self._token_count, vad_is_speech))
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
    hop_samples = model.sample_rate // (25 * model.token_mel_ratio)
    hift_right_context_mel = hift_required_right_context_mel(model, hop_samples)
    flow_right_context_tokens = model.flow.pre_lookahead_len
    hift_right_context_tokens = math.ceil(hift_right_context_mel / model.token_mel_ratio)
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
    previous_vad_active = False
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
        right_context_tokens = max(
            flow_right_context_tokens,
            hift_right_context_tokens - delayed_commit_tokens - overlap_tokens,
        )
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
        vad_speech_ratio = 1.0
        vad_muted_samples = 0
        if getattr(source_stream, "vad_gate", None) is not None and hasattr(source_stream, "speech_mask"):
            token_speech_mask = source_stream.speech_mask(start_token, chunk_end)
            if token_speech_mask.numel() != current_tokens:
                raise RuntimeError(
                    f"VAD speech mask has unexpected length: need {current_tokens} tokens, "
                    f"got {token_speech_mask.numel()}"
                )
            if token_speech_mask.numel() > 0:
                vad_speech_ratio = float(token_speech_mask.float().mean().item())
            speech_chunk, vad_muted_samples, previous_vad_active = apply_vad_speech_gate(
                speech=speech_chunk,
                token_speech_mask=token_speech_mask,
                samples_per_token=model.token_mel_ratio * hop_samples,
                previous_active=previous_vad_active,
                fade_samples=max(1, round(model.sample_rate * 0.01)),
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
                vad_speech_ratio=vad_speech_ratio,
                vad_muted_samples=vad_muted_samples,
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


def apply_vad_speech_gate(
    speech: torch.Tensor,
    token_speech_mask: torch.Tensor,
    samples_per_token: int,
    previous_active: bool,
    fade_samples: int,
) -> tuple[torch.Tensor, int, bool]:
    if token_speech_mask.numel() == 0 or speech.shape[1] == 0:
        return speech, 0, previous_active
    raw_mask = token_speech_mask.to(device=speech.device, dtype=speech.dtype).repeat_interleave(samples_per_token)
    target_samples = speech.shape[1]
    if raw_mask.numel() < target_samples:
        pad_value = raw_mask[-1] if raw_mask.numel() > 0 else torch.tensor(0.0, device=speech.device, dtype=speech.dtype)
        raw_mask = torch.cat([raw_mask, pad_value.repeat(target_samples - raw_mask.numel())])
    raw_mask = raw_mask[:target_samples]
    audio_mask = raw_mask.clone()
    fade_samples = min(max(0, fade_samples), target_samples)
    if fade_samples > 1:
        active = raw_mask > 0.5
        if bool(active[0].item()) and not previous_active:
            end = min(fade_samples, target_samples)
            audio_mask[:end] = torch.minimum(
                audio_mask[:end],
                torch.linspace(0.0, 1.0, end, device=speech.device, dtype=speech.dtype),
            )
        if not bool(active[0].item()) and previous_active:
            end = min(fade_samples, target_samples)
            audio_mask[:end] = torch.maximum(
                audio_mask[:end],
                torch.linspace(1.0, 0.0, end, device=speech.device, dtype=speech.dtype),
            )
        transitions = torch.nonzero(active[1:] != active[:-1], as_tuple=False).flatten() + 1
        for index_tensor in transitions:
            index = int(index_tensor.item())
            end = min(index + fade_samples, target_samples)
            if end <= index:
                continue
            if bool(active[index].item()):
                audio_mask[index:end] = torch.minimum(
                    audio_mask[index:end],
                    torch.linspace(0.0, 1.0, end - index, device=speech.device, dtype=speech.dtype),
                )
            else:
                audio_mask[index:end] = torch.maximum(
                    audio_mask[index:end],
                    torch.linspace(1.0, 0.0, end - index, device=speech.device, dtype=speech.dtype),
                )
    muted_samples = int(torch.count_nonzero(raw_mask <= 0.5).item())
    if muted_samples == 0 and torch.all(audio_mask >= 1.0).item():
        return speech, 0, bool(token_speech_mask[-1].item())
    return speech * audio_mask.view(1, -1), muted_samples, bool(token_speech_mask[-1].item())


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
    vad_speech_ratio: float,
    vad_muted_samples: int,
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
        "vad_speech_ratio": vad_speech_ratio,
        "vad_muted_samples": vad_muted_samples,
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
        "vad_speech_ratio={vad_speech_ratio:.2f} vad_muted_samples={vad_muted_samples} "
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


def _noop_status(message: str) -> None:
    pass


def _noop_log(message: str) -> None:
    pass


def _noop_metrics(row: dict, player_stats: dict | None = None) -> None:
    pass


class VCStudioBackend:
    def __init__(self) -> None:
        self._model_lock = threading.Lock()
        self._model_key: tuple[str, str, str, str] | None = None
        self._model: VCOnlyModel | None = None
        self._live_lock = threading.Lock()
        self._live_source: MicrophoneSourceTokenizer | None = None
        self._live_player: RealtimeAudioPlayer | None = None

    def run_offline(
        self,
        config: BackendConfig,
        *,
        stop_event: threading.Event | None = None,
        status_fn: Callable[[str], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
        metrics_fn: Callable[[dict, dict | None], None] | None = None,
    ) -> None:
        status_fn = status_fn or _noop_status
        log_fn = log_fn or _noop_log
        metrics_fn = metrics_fn or _noop_metrics

        status_fn("Loading model...")
        model = self.get_model(config, log_fn=log_fn)
        status_fn("Preparing prompt...")
        context = prepare_stream_context(model, config.prompt, config.settings)
        for line in stream_context_log_lines(model, config.settings, context):
            log_fn(line)
        status_fn("Loading Silero VAD..." if config.settings.vad_enabled else "Preparing source...")
        vad_gate = create_vad_gate(config.settings)
        source_tokenizer = AsyncSourceTokenizer(
            model,
            config.source,
            context.tokenizer_chunk_sec,
            left_context_sec=config.settings.tokenizer_left_context_sec,
            right_context_sec=config.settings.tokenizer_right_context_sec,
            vad_gate=vad_gate,
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
                hift_mode=config.settings.hift_mode,
                use_prompt_kv_cache=context.use_prompt_kv_cache,
                use_history_kv_cache=context.use_history_kv_cache,
                prompt_cache_len=context.prompt_cache_len,
                prompt_cache_steps=context.prompt_cache_steps,
                log_fn=log_fn,
                stop_event=stop_event,
                on_audio_chunk=lambda speech_chunk, row: metrics_fn(row, None),
            )
        finally:
            source_tokenizer.join()
        source_stats = source_tokenizer.stats()
        if stop_event is not None and stop_event.is_set():
            log_fn("Offline job stopped by user.")
        write_wav(config.output, speech, model.sample_rate)
        if config.csv:
            write_rows(config.csv, rows)
        for line in offline_summary_lines(model, source_stats, rows, config.output):
            log_fn(line)
        status_fn("Offline benchmark finished.")

    def run_realtime(
        self,
        config: BackendConfig,
        *,
        stop_event: threading.Event | None = None,
        status_fn: Callable[[str], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
        metrics_fn: Callable[[dict, dict | None], None] | None = None,
    ) -> None:
        status_fn = status_fn or _noop_status
        log_fn = log_fn or _noop_log
        metrics_fn = metrics_fn or _noop_metrics
        source = None
        player = None
        try:
            status_fn("Loading model...")
            model = self.get_model(config, log_fn=log_fn)
            status_fn("Preparing prompt...")
            context = prepare_stream_context(model, config.prompt, config.settings)
            for line in stream_context_log_lines(model, config.settings, context):
                log_fn(line)
            status_fn("Loading Silero VAD..." if config.settings.vad_enabled else "Opening audio output...")
            vad_gate = create_vad_gate(config.settings)
            player = RealtimeAudioPlayer(
                sample_rate=model.sample_rate,
                output_device=config.output_device,
                log_fn=log_fn,
            )
            player.start()
            source = MicrophoneSourceTokenizer(
                model=model,
                chunk_sec=context.tokenizer_chunk_sec,
                input_device=config.input_device,
                left_context_sec=config.settings.tokenizer_left_context_sec,
                right_context_sec=config.settings.tokenizer_right_context_sec,
                vad_gate=vad_gate,
                log_fn=log_fn,
            )
            with self._live_lock:
                self._live_source = source
                self._live_player = player
            source.start()
            status_fn("Live stream is running.")

            def handle_chunk(speech_chunk: torch.Tensor, row: dict) -> None:
                if player is not None:
                    player.write(speech_chunk)
                    metrics_fn(row, player.stats())

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
                hift_mode=config.settings.hift_mode,
                use_prompt_kv_cache=context.use_prompt_kv_cache,
                use_history_kv_cache=context.use_history_kv_cache,
                prompt_cache_len=context.prompt_cache_len,
                prompt_cache_steps=context.prompt_cache_steps,
                on_audio_chunk=handle_chunk,
                log_fn=log_fn,
                stop_event=stop_event,
                collect_chunks=False,
            )
            log_fn(f"live_source_stats={source.stats()}")
            status_fn("Live stream stopped.")
        finally:
            if source is not None:
                source.stop()
                try:
                    source.join()
                except Exception:
                    pass
            if player is not None:
                player.stop()
            with self._live_lock:
                if self._live_source is source:
                    self._live_source = None
                if self._live_player is player:
                    self._live_player = None

    def stop_realtime(self) -> None:
        with self._live_lock:
            source = self._live_source
        if source is not None:
            source.stop()

    def shutdown(self) -> None:
        with self._live_lock:
            source = self._live_source
            player = self._live_player
        if source is not None:
            source.stop()
        if player is not None:
            player.stop()

    def get_model(
        self,
        config: BackendConfig,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> VCOnlyModel:
        log_fn = log_fn or _noop_log
        key = (
            str(Path(config.model_dir).expanduser()),
            config.device,
            config.ort_provider,
            str(Path(config.coreml_cache_dir).expanduser()) if config.coreml_cache_dir else "",
        )
        with self._model_lock:
            if self._model is None or self._model_key != key:
                log_fn(f"Loading model from {key[0]}")
                self._model = VCOnlyModel(
                    key[0],
                    device=config.device,
                    ort_provider=config.ort_provider,
                    coreml_cache_dir=config.coreml_cache_dir,
                )
                self._model_key = key
                log_fn("Model loaded.")
            return self._model


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
