from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

from .environment import configure_environment

configure_environment()

import numpy as np
import torch

from cosyvoice.vc.audio import read_mono
from cosyvoice.vc.model import VCOnlyModel

from .vad import SileroVADGate, build_token_speech_mask


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
                    if self.vad_gate is not None:
                        if vad_is_speech:
                            self._vad_speech_chunks += 1
                        else:
                            self._vad_silence_chunks += 1
                            self._vad_silence_samples += current_end - start
                    if token.shape[1] > 0:
                        token_start = self._token_count
                        self._token_chunks.append(token)
                        self._token_count += token.shape[1]
                        if self.vad_gate is not None:
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
        self._audio_base_sample = 0
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
        total_samples = self._audio_base_sample + self._audio.shape[0]
        while self._next_start_sample < total_samples:
            start = self._next_start_sample
            target_current_end = start + self.chunk_samples
            if not final and total_samples < target_current_end + self.right_context_samples:
                break
            current_end = min(target_current_end, total_samples)
            if current_end <= start:
                break
            window_start = max(self._audio_base_sample, start - self.left_context_samples)
            window_end = min(total_samples, current_end + self.right_context_samples)
            start_local = start - self._audio_base_sample
            current_end_local = current_end - self._audio_base_sample
            window_start_local = window_start - self._audio_base_sample
            window_end_local = window_end - self._audio_base_sample
            vad_is_speech = True
            vad_seconds = 0.0
            if self.vad_gate is not None:
                vad_start = time.perf_counter()
                vad_is_speech = self.vad_gate.is_speech(self._audio[start_local:current_end_local])
                vad_seconds = time.perf_counter() - vad_start
            window_audio = self._audio[window_start_local:window_end_local].copy()
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
                if self.vad_gate is not None:
                    if vad_is_speech:
                        self._vad_speech_chunks += 1
                    else:
                        self._vad_silence_chunks += 1
                        self._vad_silence_samples += current_end - start
                if token.shape[1] > 0:
                    token_start = self._token_count
                    self._token_chunks.append(token)
                    self._token_count += token.shape[1]
                    if self.vad_gate is not None:
                        self._speech_spans.append((token_start, self._token_count, vad_is_speech))
                self._condition.notify_all()
            self._next_start_sample = current_end
        self._trim_audio_buffer()

    def _trim_audio_buffer(self) -> None:
        keep_from = max(0, self._next_start_sample - self.left_context_samples)
        if keep_from <= self._audio_base_sample:
            return
        drop = min(keep_from - self._audio_base_sample, self._audio.shape[0])
        if drop <= 0:
            return
        self._audio = self._audio[drop:].copy()
        self._audio_base_sample += drop

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
