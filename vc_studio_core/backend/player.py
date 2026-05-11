from __future__ import annotations

from collections import deque
import os
import threading
from typing import Callable

from .environment import configure_environment

configure_environment()

import numpy as np
import torch


class RealtimeAudioPlayer:
    def __init__(
        self,
        sample_rate: int,
        output_device: int | None = None,
        log_fn: Callable[[str], None] | None = None,
        prebuffer_seconds: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.log_fn = log_fn
        if prebuffer_seconds is None:
            try:
                prebuffer_seconds = float(os.environ.get("VC_STUDIO_REALTIME_PREBUFFER_SEC", "0.35"))
            except ValueError:
                prebuffer_seconds = 0.35
        self.prebuffer_seconds = max(0.0, float(prebuffer_seconds))
        self._prebuffer_samples = round(self.prebuffer_seconds * self.sample_rate)
        self._stream = None
        self._stream_lock = threading.Lock()
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._queued_samples = 0
        self._played_samples = 0
        self._underflows = 0
        self._has_written = False
        self._playback_started = self._prebuffer_samples <= 0
        self._start_requested = False

    def start(self) -> None:
        self._start_requested = True
        if self._prebuffer_samples <= 0:
            self._ensure_stream_started()

    def _ensure_stream_started(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                return
            try:
                import sounddevice as sd
            except ImportError as error:
                raise RuntimeError("sounddevice is required for realtime playback") from error
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,
                device=self.output_device,
                callback=self._callback,
            )
            self._stream = stream
            stream.start()

    def write(self, speech: torch.Tensor) -> None:
        audio = speech.detach().cpu().squeeze(0).numpy().astype(np.float32)
        audio = np.clip(audio, -0.999, 0.999)
        if audio.size == 0:
            return
        should_start = False
        with self._lock:
            self._chunks.append(audio)
            self._queued_samples += int(audio.size)
            self._has_written = True
            buffered = self._queued_samples - self._played_samples
            should_start = (
                self._start_requested
                and self._stream is None
                and buffered >= self._prebuffer_samples
            )
        if should_start:
            self._ensure_stream_started()

    def stop(self) -> None:
        with self._stream_lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def stats(self) -> dict:
        with self._lock:
            buffered = self._queued_samples - self._played_samples
            stream_started = self._stream is not None
            return {
                "buffer_seconds": max(0, buffered) / self.sample_rate,
                "played_seconds": self._played_samples / self.sample_rate,
                "underflows": self._underflows,
                "prebuffer_seconds": self.prebuffer_seconds,
                "playback_started": self._playback_started,
                "stream_started": stream_started,
            }

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status and self.log_fn is not None:
            self.log_fn(f"output status: {status}")
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            buffered = self._queued_samples - self._played_samples
            if not self._playback_started:
                if buffered < self._prebuffer_samples:
                    outdata[:, 0] = out
                    return
                self._playback_started = True
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
            if filled < frames and self._playback_started and self._has_written:
                self._underflows += 1
        outdata[:, 0] = out
