from __future__ import annotations

from .environment import configure_environment

configure_environment()

import numpy as np
import torch

from .types import StreamSettings


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
