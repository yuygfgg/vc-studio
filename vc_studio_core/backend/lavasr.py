from __future__ import annotations

import os
import time
from typing import Any, Callable

from .environment import configure_environment

configure_environment()

import numpy as np
import torch
import torchaudio.functional as torchaudio_functional

from .types import StreamSettings


LAVASR_INPUT_SAMPLE_RATE = 16000
LAVASR_OUTPUT_SAMPLE_RATE = 48000
DEFAULT_LAVASR_DEVICE = "cpu"
DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE = 48000


def validate_lavasr_lowpass_hz(lowpass_hz: float, input_sample_rate: int = LAVASR_INPUT_SAMPLE_RATE) -> float:
    lowpass_hz = float(lowpass_hz)
    nyquist = input_sample_rate / 2.0
    if lowpass_hz <= 0:
        raise ValueError("LavaSR lowpass cutoff must be greater than 0 Hz.")
    if lowpass_hz > nyquist:
        raise ValueError(f"LavaSR lowpass cutoff must be <= {nyquist:g} Hz for {input_sample_rate} Hz input.")
    return lowpass_hz


class LavaSRBandwidthExtender:
    input_sample_rate = LAVASR_INPUT_SAMPLE_RATE
    output_sample_rate = LAVASR_OUTPUT_SAMPLE_RATE

    def __init__(self, lowpass_hz: float, device: str = DEFAULT_LAVASR_DEVICE):
        self.lowpass_hz = validate_lavasr_lowpass_hz(lowpass_hz)
        self.device = device
        try:
            from LavaSR.enhancer.linkwitz_merge import FastLRMerge
            from LavaSR.model import LavaEnhance2
        except ImportError as error:
            raise RuntimeError(
                "LavaSR bandwidth extension is enabled but LavaSR is not installed. "
                "Install requirements.txt, then restart VC Studio."
            ) from error

        model_path = resolve_lavasr_model_path()
        try:
            self.model = LavaEnhance2(model_path=model_path, device=device)
        except Exception as error:
            raise RuntimeError(
                "LavaSR model weights could not be loaded. First use may need network access "
                "to download YatharthS/LavaSR from Hugging Face."
            ) from error
        if hasattr(self.model, "bwe_model") and hasattr(self.model.bwe_model, "lr_refiner"):
            refiner = FastLRMerge(cutoff=int(round(self.lowpass_hz)))
            if hasattr(refiner, "to"):
                refiner = refiner.to(getattr(self.model, "device", device))
            self.model.bwe_model.lr_refiner = refiner

    def enhance(self, audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, dict[str, Any]]:
        start = time.perf_counter()
        source_samples = int(audio.shape[-1])
        lavasr_input = prepare_lavasr_input(
            audio,
            sample_rate=sample_rate,
            target_sample_rate=self.input_sample_rate,
            lowpass_hz=self.lowpass_hz,
        )
        with torch.inference_mode():
            enhanced = self.model.enhance(lavasr_input.to(self.device), denoise=False)
        enhanced = normalize_audio_tensor(enhanced)
        target_samples = round(source_samples * self.output_sample_rate / sample_rate)
        enhanced = fit_audio_length(enhanced, target_samples)
        seconds = time.perf_counter() - start
        return enhanced, {
            "lavasr_enabled": True,
            "lavasr_device": self.device,
            "lavasr_lowpass_hz": self.lowpass_hz,
            "lavasr_input_sample_rate": self.input_sample_rate,
            "lavasr_output_sample_rate": self.output_sample_rate,
            "lavasr_input_samples": int(lavasr_input.shape[-1]),
            "lavasr_output_samples": int(enhanced.shape[-1]),
            "lavasr_seconds": seconds,
        }


def resolve_lavasr_model_path(repo_id: str = "YatharthS/LavaSR") -> str:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return repo_id


def create_lavasr_extender(settings: StreamSettings, log_fn: Callable[[str], None] | None = None) -> LavaSRBandwidthExtender | None:
    if not settings.lavasr_enabled:
        return None
    extender = LavaSRBandwidthExtender(settings.lavasr_lowpass_hz)
    if log_fn is not None:
        log_fn(
            "lavasr_loaded=True "
            f"device={extender.device} "
            f"input_sample_rate={extender.input_sample_rate} "
            f"output_sample_rate={extender.output_sample_rate} "
            f"lowpass_hz={extender.lowpass_hz:g}"
        )
    return extender


def normalize_audio_tensor(audio: Any) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        tensor = audio.detach().cpu()
    else:
        tensor = torch.as_tensor(audio)
    tensor = tensor.float()
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == 2:
        if tensor.shape[0] == 1:
            return tensor
        return tensor.mean(dim=0, keepdim=True)
    return tensor.reshape(1, -1)


def prepare_lavasr_input(
    audio: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    lowpass_hz: float,
) -> torch.Tensor:
    audio = normalize_audio_tensor(audio)
    if sample_rate <= 0:
        raise ValueError("source sample rate must be positive")
    lowpass_hz = validate_lavasr_lowpass_hz(lowpass_hz, target_sample_rate)
    if sample_rate == target_sample_rate:
        return audio
    resample_nyquist = min(sample_rate, target_sample_rate) / 2.0
    rolloff = min(0.99, max(0.01, lowpass_hz / resample_nyquist))
    return torchaudio_functional.resample(
        audio,
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
        lowpass_filter_width=32,
        rolloff=rolloff,
    )


def fit_audio_length(audio: torch.Tensor, target_samples: int) -> torch.Tensor:
    target_samples = max(0, int(target_samples))
    if audio.shape[-1] == target_samples:
        return audio
    if audio.shape[-1] > target_samples:
        return audio[..., :target_samples]
    pad = target_samples - audio.shape[-1]
    return torch.nn.functional.pad(audio, (0, pad))


def add_lavasr_row_stats(row: dict, stats: dict[str, Any]) -> dict:
    updated = dict(row)
    lavasr_seconds = float(stats.get("lavasr_seconds", 0.0))
    updated["lavasr_seconds"] = lavasr_seconds
    updated["compute_seconds"] = float(updated.get("compute_seconds", 0.0)) + lavasr_seconds
    updated["wall_end_seconds"] = float(updated.get("wall_end_seconds", 0.0)) + lavasr_seconds
    input_seconds = float(updated.get("input_seconds", 0.0))
    updated["chunk_rtf"] = updated["compute_seconds"] / input_seconds if input_seconds > 0 else 0.0
    updated["output_seconds"] = stats.get("lavasr_output_samples", 0) / LAVASR_OUTPUT_SAMPLE_RATE
    return updated


def realtime_output_sample_rate(model_sample_rate: int) -> int:
    value = os.environ.get("VC_STUDIO_REALTIME_OUTPUT_SAMPLE_RATE", "auto").strip().lower()
    if value in {"", "auto"}:
        return DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE if model_sample_rate != DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE else model_sample_rate
    if value in {"model", "native", "source"}:
        return model_sample_rate
    try:
        sample_rate = int(float(value))
    except ValueError:
        return model_sample_rate
    return sample_rate if sample_rate > 0 else model_sample_rate


def resample_realtime_output(audio: torch.Tensor, input_sample_rate: int, output_sample_rate: int) -> torch.Tensor:
    if input_sample_rate == output_sample_rate:
        return audio
    audio = normalize_audio_tensor(audio)
    target_samples = round(audio.shape[-1] * output_sample_rate / input_sample_rate)
    resampled = torchaudio_functional.resample(
        audio,
        orig_freq=input_sample_rate,
        new_freq=output_sample_rate,
        lowpass_filter_width=16,
        rolloff=0.95,
    )
    return fit_audio_length(resampled, target_samples)


def add_realtime_resample_row_stats(
    row: dict,
    *,
    seconds: float,
    output_samples: int,
    output_sample_rate: int,
) -> dict:
    updated = dict(row)
    updated["output_resample_seconds"] = float(seconds)
    updated["compute_seconds"] = float(updated.get("compute_seconds", 0.0)) + float(seconds)
    updated["wall_end_seconds"] = float(updated.get("wall_end_seconds", 0.0)) + float(seconds)
    input_seconds = float(updated.get("input_seconds", 0.0))
    updated["chunk_rtf"] = updated["compute_seconds"] / input_seconds if input_seconds > 0 else 0.0
    updated["output_seconds"] = int(output_samples) / int(output_sample_rate) if output_sample_rate > 0 else 0.0
    updated["output_sample_rate"] = int(output_sample_rate)
    return updated
