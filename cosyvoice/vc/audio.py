from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosyvoice_numba_cache")

import librosa
import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
import whisper
from scipy.signal import resample_poly


def read_mono(path: str | Path, sample_rate: int) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != sample_rate:
        wav = _resample(wav, sr, sample_rate)
    return torch.from_numpy(wav).float().unsqueeze(0)


def write_wav(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().cpu().squeeze(0).numpy()
    audio = np.clip(audio, -0.999, 0.999)
    sf.write(str(path), audio, sample_rate)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    divisor = math.gcd(orig_sr, target_sr)
    return resample_poly(wav, target_sr // divisor, orig_sr // divisor).astype(np.float32)


def ort_providers(kind: str = "auto", coreml_cache_dir: str | Path | None = None) -> list:
    available = set(onnxruntime.get_available_providers())
    if kind == "cpu":
        return ["CPUExecutionProvider"]
    if kind == "coreml":
        if "CoreMLExecutionProvider" not in available:
            raise RuntimeError("CoreMLExecutionProvider is not available in this onnxruntime build")
        return [_coreml_provider(coreml_cache_dir), "CPUExecutionProvider"]
    if kind == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if kind != "auto":
        raise ValueError(f"unknown ORT provider mode: {kind}")
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        return [_coreml_provider(coreml_cache_dir), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _coreml_provider(coreml_cache_dir: str | Path | None) -> tuple[str, dict[str, str]] | str:
    if coreml_cache_dir is None:
        return "CoreMLExecutionProvider"
    cache_dir = Path(coreml_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ("CoreMLExecutionProvider", {"ModelCacheDirectory": str(cache_dir)})


class AudioFeatureExtractor:
    def __init__(
        self,
        model_dir: str | Path,
        ort_provider: str = "auto",
        speech_tokenizer_name: str = "speech_tokenizer_v3.onnx",
        coreml_cache_dir: str | Path | None = None,
    ):
        model_dir = Path(model_dir)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ort_providers(ort_provider, coreml_cache_dir=coreml_cache_dir)
        self.campplus = onnxruntime.InferenceSession(
            str(model_dir / "campplus.onnx"),
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self.speech_tokenizer = onnxruntime.InferenceSession(
            str(model_dir / speech_tokenizer_name),
            sess_options=option,
            providers=providers,
        )

    def speech_token(self, wav_16k: torch.Tensor) -> torch.Tensor:
        if wav_16k.shape[1] / 16000 > 30:
            raise ValueError("speech tokenizer input is limited to <= 30s for this offline VC probe")
        feat = whisper.log_mel_spectrogram(wav_16k, n_mels=128)
        tokens = self.speech_tokenizer.run(
            None,
            {
                self.speech_tokenizer.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.speech_tokenizer.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten()
        return torch.from_numpy(np.asarray(tokens, dtype=np.int32).reshape(1, -1))

    def speaker_embedding(self, wav_16k: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.campplus.run(
            None,
            {self.campplus.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
        )[0].flatten()
        return torch.from_numpy(np.asarray(emb, dtype=np.float32).reshape(1, -1))


_mel_basis = {}
_hann_window = {}


def mel_spectrogram_24000(wav_24k: torch.Tensor) -> torch.Tensor:
    return _mel_spectrogram(
        wav_24k,
        n_fft=1920,
        num_mels=80,
        sample_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )


def _mel_spectrogram(
    wav: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None,
    center: bool,
) -> torch.Tensor:
    device_key = str(wav.device)
    mel_key = f"{sample_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_{device_key}"
    window_key = f"{win_size}_{device_key}"
    if mel_key not in _mel_basis:
        mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[mel_key] = torch.from_numpy(mel).float().to(wav.device)
    if window_key not in _hann_window:
        _hann_window[window_key] = torch.hann_window(win_size).to(wav.device)

    wav = torch.nn.functional.pad(
        wav.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.stft(
        wav,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[window_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(_mel_basis[mel_key], spec)
    return torch.log(torch.clamp(mel, min=1e-5))
