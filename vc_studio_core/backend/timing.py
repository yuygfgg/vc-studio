from __future__ import annotations

from .environment import configure_environment

configure_environment()

from cosyvoice.vc.model import VCOnlyModel


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
