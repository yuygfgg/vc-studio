from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vc_studio_core.backend import (
    StreamSettings,
    choose_full_prompt_cache_frames,
    choose_prompt_cache_budget_frames,
    prompt_cache_offload_kv_to_cpu,
    prompt_cache_storage_dtype,
    realtime_output_sample_rate,
)


def test_stream_settings_is_constructible() -> None:
    settings = StreamSettings(
        chunk_sec=2.0,
        tokenizer_chunk_sec=None,
        tokenizer_left_context_sec=0.5,
        tokenizer_right_context_sec=0.2,
        history_sec=3.0,
        mel_overlap_sec=0.25,
        delayed_commit_sec=0.5,
        audio_declick_ms=0.0,
        audio_blend_ms=0.0,
        vad_enabled=False,
        vad_threshold=0.5,
        vad_min_speech_ms=100.0,
        vad_min_silence_ms=100.0,
        vad_speech_pad_ms=30.0,
        flow_context="streaming",
        hift_mode="stateful",
        disable_prompt_kv_cache=False,
        disable_history_kv_cache=False,
    )
    assert settings.chunk_sec == 2.0
    assert settings.prompt_runtime_policy == "auto"


class _DummyModel:
    def __init__(self, device_type: str = "cuda") -> None:
        self.device = torch.device(device_type)
        self.flow = SimpleNamespace(
            decoder=SimpleNamespace(
                estimator=SimpleNamespace(
                    static_chunk_size=50,
                    transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(heads=4, inner_dim=256)) for _ in range(2)],
                )
            )
        )


def test_prompt_cache_dtype_and_offload_from_ui_settings() -> None:
    model = _DummyModel("cpu")
    assert prompt_cache_storage_dtype(model, "float16") == torch.float16
    assert prompt_cache_storage_dtype(model, "float32") is None
    assert prompt_cache_offload_kv_to_cpu("cpu_offload") is True
    assert prompt_cache_offload_kv_to_cpu("device") is False
    assert prompt_cache_storage_dtype(_DummyModel("cuda"), "auto") == torch.float16
    assert prompt_cache_storage_dtype(_DummyModel("mps"), "auto") is None


def test_realtime_output_sample_rate_defaults_to_device_friendly_rate(monkeypatch) -> None:
    monkeypatch.delenv("VC_STUDIO_REALTIME_OUTPUT_SAMPLE_RATE", raising=False)
    assert realtime_output_sample_rate(24000) == 48000
    monkeypatch.setenv("VC_STUDIO_REALTIME_OUTPUT_SAMPLE_RATE", "model")
    assert realtime_output_sample_rate(24000) == 24000


def test_prompt_cache_budget_clips_by_seconds_and_memory() -> None:
    model = _DummyModel("cuda")
    frames, note = choose_prompt_cache_budget_frames(
        model,
        prompt_mel_frames=1000,
        branch_count=2,
        dtype_bytes=2,
        max_mb=8.0,
        max_seconds=1.5,
    )
    assert frames > 0
    assert frames % 50 == 0
    assert note is not None


def test_full_prompt_cache_disables_instead_of_clipping() -> None:
    model = _DummyModel("cuda")
    frames, note = choose_full_prompt_cache_frames(
        model,
        prompt_mel_frames=1000,
        branch_count=2,
        dtype_bytes=2,
        max_mb=0.01,
        max_seconds=0.0,
    )
    assert frames == 0
    assert note is not None
    assert "without cache preserves prompt quality" in note


def test_full_prompt_cache_requires_static_alignment() -> None:
    model = _DummyModel("cuda")
    frames, note = choose_full_prompt_cache_frames(
        model,
        prompt_mel_frames=75,
        dtype_bytes=2,
        max_mb=0.0,
        max_seconds=0.0,
    )
    assert frames == 0
    assert note is not None
    assert "without cache preserves prompt quality" in note
