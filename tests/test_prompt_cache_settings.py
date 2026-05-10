from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vc_studio_backend import (
    choose_prompt_cache_budget_frames,
    choose_full_prompt_cache_frames,
    prompt_cache_offload_kv_to_cpu,
    prompt_cache_storage_dtype,
)


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
