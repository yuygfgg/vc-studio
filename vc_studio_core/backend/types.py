from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .environment import configure_environment

configure_environment()

import torch


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
    prompt_cache_max_mb: float = 1024.0
    prompt_cache_max_seconds: float = 0.0
    prompt_cache_dtype: str = "auto"
    prompt_cache_storage: str = "device"
    prompt_runtime_policy: str = "auto"
    enable_grouped_prompt: bool = False
    enable_grouped_prompt_cache: bool = False
    lavasr_enabled: bool = True
    lavasr_lowpass_hz: float = 7800.0


@dataclass(frozen=True)
class BackendConfig:
    model_dir: str
    prompt: str
    settings: StreamSettings
    voice_package: str = ""
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
    prompt_source_kind: str = "legacy_wav"
    voice_package_metadata: dict[str, Any] | None = None
    selected_branch_index: int | None = None
    prompt_cache_disabled_reason: str | None = None
    grouped_prompt_inputs: dict[str, Any] | None = None
    soft_prompt_inputs: dict[str, torch.Tensor] | None = None
    use_soft_prompt: bool = False
    use_grouped_prompt: bool = False
    use_grouped_prompt_cache: bool = False
