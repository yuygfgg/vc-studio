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
from typing import Any, Callable, Mapping

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/tmp/cosyvoice_hf_cache")
warnings.filterwarnings(
    "ignore",
    message=".*LoRACompatibleLinear.*PEFT backend.*",
    category=FutureWarning,
)

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as torchaudio_functional

from cosyvoice.vc.audio import mel_spectrogram_24000, read_mono, write_wav
from cosyvoice.vc.device import empty_cache as empty_device_cache
from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat
from cosyvoice.vc.voice_package import (
    APP_VERSION,
    FORMAT_NAME,
    FORMAT_VERSION,
    MEL_BINS,
    MODEL_FAMILY,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
    TOKENIZER_SAFE_SECONDS,
    TOKENIZER_SAMPLE_RATE,
    TOKEN_MEL_RATIO,
    TOKEN_RATE,
    VoicePromptInputs,
    VoicePromptBranch,
    l2_normalize_array,
    load_voice_package,
    model_compatibility_fields,
    new_package_id,
    save_voice_package,
    sha256_file,
    sharpen_weights,
    utc_now_iso,
)
from cosyvoice.vc.soft_prompt import SoftPromptTrainingConfig, distill_soft_prompt_v1


def prepare_prompt_inputs_from_wav(
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


def prepare_prompt_inputs_from_package(
    model: VCOnlyModel,
    package_path: str | Path,
) -> VoicePromptInputs:
    return load_voice_package(package_path, model=model, device=model.device)


def prepare_prompt_inputs(
    model: VCOnlyModel,
    prompt_wav: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return prepare_prompt_inputs_from_wav(model, prompt_wav)


def select_runtime_prompt_inputs(prompt_inputs: VoicePromptInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    branch_index = prompt_inputs.dominant_branch_index()
    branch = prompt_inputs.branches[branch_index]
    return branch.prompt_token, branch.prompt_feat, prompt_inputs.fused_embedding, branch_index


def select_soft_prompt_runtime_inputs(
    prompt_inputs: VoicePromptInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, torch.Tensor]]:
    if prompt_inputs.soft_prompt is None:
        raise ValueError("voice package does not contain soft prompt tensors")
    soft_prompt = prompt_inputs.soft_prompt
    selected_branch_index = prompt_inputs.dominant_branch_index()
    prompt_token = torch.empty(
        (1, 0),
        dtype=torch.int32,
        device=soft_prompt.prompt_mu.device,
    )
    soft_prompt_inputs = {
        "soft_prompt_mu": soft_prompt.prompt_mu,
        "soft_prompt_feat": soft_prompt.prompt_feat,
        "soft_speaker_embedding": soft_prompt.speaker_embedding,
    }
    return (
        prompt_token,
        soft_prompt.prompt_feat,
        soft_prompt.speaker_embedding,
        selected_branch_index,
        soft_prompt_inputs,
    )


def prepare_grouped_prompt_cache(
    model: VCOnlyModel,
    prompt_inputs: VoicePromptInputs,
    streaming: bool,
    max_cache_mel_frames: int | None = None,
    cache_storage_dtype: torch.dtype | None = None,
    offload_kv_to_cpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    sharpened_weights = prompt_inputs.sharpened_weights()
    active_indices = [index for index, weight in enumerate(sharpened_weights) if weight > 0.0]
    if not active_indices:
        raise ValueError("voice package has no active prompt branches")
    dominant_index = prompt_inputs.dominant_branch_index()
    if dominant_index not in active_indices:
        dominant_index = active_indices[0]

    branch_specs = []
    for branch_index in active_indices:
        branch = prompt_inputs.branches[branch_index]
        branch_token, branch_feat = trim_prompt_to_static_cache(
            branch.prompt_token,
            branch.prompt_feat,
            model,
            max_mel_frames=max_cache_mel_frames,
        )
        expected_cache_len = trim_cache_mel_frames_to_static(model, branch_feat.shape[1])
        if expected_cache_len <= 0:
            raise ValueError(
                "multi-branch voice packages require each active reference to be long enough for static prompt cache"
            )
        branch_specs.append(
            {
                "branch_index": branch_index,
                "branch": branch,
                "prompt_token": branch_token,
                "prompt_feat": branch_feat,
                "weight": float(sharpened_weights[branch_index]),
                "expected_cache_len": int(expected_cache_len),
            }
        )

    branch_weights = torch.tensor(
        [record["weight"] for record in branch_specs],
        dtype=torch.float32,
        device=model.device,
    )
    cache_target_device = torch.device("cpu") if offload_kv_to_cpu else None
    steps = None
    dominant_token = None
    dominant_feat = None
    dominant_cache_len = 0
    final_prompt_x = None
    flow_token_tail = None

    for branch_position, spec in enumerate(branch_specs):
        branch_index = spec["branch_index"]
        branch = spec["branch"]
        branch_token = spec["prompt_token"]
        branch_feat = spec["prompt_feat"]
        token_len = torch.tensor([branch_token.shape[1]], dtype=torch.int32, device=model.device)
        feat_len = torch.tensor([branch_feat.shape[1]], dtype=torch.int32, device=model.device)
        cache_len, cache = model.flow.prepare_prompt_cache(
            prompt_token=branch_token,
            prompt_token_len=token_len,
            prompt_feat=branch_feat,
            prompt_feat_len=feat_len,
            embedding=branch.embedding,
            streaming=streaming,
            cache_storage_dtype=cache_storage_dtype,
            cache_target_device=cache_target_device,
            keep_prompt_inputs=branch_index == dominant_index,
        )
        if cache is None or cache_len <= 0:
            raise ValueError(
                "multi-branch voice packages require each active reference to be long enough for static prompt cache"
            )
        spec["cache_len"] = int(cache_len)
        if steps is None:
            steps = _create_sequential_grouped_prompt_cache_steps(
                cache,
                branch_weights=branch_weights,
            )
        _append_branch_prompt_cache_to_sequential_steps(
            steps,
            cache,
            branch_position=branch_position,
            branch_index=branch_index,
            cache_len=int(cache_len),
            storage_dtype=cache_storage_dtype,
            target_device=cache_target_device,
        )
        if branch_index == dominant_index:
            dominant_token = branch_token
            dominant_feat = branch_feat
            dominant_cache_len = int(cache_len)
            final_prompt_x = cache["final_prompt_x"]
            flow_token_tail = cache.get("flow_token_tail")
            _attach_dominant_grouped_prompt_inputs(
                steps,
                cache,
                fused_embedding=prompt_inputs.fused_embedding,
                model=model,
                storage_dtype=cache_storage_dtype,
                target_device=cache_target_device,
                attention_temperature=float(prompt_inputs.metadata.get("attention_temperature", 1.0)),
            )

        del cache
        sync_device(model.device)
        empty_device_cache(model.device)

    if dominant_token is None or dominant_feat is None or final_prompt_x is None or steps is None:
        raise ValueError("dominant prompt branch was not prepared")

    for step in steps:
        prompt_cache = step["prompt_cache"]
        prompt_cache["prompt_len"] = int(dominant_cache_len)
        prompt_cache["base_prompt_len"] = int(dominant_cache_len)
        prompt_cache["branch_indices"] = [record["branch_index"] for record in branch_specs]
        prompt_cache["branch_cache_lens"] = [
            int(record.get("cache_len", record["expected_cache_len"])) for record in branch_specs
        ]

    grouped_cache = {
        "grouped_branch_attention": True,
        "steps": steps,
        "final_prompt_x": final_prompt_x,
        "cache_len": int(dominant_cache_len),
        "base_cache_len": int(dominant_cache_len),
        "history_cache_len": 0,
        "base_cache": None,
        "flow_token_tail": flow_token_tail,
        "branch_indices": [record["branch_index"] for record in branch_specs],
        "branch_weights": [record["weight"] for record in branch_specs],
    }
    return dominant_token, dominant_feat, prompt_inputs.fused_embedding, dominant_index, int(dominant_cache_len), grouped_cache


def _create_sequential_grouped_prompt_cache_steps(
    branch_cache: dict[str, Any],
    *,
    branch_weights: torch.Tensor,
) -> list[dict[str, Any]]:
    steps = []
    for _ in branch_cache["steps"]:
        steps.append(
            {
                "prompt_cache": {
                    "grouped_branch_attention": True,
                    "grouped_attention_mode": "sequential",
                    "sequential_branch_caches": [],
                    "branch_weights": branch_weights,
                    "branch_indices": [],
                    "branch_cache_lens": [],
                    "prompt_len": 0,
                    "base_prompt_len": 0,
                },
                "prompt_inputs": None,
            }
        )
    return steps


def _append_branch_prompt_cache_to_sequential_steps(
    steps: list[dict[str, Any]],
    branch_cache: dict[str, Any],
    *,
    branch_position: int,
    branch_index: int,
    cache_len: int,
    storage_dtype: torch.dtype | None,
    target_device: torch.device | None,
) -> None:
    mask_device = target_device or branch_cache["steps"][0]["prompt_cache"]["kv"][0][0].device
    prompt_mask = torch.ones((cache_len,), dtype=torch.bool, device=mask_device)
    for step, branch_step in zip(steps, branch_cache["steps"]):
        source_kv = branch_step["prompt_cache"]["kv"]
        stored_kv = []
        for kv_index, (key, value) in enumerate(source_kv):
            stored_kv.append(
                (
                    optimize_tensor(key.detach(), storage_dtype, target_device),
                    optimize_tensor(value.detach(), storage_dtype, target_device),
                )
            )
            source_kv[kv_index] = (None, None)
        branch_step["prompt_cache"]["kv"] = []
        step["prompt_cache"]["sequential_branch_caches"].append(
            {
                "branch_position": int(branch_position),
                "branch_index": int(branch_index),
                "cache_len": int(cache_len),
                "prompt_mask": prompt_mask,
                "kv": stored_kv,
            }
    )


def _attach_dominant_grouped_prompt_inputs(
    steps: list[dict[str, Any]],
    branch_cache: dict[str, Any],
    *,
    fused_embedding: torch.Tensor,
    model: VCOnlyModel,
    storage_dtype: torch.dtype | None,
    target_device: torch.device | None,
    attention_temperature: float,
) -> None:
    fused_spks = torch.nn.functional.normalize(fused_embedding, dim=1)
    fused_spks = model.flow.spk_embed_affine_layer(fused_spks)
    for step, dominant_step in zip(steps, branch_cache["steps"]):
        prompt_inputs_for_step = dict(dominant_step["prompt_inputs"])
        spks_in = torch.zeros_like(prompt_inputs_for_step["spks_in"])
        spks_in[: fused_spks.shape[0]] = fused_spks
        prompt_inputs_for_step["spks_in"] = spks_in
        prompt_cache = step["prompt_cache"]
        prompt_cache["input_embed_cache"] = optimize_tensor_tree(
            dominant_step["prompt_cache"].get("input_embed_cache"),
            storage_dtype,
            target_device,
        )
        prompt_cache["attention_temperature"] = float(attention_temperature)
        step["prompt_inputs"] = prompt_inputs_for_step


def prepare_grouped_prompt_runtime_inputs(prompt_inputs: VoicePromptInputs) -> dict[str, Any]:
    sharpened_weights = prompt_inputs.sharpened_weights()
    active_indices = [index for index, weight in enumerate(sharpened_weights) if weight > 0.0]
    if not active_indices:
        raise ValueError("voice package has no active prompt branches")
    dominant_index = prompt_inputs.dominant_branch_index()
    if dominant_index not in active_indices:
        dominant_index = active_indices[0]
    return {
        "prompt_tokens": [prompt_inputs.branches[index].prompt_token for index in active_indices],
        "prompt_feats": [prompt_inputs.branches[index].prompt_feat for index in active_indices],
        "embeddings": [prompt_inputs.branches[index].embedding for index in active_indices],
        "branch_weights": [float(sharpened_weights[index]) for index in active_indices],
        "branch_indices": active_indices,
        "dominant_branch_index": dominant_index,
        "dominant_branch_position": active_indices.index(dominant_index),
        "attention_temperature": float(prompt_inputs.metadata.get("attention_temperature", 1.0)),
    }


def prompt_cache_memory_limit_bytes(max_mb: float | None = None) -> int:
    if max_mb is None:
        value = os.environ.get("VC_STUDIO_PROMPT_CACHE_MAX_MB", "1024").strip()
        try:
            megabytes = float(value)
        except ValueError:
            megabytes = 1024.0
    else:
        megabytes = float(max_mb)
    if megabytes <= 0:
        return 0
    return int(megabytes * 1024 * 1024)


def grouped_prompt_cache_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("VC_STUDIO_ENABLE_GROUPED_PROMPT_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def grouped_prompt_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("VC_STUDIO_ENABLE_GROUPED_PROMPT", "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_prompt_runtime_policy(policy: str | None) -> str:
    value = (policy or "auto").strip().lower()
    aliases = {
        "soft_prompt": "soft",
        "soft_prompt_v1": "soft",
        "grouped_prompt": "grouped",
        "grouped_branch_attention": "grouped",
        "dominant_branch": "dominant",
        "dominant_branch_prompt": "dominant",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "soft", "grouped", "dominant"}:
        raise ValueError("prompt runtime policy must be auto, soft, grouped, or dominant")
    return value


def prompt_cache_storage_dtype(model: VCOnlyModel, mode: str | None = None) -> torch.dtype | None:
    value = (mode or os.environ.get("VC_STUDIO_PROMPT_CACHE_DTYPE", "auto")).strip().lower()
    if value in {"", "auto"}:
        if model.device.type == "cuda":
            return torch.float16
        return None
    if value in {"float32", "fp32", "none", "off"}:
        return None
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return None


def prompt_cache_dtype_bytes(storage_dtype: torch.dtype | None) -> int:
    if storage_dtype in {torch.float16, torch.bfloat16}:
        return 2
    return 4


def prompt_cache_offload_kv_to_cpu(storage: str | None = None) -> bool:
    if storage is not None:
        return storage.strip().lower() in {"cpu", "cpu_offload", "offload"}
    return os.environ.get("VC_STUDIO_PROMPT_CACHE_OFFLOAD", "").strip().lower() in {"1", "true", "yes", "on", "cpu"}


def prompt_cache_max_seconds(max_seconds: float | None = None) -> float | None:
    if max_seconds is not None:
        seconds = float(max_seconds)
        return seconds if seconds > 0 else None
    value = os.environ.get("VC_STUDIO_PROMPT_CACHE_MAX_SECONDS", "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return seconds


def choose_prompt_cache_budget_frames(
    model: VCOnlyModel,
    prompt_mel_frames: int,
    *,
    branch_count: int = 1,
    dtype_bytes: int = 4,
    max_mb: float | None = None,
    max_seconds: float | None = None,
) -> tuple[int, str | None]:
    trimmed_frames = trim_cache_mel_frames_to_static(model, prompt_mel_frames)
    if trimmed_frames <= 0:
        return 0, "prompt is too short for static prompt KV cache"

    notes = []
    max_cache_seconds = prompt_cache_max_seconds(max_seconds)
    if max_cache_seconds is not None:
        second_limited = trim_cache_mel_frames_to_static(model, int(max_cache_seconds * TOKEN_RATE * TOKEN_MEL_RATIO))
        if second_limited > 0 and second_limited < trimmed_frames:
            notes.append(
                f"prompt KV cache clipped by max seconds from "
                f"{trimmed_frames / (TOKEN_RATE * TOKEN_MEL_RATIO):.2f}s to "
                f"{second_limited / (TOKEN_RATE * TOKEN_MEL_RATIO):.2f}s"
            )
            trimmed_frames = second_limited

    limit = prompt_cache_memory_limit_bytes(max_mb)
    estimated = estimate_prompt_cache_bytes(model, trimmed_frames, branch_count=branch_count, dtype_bytes=dtype_bytes)
    if limit > 0 and estimated > limit:
        bytes_per_frame = max(1, estimate_prompt_cache_bytes(model, 1, branch_count=branch_count, dtype_bytes=dtype_bytes))
        budget_frames = trim_cache_mel_frames_to_static(model, limit // bytes_per_frame)
        if budget_frames <= 0:
            return 0, (
                f"prompt KV cache estimate {format_bytes(estimated)} exceeds limit {format_bytes(limit)} "
                "and even one static cache chunk does not fit"
            )
        if budget_frames < trimmed_frames:
            notes.append(
                f"prompt KV cache clipped by memory budget from "
                f"{trimmed_frames / (TOKEN_RATE * TOKEN_MEL_RATIO):.2f}s to "
                f"{budget_frames / (TOKEN_RATE * TOKEN_MEL_RATIO):.2f}s "
                f"({format_bytes(estimate_prompt_cache_bytes(model, budget_frames, branch_count=branch_count, dtype_bytes=dtype_bytes))} "
                f"<= {format_bytes(limit)})"
            )
            trimmed_frames = budget_frames
    return trimmed_frames, "; ".join(notes) if notes else None


def choose_full_prompt_cache_frames(
    model: VCOnlyModel,
    prompt_mel_frames: int,
    *,
    branch_count: int = 1,
    dtype_bytes: int = 4,
    max_mb: float | None = None,
    max_seconds: float | None = None,
) -> tuple[int, str | None]:
    full_frames = int(prompt_mel_frames)
    aligned_frames = trim_cache_mel_frames_to_static(model, full_frames)
    if aligned_frames <= 0:
        return 0, "prompt is too short for static prompt KV cache"
    if aligned_frames != full_frames:
        return 0, (
            "prompt KV cache disabled because the full prompt is not aligned to the "
            "DiT static cache grid; running without cache preserves prompt quality"
        )

    max_cache_seconds = prompt_cache_max_seconds(max_seconds)
    if max_cache_seconds is not None:
        max_frames = trim_cache_mel_frames_to_static(model, int(max_cache_seconds * TOKEN_RATE * TOKEN_MEL_RATIO))
        if max_frames <= 0 or full_frames > max_frames:
            return 0, (
                f"prompt KV cache disabled because the full prompt "
                f"({full_frames / (TOKEN_RATE * TOKEN_MEL_RATIO):.2f}s) exceeds "
                f"prompt cache max seconds ({max_cache_seconds:.2f}s); running without cache preserves prompt quality"
            )

    limit = prompt_cache_memory_limit_bytes(max_mb)
    estimated = estimate_prompt_cache_bytes(model, full_frames, branch_count=branch_count, dtype_bytes=dtype_bytes)
    if limit > 0 and estimated > limit:
        return 0, (
            f"prompt KV cache estimate {format_bytes(estimated)} exceeds limit {format_bytes(limit)}; "
            "running without cache preserves prompt quality"
        )
    return full_frames, None


def trim_cache_mel_frames_to_static(model: VCOnlyModel, mel_frames: int) -> int:
    static_chunk_mel = model.flow.decoder.estimator.static_chunk_size
    if static_chunk_mel <= 0:
        return max(0, int(mel_frames))
    return max(0, int(mel_frames) // static_chunk_mel * static_chunk_mel)


def estimate_prompt_cache_bytes(
    model: VCOnlyModel,
    prompt_mel_frames: int,
    *,
    branch_count: int = 1,
    diffusion_steps: int = 10,
    cfg_batch: int = 2,
    dtype_bytes: int = 4,
) -> int:
    estimator = model.flow.decoder.estimator
    layers = len(estimator.transformer_blocks)
    if layers <= 0 or prompt_mel_frames <= 0 or branch_count <= 0:
        return 0
    first_attn = estimator.transformer_blocks[0].attn
    heads = int(first_attn.heads)
    head_dim = int(first_attn.inner_dim // first_attn.heads)
    kv_count = 2
    return int(branch_count * diffusion_steps * layers * kv_count * cfg_batch * heads * prompt_mel_frames * head_dim * dtype_bytes)


def optimize_prompt_cache_storage(
    prompt_cache_steps,
    storage_dtype: torch.dtype | None,
    *,
    offload_kv_to_cpu: bool = False,
) -> None:
    if prompt_cache_steps is None:
        return
    target_device = torch.device("cpu") if offload_kv_to_cpu else None
    for step in prompt_cache_steps.get("steps", []):
        prompt_cache = step.get("prompt_cache", {})
        if "kv" in prompt_cache:
            prompt_cache["kv"] = optimize_kv_list(prompt_cache["kv"], storage_dtype, target_device)
        if "history_kv" in prompt_cache:
            prompt_cache["history_kv"] = optimize_kv_list(prompt_cache["history_kv"], storage_dtype, target_device)
        if "grouped_kv" in prompt_cache:
            prompt_cache["grouped_kv"] = optimize_kv_list(prompt_cache["grouped_kv"], storage_dtype, target_device)
        if offload_kv_to_cpu and "grouped_prompt_mask" in prompt_cache:
            prompt_cache["grouped_prompt_mask"] = prompt_cache["grouped_prompt_mask"].cpu()
        for branch_cache in prompt_cache.get("sequential_branch_caches", []):
            branch_cache["kv"] = optimize_kv_list(branch_cache["kv"], storage_dtype, target_device)
            if offload_kv_to_cpu and "prompt_mask" in branch_cache:
                branch_cache["prompt_mask"] = branch_cache["prompt_mask"].cpu()
        input_embed_cache = prompt_cache.get("input_embed_cache")
        if isinstance(input_embed_cache, dict):
            prompt_cache["input_embed_cache"] = optimize_tensor_tree(input_embed_cache, storage_dtype, target_device)


def optimize_kv_list(
    kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    storage_dtype: torch.dtype | None,
    target_device: torch.device | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            optimize_tensor(key, storage_dtype, target_device),
            optimize_tensor(value, storage_dtype, target_device),
        )
        for key, value in kv_list
    ]


def optimize_tensor_tree(value, storage_dtype: torch.dtype | None, target_device: torch.device | None):
    if isinstance(value, torch.Tensor):
        return optimize_tensor(value, storage_dtype, target_device)
    if isinstance(value, dict):
        return {key: optimize_tensor_tree(item, storage_dtype, target_device) for key, item in value.items()}
    if isinstance(value, list):
        return [optimize_tensor_tree(item, storage_dtype, target_device) for item in value]
    if isinstance(value, tuple):
        return tuple(optimize_tensor_tree(item, storage_dtype, target_device) for item in value)
    return value


def optimize_tensor(
    tensor: torch.Tensor,
    storage_dtype: torch.dtype | None,
    target_device: torch.device | None,
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor.to(device=target_device) if target_device is not None else tensor
    kwargs: dict[str, Any] = {}
    if storage_dtype is not None:
        kwargs["dtype"] = storage_dtype
    if target_device is not None:
        kwargs["device"] = target_device
    if not kwargs:
        return tensor
    return tensor.to(**kwargs)


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "MiB", "GiB"):
        if unit == "B":
            if amount < 1024:
                return f"{int(amount)} B"
            amount /= 1024.0
            continue
        if amount < 1024 or unit == "GiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def trim_prompt_to_static_cache(
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    model: VCOnlyModel,
    max_mel_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_chunk_mel = model.flow.decoder.estimator.static_chunk_size
    token_multiple = max(1, static_chunk_mel // model.token_mel_ratio)
    token_frames = (prompt_token.shape[1] // token_multiple) * token_multiple
    if max_mel_frames is not None:
        max_token_frames = max(0, int(max_mel_frames) // model.token_mel_ratio)
        max_token_frames = (max_token_frames // token_multiple) * token_multiple
        token_frames = min(token_frames, max_token_frames)
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


def prepare_stream_context(
    model: VCOnlyModel,
    prompt_source: str | Path | VoicePromptInputs,
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
    prompt_cache_requested = flow_streaming and not settings.disable_prompt_kv_cache
    use_prompt_kv_cache = prompt_cache_requested
    prompt_runtime_policy = normalize_prompt_runtime_policy(settings.prompt_runtime_policy)

    prompt_prepare_start = time.perf_counter()
    prompt_source_kind = "legacy_wav"
    voice_package_metadata = None
    selected_branch_index = None
    voice_inputs = None
    soft_prompt_inputs = None
    use_soft_prompt = False
    if isinstance(prompt_source, VoicePromptInputs):
        voice_inputs = prompt_source
        voice_package_metadata = voice_inputs.metadata
        prompt_source_kind = "voice_package"
    else:
        prompt_path = Path(prompt_source).expanduser()
        if prompt_path.suffix.lower() == ".cvvoice":
            voice_inputs = prepare_prompt_inputs_from_package(model, prompt_path)
            voice_package_metadata = voice_inputs.metadata
            prompt_source_kind = "voice_package"
        else:
            prompt_token, prompt_feat, embedding = prepare_prompt_inputs_from_wav(model, prompt_path)
    if voice_inputs is not None:
        use_soft_prompt = voice_inputs.has_soft_prompt() and prompt_runtime_policy in {"auto", "soft"}
        if prompt_runtime_policy == "soft" and not voice_inputs.has_soft_prompt():
            raise ValueError("prompt runtime policy soft requires a voice package with soft_prompt_v1")
        if use_soft_prompt:
            prompt_token, prompt_feat, embedding, selected_branch_index, soft_prompt_inputs = select_soft_prompt_runtime_inputs(
                voice_inputs
            )
        else:
            prompt_token, prompt_feat, embedding, selected_branch_index = select_runtime_prompt_inputs(voice_inputs)
    prompt_cache_disabled_reason = None
    prompt_cache_notes: list[str] = []
    cache_storage_dtype = prompt_cache_storage_dtype(model, settings.prompt_cache_dtype)
    cache_storage_dtype_bytes = prompt_cache_dtype_bytes(cache_storage_dtype)
    offload_prompt_kv = prompt_cache_offload_kv_to_cpu(settings.prompt_cache_storage)
    if cache_storage_dtype is not None:
        prompt_cache_notes.append(f"prompt KV storage dtype={str(cache_storage_dtype).replace('torch.', '')}")
    if offload_prompt_kv:
        prompt_cache_notes.append("prompt KV storage offloaded to CPU")
    active_branch_count = len(voice_inputs.active_branch_indices()) if voice_inputs is not None else 1
    use_grouped_prompt = (
        voice_inputs is not None
        and not use_soft_prompt
        and len(voice_inputs.branches) > 1
        and active_branch_count > 1
        and (
            prompt_runtime_policy == "grouped"
            or (
                prompt_runtime_policy == "auto"
                and grouped_prompt_enabled(settings.enable_grouped_prompt)
            )
        )
    )
    grouped_prompt_inputs = prepare_grouped_prompt_runtime_inputs(voice_inputs) if use_grouped_prompt and voice_inputs is not None else None
    grouped_cache_mel_frames = None
    use_grouped_prompt_cache = (
        use_grouped_prompt
        and prompt_cache_requested
        and (
            prompt_runtime_policy == "grouped"
            or grouped_prompt_cache_enabled(settings.enable_grouped_prompt_cache)
        )
    )

    if use_grouped_prompt and prompt_cache_requested and not use_grouped_prompt_cache:
        prompt_cache_notes.append(
            "grouped prompt is enabled without grouped prompt KV cache; grouped attention will be recomputed per window"
        )

    if use_grouped_prompt_cache and voice_inputs is not None:
        branch_frames = [voice_inputs.branches[index].prompt_feat.shape[1] for index in voice_inputs.active_branch_indices()]
        unaligned_branch = next(
            (
                frames
                for frames in branch_frames
                if trim_cache_mel_frames_to_static(model, frames) != frames
            ),
            None,
        )
        if unaligned_branch is not None:
            grouped_cache_mel_frames = 0
            budget_note = (
                "grouped prompt KV cache disabled because at least one full prompt branch is not aligned "
                "to the DiT static cache grid; running without cache preserves prompt quality"
            )
        else:
            grouped_cache_mel_frames, budget_note = choose_full_prompt_cache_frames(
                model,
                max(branch_frames, default=0),
                branch_count=len(branch_frames),
                dtype_bytes=cache_storage_dtype_bytes,
                max_mb=settings.prompt_cache_max_mb,
                max_seconds=settings.prompt_cache_max_seconds,
            )
        if grouped_cache_mel_frames <= 0:
            use_grouped_prompt_cache = False
            use_prompt_kv_cache = False
            prompt_cache_notes.append(
                (budget_note or "grouped prompt KV cache does not fit the configured memory budget")
                + "; keeping grouped prompt quality path without cache"
            )
        else:
            if budget_note:
                prompt_cache_notes.append(budget_note)
    elif use_grouped_prompt:
        use_prompt_kv_cache = False

    if use_prompt_kv_cache and not use_grouped_prompt:
        cache_mel_frames, budget_note = choose_full_prompt_cache_frames(
            model,
            prompt_feat.shape[1],
            dtype_bytes=cache_storage_dtype_bytes,
            max_mb=settings.prompt_cache_max_mb,
            max_seconds=settings.prompt_cache_max_seconds,
        )
        if cache_mel_frames <= 0:
            use_prompt_kv_cache = False
            prompt_cache_notes.append(budget_note or "prompt KV cache does not fit the configured memory budget")
        elif budget_note:
            prompt_cache_notes.append(budget_note)
    if prompt_cache_disabled_reason is None and prompt_cache_notes:
        prompt_cache_disabled_reason = "; ".join(prompt_cache_notes)

    history_cache_enabled = (
        flow_streaming
        and not settings.disable_history_kv_cache
        and use_prompt_kv_cache
        and (not use_grouped_prompt or use_grouped_prompt_cache)
        and is_static_cache_aligned(history_tokens, model)
    )
    sync_device(model.device)
    prompt_prepare_seconds = time.perf_counter() - prompt_prepare_start

    prompt_cache_len = 0
    prompt_cache_steps = None
    prompt_cache_prepare_seconds = 0.0
    if use_grouped_prompt_cache and voice_inputs is not None:
        cache_start = time.perf_counter()
        prompt_token, prompt_feat, embedding, selected_branch_index, prompt_cache_len, prompt_cache_steps = (
            prepare_grouped_prompt_cache(
                model,
                voice_inputs,
                flow_streaming,
                max_cache_mel_frames=grouped_cache_mel_frames,
                cache_storage_dtype=cache_storage_dtype,
                offload_kv_to_cpu=offload_prompt_kv,
            )
        )
        sync_device(model.device)
        prompt_cache_prepare_seconds = time.perf_counter() - cache_start
    elif use_prompt_kv_cache and use_soft_prompt and soft_prompt_inputs is not None:
        cache_start = time.perf_counter()
        prompt_cache_len, prompt_cache_steps = model.flow.prepare_soft_prompt_cache(
            soft_prompt_mu=soft_prompt_inputs["soft_prompt_mu"],
            soft_prompt_feat=soft_prompt_inputs["soft_prompt_feat"],
            soft_speaker_embedding=soft_prompt_inputs["soft_speaker_embedding"],
            streaming=flow_streaming,
            cache_storage_dtype=cache_storage_dtype,
            cache_target_device=torch.device("cpu") if offload_prompt_kv else None,
            keep_prompt_inputs=True,
        )
        optimize_prompt_cache_storage(
            prompt_cache_steps,
            cache_storage_dtype,
            offload_kv_to_cpu=offload_prompt_kv,
        )
        sync_device(model.device)
        prompt_cache_prepare_seconds = time.perf_counter() - cache_start
    elif use_prompt_kv_cache:
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
            cache_storage_dtype=cache_storage_dtype,
            cache_target_device=torch.device("cpu") if offload_prompt_kv else None,
            keep_prompt_inputs=True,
        )
        optimize_prompt_cache_storage(
            prompt_cache_steps,
            cache_storage_dtype,
            offload_kv_to_cpu=offload_prompt_kv,
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
        prompt_source_kind=prompt_source_kind,
        voice_package_metadata=voice_package_metadata,
        selected_branch_index=selected_branch_index,
        prompt_cache_disabled_reason=prompt_cache_disabled_reason,
        grouped_prompt_inputs=grouped_prompt_inputs,
        soft_prompt_inputs=soft_prompt_inputs,
        use_soft_prompt=use_soft_prompt,
        use_grouped_prompt=use_grouped_prompt,
        use_grouped_prompt_cache=use_grouped_prompt_cache,
    )


def stream_context_log_lines(
    model: VCOnlyModel,
    settings: StreamSettings,
    context: PreparedStreamContext,
) -> list[str]:
    lines = [
        f"device={model.device}",
        f"flow_context={settings.flow_context}",
        f"hift_mode={settings.hift_mode}",
        f"prompt_source={context.prompt_source_kind}",
        f"prompt_kv_cache={context.use_prompt_kv_cache}",
        f"prompt_cache_max_mb={settings.prompt_cache_max_mb:g}",
        f"prompt_cache_max_seconds={settings.prompt_cache_max_seconds:g}",
        f"prompt_cache_dtype={settings.prompt_cache_dtype}",
        f"prompt_cache_storage={settings.prompt_cache_storage}",
        f"prompt_runtime_policy={normalize_prompt_runtime_policy(settings.prompt_runtime_policy)}",
        f"soft_prompt={context.use_soft_prompt}",
        f"grouped_prompt={context.use_grouped_prompt}",
        f"grouped_prompt_cache={context.use_grouped_prompt_cache}",
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
        f"lavasr_enabled={settings.lavasr_enabled}",
        f"lavasr_lowpass_hz={settings.lavasr_lowpass_hz:g}",
        f"vad_enabled={settings.vad_enabled}",
        f"vad_threshold={settings.vad_threshold:.3f}",
        f"vad_min_speech_ms={settings.vad_min_speech_ms:.1f}",
        f"vad_min_silence_ms={settings.vad_min_silence_ms:.1f}",
        f"vad_speech_pad_ms={settings.vad_speech_pad_ms:.1f}",
        f"prompt_prepare_seconds={context.prompt_prepare_seconds:.3f}",
        f"prompt_cache_prepare_seconds={context.prompt_cache_prepare_seconds:.3f}",
    ]
    if context.prompt_cache_disabled_reason:
        lines.append(f"prompt_cache_disabled_reason={context.prompt_cache_disabled_reason}")
    if context.use_grouped_prompt_cache and isinstance(context.prompt_cache_steps, dict):
        steps = context.prompt_cache_steps.get("steps") or []
        if steps:
            mode = steps[0].get("prompt_cache", {}).get("grouped_attention_mode", "vectorized")
            lines.append(f"grouped_prompt_cache_mode={mode}")
    if context.voice_package_metadata is not None:
        metadata = context.voice_package_metadata
        runtime_policy = "dominant_branch_prompt_with_fused_embedding"
        if context.use_grouped_prompt:
            runtime_policy = "grouped_branch_attention_output_mix"
            if context.use_grouped_prompt_cache:
                runtime_policy += "_cached"
            else:
                runtime_policy += "_uncached"
        elif context.use_soft_prompt:
            runtime_policy = "soft_prompt_v1"
            if context.use_prompt_kv_cache:
                runtime_policy += "_cached"
        lines.extend(
            [
                f"voice_package_id={metadata.get('package_id')}",
                f"voice_package_branch_count={metadata.get('branch_count')}",
                f"voice_package_selected_branch={context.selected_branch_index}",
                f"voice_package_fusion_mode={metadata.get('fusion_mode')}",
                f"voice_package_prompt_fusion={metadata.get('prompt_fusion_algorithm')}",
                f"voice_package_runtime_prompt_policy={runtime_policy}",
            ]
        )
        if context.use_soft_prompt:
            lines.extend(
                [
                    f"soft_prompt_seconds={float(metadata.get('soft_prompt_seconds', 0.0)):.3f}",
                    f"soft_prompt_mel_frames={metadata.get('soft_prompt_mel_frames')}",
                ]
            )
    return lines


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


def offline_summary_lines(
    model: VCOnlyModel,
    source_stats: dict,
    rows: list[dict],
    output: str | Path,
    output_sample_rate: int | None = None,
    postprocess_stats: dict[str, Any] | None = None,
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
    output_sample_rate = output_sample_rate or model.sample_rate
    lines = [
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
        f"output_sample_rate={output_sample_rate}",
        f"output={Path(output).resolve()}",
    ]
    if postprocess_stats is not None:
        lines.extend(
            [
                f"lavasr_output_sample_rate={postprocess_stats['lavasr_output_sample_rate']}",
                f"lavasr_lowpass_hz={postprocess_stats['lavasr_lowpass_hz']:g}",
                f"lavasr_compute_seconds={postprocess_stats['lavasr_seconds']:.3f}",
                f"lavasr_rtf={postprocess_stats['lavasr_seconds'] / source_seconds if source_seconds > 0 else 0.0:.3f}",
            ]
        )
    return lines


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
    grouped_prompt_inputs: dict[str, Any] | None = None,
    soft_prompt_inputs: dict[str, torch.Tensor] | None = None,
    on_audio_chunk: Callable[[torch.Tensor, dict], dict | None] | None = None,
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
            grouped_prompt_inputs=grouped_prompt_inputs,
            soft_prompt_inputs=soft_prompt_inputs,
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
            updated_row = on_audio_chunk(speech_chunk_cpu, rows[-1])
            if updated_row is not None:
                rows[-1] = updated_row
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
    grouped_prompt_inputs: dict[str, Any] | None,
    soft_prompt_inputs: dict[str, torch.Tensor] | None,
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
            grouped_prompt_inputs=grouped_prompt_inputs,
            soft_prompt_inputs=soft_prompt_inputs,
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


def create_voice_package(
    model: VCOnlyModel,
    prompt_wavs: list[str | Path],
    output_path: str | Path,
    options: MappingLike | None = None,
    status_fn: Callable[[str], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    status_fn = status_fn or _noop_status
    log_fn = log_fn or _noop_log
    options_dict = dict(options or {})
    prompt_paths = [Path(path).expanduser() for path in prompt_wavs]
    if not prompt_paths:
        raise ValueError("at least one reference WAV is required")

    fusion_mode = str(options_dict.get("fusion_mode", "equal_weight"))
    if fusion_mode not in {"equal_weight", "duration_weight", "manual_weight"}:
        raise ValueError("fusion_mode must be equal_weight, duration_weight, or manual_weight")
    manual_weights = list(options_dict.get("manual_weights", []))
    branch_weight_gamma = float(options_dict.get("branch_weight_gamma", 1.0))
    attention_temperature = float(options_dict.get("attention_temperature", 1.0))
    canonical_prompt_length_seconds = float(options_dict.get("canonical_prompt_length_seconds", 10.0))
    enable_soft_prompt = bool(options_dict.get("enable_soft_prompt", False))
    soft_prompt_seconds = float(options_dict.get("soft_prompt_seconds", 15.0))
    soft_prompt_steps = int(options_dict.get("soft_prompt_steps", 300))
    soft_prompt_teacher_mode = str(options_dict.get("soft_prompt_teacher_mode", "grouped_branch_attention"))
    soft_prompt_activation_checkpointing = str(options_dict.get("soft_prompt_activation_checkpointing", "auto"))
    soft_prompt_checkpoint_segments = int(options_dict.get("soft_prompt_checkpoint_segments", 3))
    skip_unreadable = bool(options_dict.get("skip_unreadable", False))
    if branch_weight_gamma <= 0:
        raise ValueError("branch_weight_gamma must be greater than 0")
    if attention_temperature <= 0:
        raise ValueError("attention_temperature must be greater than 0")
    if canonical_prompt_length_seconds <= 0:
        raise ValueError("canonical_prompt_length_seconds must be greater than 0")
    if enable_soft_prompt:
        if soft_prompt_seconds <= 0:
            raise ValueError("soft_prompt_seconds must be greater than 0")
        if soft_prompt_steps < 0:
            raise ValueError("soft_prompt_steps must be 0 or greater")
        if soft_prompt_teacher_mode not in {"grouped_branch_attention", "init_only"}:
            raise ValueError("soft_prompt_teacher_mode must be grouped_branch_attention or init_only")
        if soft_prompt_activation_checkpointing not in {"off", "auto", "on"}:
            raise ValueError("soft_prompt_activation_checkpointing must be off, auto, or on")
        if soft_prompt_checkpoint_segments <= 0:
            raise ValueError("soft_prompt_checkpoint_segments must be greater than 0")

    accepted: list[dict[str, Any]] = []
    total_reference_seconds = 0.0
    for source_index, path in enumerate(prompt_paths):
        status_fn(f"Preparing reference {source_index + 1}/{len(prompt_paths)}...")
        try:
            source = _prepare_voice_reference(model, path, source_index, log_fn=log_fn)
        except Exception as error:
            if skip_unreadable:
                log_fn(f"Skipping unreadable reference {path}: {error}")
                continue
            raise
        total_reference_seconds += float(source["duration_seconds"])
        accepted.append(source)

    if not accepted:
        raise ValueError("no reference WAV files were accepted")

    raw_weights = _resolve_raw_fusion_weights(accepted, fusion_mode, manual_weights)
    positive_sum = sum(weight for weight in raw_weights if weight > 0)
    if positive_sum <= 0:
        raise ValueError("at least one accepted reference must have a positive fusion weight")
    normalized_weights = [weight / positive_sum if weight > 0 else 0.0 for weight in raw_weights]
    sharpened_weights = sharpen_weights(normalized_weights, branch_weight_gamma)

    tensors: dict[str, np.ndarray] = {}
    prompt_sources = []
    normalized_embeddings = []
    for branch_index, source in enumerate(accepted):
        embedding = source["embedding"]
        normalized_embedding = l2_normalize_array(embedding)
        normalized_embeddings.append(normalized_embedding)
        speaker_key, token_key, feat_key = (
            f"branch_{branch_index}_speaker_embedding",
            f"branch_{branch_index}_prompt_token",
            f"branch_{branch_index}_prompt_feat",
        )
        tensors[speaker_key] = normalized_embedding.astype(np.float32)
        tensors[token_key] = source["prompt_token"].astype(np.int32)
        tensors[feat_key] = source["prompt_feat"].astype(np.float32)
        source_metadata = {
            "source_index": int(source["source_index"]),
            "branch_index": branch_index,
            "display_name": str(options_dict.get("display_name") or source["path"].stem),
            "path_basename": source["path"].name,
            "file_sha256": source["file_sha256"],
            "original_sample_rate": int(source["original_sample_rate"]),
            "duration_seconds": float(source["duration_seconds"]),
            "accepted_seconds": float(source["accepted_seconds"]),
            "token_frames": int(source["prompt_token"].shape[1]),
            "mel_frames": int(source["prompt_feat"].shape[1]),
            "embedding_norm": float(source["embedding_norm"]),
            "fusion_weight_raw": float(raw_weights[branch_index]),
            "fusion_weight_normalized": float(normalized_weights[branch_index]),
            "branch_weight_after_gamma": float(sharpened_weights[branch_index]),
            "is_masked": bool(normalized_weights[branch_index] <= 0.0),
        }
        prompt_sources.append(source_metadata)
        log_fn(
            "reference branch={branch} seconds={seconds:.3f} tokens={tokens} mel_frames={mel} "
            "raw_weight={raw:.4f} normalized_weight={norm:.4f} masked={masked}".format(
                branch=branch_index,
                seconds=source_metadata["accepted_seconds"],
                tokens=source_metadata["token_frames"],
                mel=source_metadata["mel_frames"],
                raw=source_metadata["fusion_weight_raw"],
                norm=source_metadata["fusion_weight_normalized"],
                masked=source_metadata["is_masked"],
            )
        )

    fused_embedding = _fuse_speaker_embeddings(normalized_embeddings, normalized_weights)
    tensors["fused_speaker_embedding"] = fused_embedding.astype(np.float32)
    accepted_reference_seconds = sum(float(source["accepted_seconds"]) for source in accepted)
    prompt_token_frames = sum(int(source["prompt_token"].shape[1]) for source in accepted)
    prompt_mel_frames = sum(int(source["prompt_feat"].shape[1]) for source in accepted)
    metadata = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "package_id": str(options_dict.get("package_id") or new_package_id()),
        "created_at": str(options_dict.get("created_at") or utc_now_iso()),
        "created_by": str(options_dict.get("created_by") or "vc_studio"),
        "app_version": APP_VERSION,
        "model_family": MODEL_FAMILY,
        "model_dir_name": Path(model.model_dir).name,
        **model_compatibility_fields(model),
        "branch_count": len(accepted),
        "prompt_seconds": float(accepted_reference_seconds),
        "prompt_token_frames": int(prompt_token_frames),
        "prompt_mel_frames": int(prompt_mel_frames),
        "fused_speaker_embedding_norm": float(np.linalg.norm(fused_embedding)),
        "tensor_sha256": "",
        "reference_count": len(accepted),
        "total_reference_seconds": float(total_reference_seconds),
        "accepted_reference_seconds": float(accepted_reference_seconds),
        "prompt_sources": prompt_sources,
        "fusion_mode": fusion_mode,
        "speaker_embedding_fusion_algorithm": "l2_normalize_each_then_weighted_average_then_l2_normalize",
        "prompt_fusion_algorithm": "grouped_branch_attention_output_mix",
        "experimental_prompt_fusion_algorithms": [
            "concat_branch_prompt_kv_with_attention_logit_bias",
            "concat_branch_prompt_kv_with_value_scaling",
        ],
        "fusion_weight_sum_raw": float(positive_sum),
        "fusion_weight_normalization": "divide_by_sum_of_positive_raw_weights",
        "attention_weight_zero_policy": "mask_branch",
        "branch_weight_gamma": float(branch_weight_gamma),
        "attention_temperature": float(attention_temperature),
        "single_speaker_package": True,
        "source_position_policy": "canonical_prompt_length",
        "canonical_prompt_length_seconds": float(canonical_prompt_length_seconds),
        "canonical_prompt_length_mel_frames": int(round(canonical_prompt_length_seconds * TOKEN_RATE * TOKEN_MEL_RATIO)),
        "prompt_length_normalization_policy": "reject_over_limit_until_vad_segmentation",
        "flow_token_tail_fusion_policy": "dominant_branch",
        "dominant_branch_tie_breaker": "lowest_branch_index",
        "sample_rate": SAMPLE_RATE,
        "tokenizer_sample_rate": TOKENIZER_SAMPLE_RATE,
        "token_rate": TOKEN_RATE,
        "token_mel_ratio": TOKEN_MEL_RATIO,
        "speaker_embedding_dim": SPEAKER_EMBEDDING_DIM,
        "mel_bins": MEL_BINS,
        "feature_dtype": "float32",
    }
    for optional_key in (
        "display_name",
        "author",
        "license",
        "tags",
        "short_description",
        "long_description",
        "quality_notes",
        "privacy_notes",
    ):
        if optional_key in options_dict:
            metadata[optional_key] = options_dict[optional_key]

    if enable_soft_prompt:
        soft_config = SoftPromptTrainingConfig(
            enabled=True,
            seconds=soft_prompt_seconds,
            steps=soft_prompt_steps,
            teacher_mode=soft_prompt_teacher_mode,
            activation_checkpointing=soft_prompt_activation_checkpointing,
            checkpoint_segments=soft_prompt_checkpoint_segments,
            distill_layer=int(options_dict.get("soft_prompt_distill_layer", 6)),
            source_window_min_mel_frames=int(options_dict.get("soft_prompt_source_window_min_mel_frames", 50)),
            source_window_max_mel_frames=int(options_dict.get("soft_prompt_source_window_max_mel_frames", 100)),
            learning_rate=float(options_dict.get("soft_prompt_learning_rate", 1e-2)),
            hidden_mse_weight=float(options_dict.get("soft_prompt_hidden_mse_weight", 1.0)),
            prompt_delta_l2_weight=float(options_dict.get("soft_prompt_delta_l2_weight", 1e-4)),
            prompt_smoothness_weight=float(options_dict.get("soft_prompt_smoothness_weight", 1e-4)),
            gradient_clip_norm=float(options_dict.get("soft_prompt_gradient_clip_norm", 1.0)),
            validation_windows=int(options_dict.get("soft_prompt_validation_windows", 2)),
            validation_every=int(options_dict.get("soft_prompt_validation_every", 50)),
            low_memory_free_gb=float(options_dict.get("soft_prompt_low_memory_free_gb", 14.0)),
            teacher_branch_subset_size=int(options_dict.get("soft_prompt_teacher_branch_subset_size", 0)),
        )
        voice_inputs = _voice_prompt_inputs_from_package_parts(
            tensors=tensors,
            metadata=metadata,
            prompt_sources=prompt_sources,
            fused_embedding=fused_embedding,
            device=model.device,
        )
        log_fn(
            "soft_prompt_requested=True "
            f"seconds={soft_config.seconds:g} steps={soft_config.steps} "
            f"teacher={soft_config.teacher_mode} checkpointing={soft_config.activation_checkpointing} "
            f"segments={soft_config.checkpoint_segments}"
        )
        soft_result = distill_soft_prompt_v1(
            model,
            voice_inputs,
            soft_config,
            status_fn=status_fn,
            log_fn=log_fn,
        )
        tensors.update(soft_result.tensors)
        metadata.update(soft_result.metadata)
        log_fn(
            "soft_prompt_created=True "
            f"mel_frames={metadata['soft_prompt_mel_frames']} "
            f"seconds={metadata['soft_prompt_seconds']:.3f} "
            f"steps={soft_result.training_steps} "
            f"final_loss={soft_result.final_loss:.6f}"
        )
        sync_device(model.device)
        empty_device_cache(model.device)

    status_fn("Saving voice package...")
    final_metadata = save_voice_package(
        output_path,
        tensors=tensors,
        metadata=metadata,
        portrait_path=options_dict.get("portrait_path") or None,
    )
    log_fn(f"voice_package={Path(output_path).expanduser().resolve()}")
    log_fn(f"voice_package_branches={final_metadata['branch_count']} tensor_sha256={final_metadata['tensor_sha256']}")
    status_fn("Voice package created.")
    return final_metadata


def _prepare_voice_reference(
    model: VCOnlyModel,
    path: Path,
    source_index: int,
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"reference WAV does not exist: {path}")
    try:
        info = sf.info(str(path))
    except Exception as error:
        raise ValueError(f"unsupported or unreadable audio file: {path}") from error
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"reference WAV is empty: {path}")
    duration_seconds = float(info.frames) / float(info.samplerate)
    if duration_seconds <= 0:
        raise ValueError(f"reference WAV is empty: {path}")
    if duration_seconds > TOKENIZER_SAFE_SECONDS:
        raise ValueError(
            f"reference WAV is {duration_seconds:.2f}s, but tokenizer-safe package creation is limited "
            f"to {TOKENIZER_SAFE_SECONDS:.0f}s until VAD segmentation is implemented: {path}"
        )

    wav_16k = read_mono(path, TOKENIZER_SAMPLE_RATE)
    wav_24k = read_mono(path, SAMPLE_RATE)
    if wav_16k.numel() == 0 or wav_24k.numel() == 0:
        raise ValueError(f"reference WAV is empty after decoding: {path}")

    token = model.features.speech_token(wav_16k).to(dtype=torch.int32).cpu()
    if token.shape[1] <= 0:
        raise ValueError(f"reference WAV produced zero speech tokens: {path}")
    feat = mel_spectrogram_24000(wav_24k).squeeze(0).transpose(0, 1).unsqueeze(0).to(dtype=torch.float32).cpu()
    token, feat = align_prompt_token_feat(token, feat, model.token_mel_ratio)
    embedding = model.features.speaker_embedding(wav_16k).to(dtype=torch.float32).cpu().numpy()
    embedding_norm = float(np.linalg.norm(embedding))
    log_fn(
        f"accepted reference index={source_index} file={path.name} duration={duration_seconds:.3f}s "
        f"tokens={token.shape[1]} mel_frames={feat.shape[1]} embedding_norm={embedding_norm:.6f}"
    )
    return {
        "source_index": source_index,
        "path": path,
        "file_sha256": sha256_file(path),
        "original_sample_rate": int(info.samplerate),
        "duration_seconds": duration_seconds,
        "accepted_seconds": float(token.shape[1]) / TOKEN_RATE,
        "prompt_token": token.numpy().astype(np.int32),
        "prompt_feat": feat.numpy().astype(np.float32),
        "embedding": embedding.astype(np.float32),
        "embedding_norm": embedding_norm,
    }


def _resolve_raw_fusion_weights(
    accepted: list[dict[str, Any]],
    fusion_mode: str,
    manual_weights: list[Any],
) -> list[float]:
    weights = []
    for source in accepted:
        if fusion_mode == "equal_weight":
            weight = 1.0
        elif fusion_mode == "duration_weight":
            weight = float(source["accepted_seconds"])
        else:
            source_index = int(source["source_index"])
            if source_index >= len(manual_weights):
                raise ValueError("manual fusion requires one non-negative raw weight per reference")
            weight = float(manual_weights[source_index])
            if weight < 0:
                raise ValueError("manual fusion weights must be non-negative")
        weights.append(float(weight))
    return weights


def _fuse_speaker_embeddings(normalized_embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    fused = np.zeros_like(normalized_embeddings[0], dtype=np.float32)
    for embedding, weight in zip(normalized_embeddings, weights):
        if weight > 0:
            fused += embedding.astype(np.float32) * float(weight)
    return l2_normalize_array(fused).astype(np.float32)


def _voice_prompt_inputs_from_package_parts(
    *,
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    prompt_sources: list[dict[str, Any]],
    fused_embedding: np.ndarray,
    device: torch.device,
) -> VoicePromptInputs:
    branches = []
    source_by_branch = {int(source["branch_index"]): source for source in prompt_sources}
    branch_count = int(metadata["branch_count"])
    for branch_index in range(branch_count):
        source_metadata = source_by_branch[branch_index]
        branches.append(
            VoicePromptBranch(
                prompt_token=torch.from_numpy(tensors[f"branch_{branch_index}_prompt_token"]).to(
                    device=device,
                    dtype=torch.int32,
                ),
                prompt_feat=torch.from_numpy(tensors[f"branch_{branch_index}_prompt_feat"]).to(
                    device=device,
                    dtype=torch.float32,
                ),
                embedding=torch.from_numpy(tensors[f"branch_{branch_index}_speaker_embedding"]).to(
                    device=device,
                    dtype=torch.float32,
                ),
                weight_raw=float(source_metadata["fusion_weight_raw"]),
                weight_normalized=float(source_metadata["fusion_weight_normalized"]),
                metadata=dict(source_metadata),
            )
        )
    return VoicePromptInputs(
        branches=branches,
        fused_embedding=torch.from_numpy(fused_embedding).to(device=device, dtype=torch.float32),
        metadata=dict(metadata),
    )


MappingLike = Mapping[str, Any]


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
        status_fn("Preparing voice package..." if config.voice_package else "Preparing prompt...")
        context = prepare_stream_context(model, config.voice_package or config.prompt, config.settings)
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
                grouped_prompt_inputs=context.grouped_prompt_inputs,
                soft_prompt_inputs=context.soft_prompt_inputs,
                log_fn=log_fn,
                stop_event=stop_event,
                on_audio_chunk=lambda speech_chunk, row: metrics_fn(row, None),
            )
        finally:
            source_tokenizer.join()
        source_stats = source_tokenizer.stats()
        if stop_event is not None and stop_event.is_set():
            log_fn("Offline job stopped by user.")
        output_sample_rate = model.sample_rate
        postprocess_stats = None
        if config.settings.lavasr_enabled:
            status_fn("Enhancing with LavaSR...")
            lavasr = create_lavasr_extender(config.settings, log_fn=log_fn)
            if lavasr is not None:
                speech, postprocess_stats = lavasr.enhance(speech, model.sample_rate)
                output_sample_rate = lavasr.output_sample_rate
                log_fn(
                    "lavasr_offline "
                    f"input_samples={postprocess_stats['lavasr_input_samples']} "
                    f"output_samples={postprocess_stats['lavasr_output_samples']} "
                    f"seconds={postprocess_stats['lavasr_seconds']:.3f}"
                )
        write_wav(config.output, speech, output_sample_rate)
        if config.csv:
            write_rows(config.csv, rows)
        for line in offline_summary_lines(
            model,
            source_stats,
            rows,
            config.output,
            output_sample_rate=output_sample_rate,
            postprocess_stats=postprocess_stats,
        ):
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
            status_fn("Preparing voice package..." if config.voice_package else "Preparing prompt...")
            context = prepare_stream_context(model, config.voice_package or config.prompt, config.settings)
            for line in stream_context_log_lines(model, config.settings, context):
                log_fn(line)
            lavasr = None
            output_sample_rate = realtime_output_sample_rate(model.sample_rate)
            if config.settings.lavasr_enabled:
                status_fn("Loading LavaSR...")
                lavasr = create_lavasr_extender(config.settings, log_fn=log_fn)
                if lavasr is not None:
                    output_sample_rate = lavasr.output_sample_rate
            status_fn("Loading Silero VAD..." if config.settings.vad_enabled else "Opening audio output...")
            vad_gate = create_vad_gate(config.settings)
            player = RealtimeAudioPlayer(
                sample_rate=output_sample_rate,
                output_device=config.output_device,
                log_fn=log_fn,
            )
            log_fn(f"realtime_prebuffer_seconds={player.prebuffer_seconds:.3f}")
            log_fn(f"realtime_output_sample_rate={output_sample_rate}")
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

            def handle_chunk(speech_chunk: torch.Tensor, row: dict) -> dict:
                metric_row = row
                chunk_sample_rate = model.sample_rate
                if lavasr is not None:
                    speech_chunk, lavasr_stats = lavasr.enhance(speech_chunk, model.sample_rate)
                    chunk_sample_rate = lavasr.output_sample_rate
                    metric_row = add_lavasr_row_stats(row, lavasr_stats)
                if chunk_sample_rate != output_sample_rate:
                    resample_start = time.perf_counter()
                    speech_chunk = resample_realtime_output(speech_chunk, chunk_sample_rate, output_sample_rate)
                    metric_row = add_realtime_resample_row_stats(
                        metric_row,
                        seconds=time.perf_counter() - resample_start,
                        output_samples=int(speech_chunk.shape[-1]),
                        output_sample_rate=output_sample_rate,
                    )
                if player is not None:
                    player.write(speech_chunk)
                    metrics_fn(metric_row, player.stats())
                return metric_row

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
                grouped_prompt_inputs=context.grouped_prompt_inputs,
                soft_prompt_inputs=context.soft_prompt_inputs,
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

    def create_voice_package(
        self,
        config: BackendConfig,
        prompt_wavs: list[str | Path],
        output_path: str | Path,
        options: Mapping[str, Any] | None = None,
        *,
        status_fn: Callable[[str], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        status_fn = status_fn or _noop_status
        log_fn = log_fn or _noop_log
        status_fn("Loading model...")
        model = self.get_model(config, log_fn=log_fn)
        status_fn("Creating voice package...")
        return create_voice_package(
            model,
            prompt_wavs=prompt_wavs,
            output_path=output_path,
            options=options,
            status_fn=status_fn,
            log_fn=log_fn,
        )

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
