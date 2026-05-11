from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .environment import configure_environment

configure_environment()

import torch

from cosyvoice.vc.audio import mel_spectrogram_24000, read_mono
from cosyvoice.vc.device import empty_cache as empty_device_cache
from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat
from cosyvoice.vc.voice_package import (
    TOKEN_MEL_RATIO,
    TOKEN_RATE,
    VoicePromptInputs,
    load_voice_package,
)

from .device import sync_device


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
