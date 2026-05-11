from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import random
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from cosyvoice.vc.voice_package import (
    DEFAULT_SOFT_PROMPT_CHECKPOINT_SEGMENTS,
    DEFAULT_SOFT_PROMPT_DISTILL_LAYER,
    DEFAULT_SOFT_PROMPT_SECONDS,
    SOFT_PROMPT_ALGORITHM,
    SOFT_PROMPT_VERSION,
    TOKEN_MEL_RATIO,
    TOKEN_RATE,
    VoicePromptInputs,
)


@dataclass(frozen=True)
class SoftPromptTrainingConfig:
    enabled: bool = False
    seconds: float = DEFAULT_SOFT_PROMPT_SECONDS
    steps: int = 300
    teacher_mode: str = "grouped_branch_attention"
    distill_layer: int = DEFAULT_SOFT_PROMPT_DISTILL_LAYER
    activation_checkpointing: str = "auto"
    checkpoint_segments: int = DEFAULT_SOFT_PROMPT_CHECKPOINT_SEGMENTS
    source_window_min_mel_frames: int = 50
    source_window_max_mel_frames: int = 100
    learning_rate: float = 1e-2
    hidden_mse_weight: float = 1.0
    prompt_delta_l2_weight: float = 1e-4
    prompt_smoothness_weight: float = 1e-4
    gradient_clip_norm: float = 1.0
    log_every: int = 25
    validation_windows: int = 2
    validation_every: int = 50
    low_memory_free_gb: float = 14.0
    teacher_branch_subset_size: int = 0


@dataclass(frozen=True)
class SoftPromptTrainingResult:
    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]
    final_loss: float
    training_steps: int
    checkpoint_segments_used: int


def distill_soft_prompt_v1(
    model: Any,
    prompt_inputs: VoicePromptInputs,
    config: SoftPromptTrainingConfig,
    status_fn: Callable[[str], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> SoftPromptTrainingResult:
    status_fn = status_fn or _noop
    log_fn = log_fn or _noop
    if not config.enabled:
        raise ValueError("soft prompt distillation was requested with enabled=False")
    if config.seconds <= 0:
        raise ValueError("soft_prompt_seconds must be greater than 0")
    if config.steps < 0:
        raise ValueError("soft_prompt_steps must be 0 or greater")
    if config.teacher_mode not in {"grouped_branch_attention", "init_only"}:
        raise ValueError("soft_prompt_teacher_mode must be grouped_branch_attention or init_only")
    if config.activation_checkpointing not in {"off", "auto", "on"}:
        raise ValueError("soft_prompt_activation_checkpointing must be off, auto, or on")
    if config.checkpoint_segments <= 0:
        raise ValueError("soft_prompt_checkpoint_segments must be greater than 0")
    if not isinstance(model.flow.decoder.estimator, torch.nn.Module):
        raise RuntimeError("soft prompt distillation requires the PyTorch DiT estimator")

    device = model.device
    target_frames = align_soft_prompt_mel_frames(model, config.seconds)
    active_indices = prompt_inputs.active_branch_indices()
    if not active_indices:
        raise ValueError("voice package has no active prompt branches")
    dominant_index = prompt_inputs.dominant_branch_index()

    status_fn("Initializing soft prompt...")
    with torch.no_grad():
        branch_weights = torch.tensor(
            [prompt_inputs.sharpened_weights()[index] for index in active_indices],
            dtype=torch.float32,
            device=device,
        )
        branch_weights = branch_weights / branch_weights.sum()
        initial_mu, initial_feat = multi_reference_soft_prompt_initialization(
            model,
            prompt_inputs,
            active_indices=active_indices,
            branch_weights=branch_weights,
            target_frames=target_frames,
        )
        soft_speaker_embedding = F.normalize(prompt_inputs.fused_embedding.to(device=device, dtype=torch.float32), dim=1)
        branch_records = prepare_branch_teacher_records(
            model,
            prompt_inputs,
            active_indices=active_indices,
            target_device=device,
            target_frames=target_frames,
        )
        for branch_index in active_indices:
            source_frames = int(prompt_inputs.branches[branch_index].prompt_feat.shape[1])
            log_fn(
                "soft_prompt_reference_fit branch={branch} original_mel_frames={source} "
                "target_mel_frames={target} policy={policy}".format(
                    branch=branch_index,
                    source=source_frames,
                    target=target_frames,
                    policy=frame_fit_policy(source_frames, target_frames),
                )
            )
        log_fn(
            "soft_prompt_init=weighted_resampled_references "
            "soft_prompt_teacher_branch_count={count} soft_prompt_teacher_mel_frames={frames}".format(
                count=len(branch_records),
                frames=target_frames,
            )
        )

    checkpoint_segments = resolve_checkpoint_segments(
        model,
        mode=config.activation_checkpointing,
        segments=config.checkpoint_segments,
        low_memory_free_gb=config.low_memory_free_gb,
    )
    if checkpoint_segments > 0:
        log_fn(f"soft_prompt_activation_checkpointing=on segments={checkpoint_segments}")
    else:
        log_fn("soft_prompt_activation_checkpointing=off")

    final_loss = 0.0
    trained_delta = None
    actual_steps = 0
    if config.steps > 0 and config.teacher_mode != "init_only":
        status_fn("Training soft prompt...")
        final_loss, trained_delta = train_soft_prompt_delta(
            model,
            prompt_inputs,
            branch_records,
            initial_mu=initial_mu,
            initial_feat=initial_feat,
            soft_speaker_embedding=soft_speaker_embedding,
            active_indices=active_indices,
            dominant_index=dominant_index,
            config=config,
            checkpoint_segments=checkpoint_segments,
            log_fn=log_fn,
        )
        actual_steps = int(config.steps)
    else:
        final_loss = 0.0
        log_fn("soft_prompt_training_skipped=True")

    soft_prompt_mu = initial_mu.detach()
    if isinstance(trained_delta, torch.Tensor):
        soft_prompt_mu = soft_prompt_mu + trained_delta.to(device=soft_prompt_mu.device, dtype=soft_prompt_mu.dtype)
    soft_prompt_mu = soft_prompt_mu.detach().cpu().numpy().astype(np.float32)
    soft_prompt_feat = initial_feat.detach().cpu().numpy().astype(np.float32)
    soft_speaker = soft_speaker_embedding.detach().cpu().numpy().astype(np.float32)

    metadata = {
        "prompt_fusion_algorithm": SOFT_PROMPT_ALGORITHM,
        "soft_prompt_version": SOFT_PROMPT_VERSION,
        "soft_prompt_mel_frames": int(target_frames),
        "soft_prompt_seconds": float(target_frames / (TOKEN_RATE * TOKEN_MEL_RATIO)),
        "soft_prompt_init": "weighted_resampled_references",
        "soft_prompt_training_steps": int(actual_steps),
        "soft_prompt_training_loss": "layer_hidden_distill_v1",
        "soft_prompt_teacher": "grouped_branch_attention_hidden_state",
        "soft_prompt_distill_layer": int(config.distill_layer),
        "soft_speaker_embedding_init": "weighted_fused_embedding",
        "soft_speaker_embedding_trainable": False,
        "soft_prompt_activation_checkpointing": config.activation_checkpointing,
        "soft_prompt_checkpoint_segments": int(config.checkpoint_segments),
        "soft_prompt_checkpoint_segments_used": int(checkpoint_segments),
        "soft_prompt_final_loss": float(final_loss),
        "soft_prompt_reference_fit_policy": "linear_resample_each_reference_to_target_mel_frames",
        "soft_prompt_source_window_policy": "weighted_random_reference_token_windows",
    }
    return SoftPromptTrainingResult(
        tensors={
            "soft_prompt_mu": soft_prompt_mu,
            "soft_prompt_feat": soft_prompt_feat,
            "soft_speaker_embedding": soft_speaker,
        },
        metadata=metadata,
        final_loss=float(final_loss),
        training_steps=int(actual_steps),
        checkpoint_segments_used=int(checkpoint_segments),
    )


def train_soft_prompt_delta(
    model: Any,
    prompt_inputs: VoicePromptInputs,
    branch_records: list[dict[str, Any]],
    *,
    initial_mu: torch.Tensor,
    initial_feat: torch.Tensor,
    soft_speaker_embedding: torch.Tensor,
    active_indices: list[int],
    dominant_index: int,
    config: SoftPromptTrainingConfig,
    checkpoint_segments: int,
    log_fn: Callable[[str], None],
) -> tuple[float, torch.Tensor]:
    device = model.device
    estimator = model.flow.decoder.estimator
    ratio = int(getattr(model, "token_mel_ratio", TOKEN_MEL_RATIO))
    branch_weights = torch.tensor(
        [prompt_inputs.sharpened_weights()[index] for index in active_indices],
        dtype=torch.float32,
        device=device,
    )
    branch_weights = branch_weights / branch_weights.sum()
    soft_spk = model.flow.spk_embed_affine_layer(soft_speaker_embedding).detach()
    mu_delta = torch.nn.Parameter(torch.zeros_like(initial_mu, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([mu_delta], lr=float(config.learning_rate))
    source_min_tokens = max(1, int(config.source_window_min_mel_frames) // ratio)
    source_max_tokens = max(source_min_tokens, int(config.source_window_max_mel_frames) // ratio)
    source_max_tokens = max(1, source_max_tokens)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16
    last_loss = 0.0
    ema_loss = None
    last_validation_loss = None
    best_validation_loss = None
    best_validation_step = 0
    best_delta = None
    validation_windows = build_validation_windows(
        prompt_inputs,
        active_indices,
        source_min_tokens,
        source_max_tokens,
        count=config.validation_windows,
    )

    with freeze_module_parameters(model.flow), torch.enable_grad():
        estimator.eval()
        for step in range(1, int(config.steps) + 1):
            window = sample_source_window(prompt_inputs, active_indices, source_min_tokens, source_max_tokens)
            source_token = window["token"].to(device=device, dtype=torch.int64)
            source_mel_offset = int(window["token_start"]) * ratio
            with torch.no_grad():
                source_mu = source_token_mu(model, source_token)
                source_len = source_mu.shape[2]
                source_cond = torch.zeros_like(source_mu)
                source_x = model.flow.decoder.noise_slice(
                    source_mel_offset,
                    source_len,
                    device=device,
                    dtype=torch.float32,
                )
                t = torch.rand((), device=device, dtype=torch.float32).clamp(0.02, 0.98)
                teacher_records = choose_teacher_records(
                    branch_records,
                    branch_weights,
                    config.teacher_branch_subset_size,
                    dominant_index=dominant_index,
                )
                teacher_branch_weights = teacher_records["weights"]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    teacher = estimator.forward_grouped_prompt_source_until_layer(
                        branch_prompt_x=[
                            model.flow.decoder.noise_slice(0, record["mu"].shape[2], device=device, dtype=torch.float32)
                            for record in teacher_records["records"]
                        ],
                        branch_prompt_mu=[record["mu"].to(device=device) for record in teacher_records["records"]],
                        branch_prompt_cond=[record["cond"].to(device=device) for record in teacher_records["records"]],
                        branch_spks=[record["spk"].to(device=device) for record in teacher_records["records"]],
                        source_x=source_x,
                        source_mu=source_mu,
                        t=t,
                        source_spks=soft_spk,
                        source_cond=source_cond,
                        branch_weights=teacher_branch_weights.to(device=device),
                        distill_layer=config.distill_layer,
                        dominant_branch_position=teacher_records["dominant_position"],
                        streaming=True,
                        source_mel_offset=source_mel_offset,
                        attention_temperature=float(prompt_inputs.metadata.get("attention_temperature", 1.0)),
                    )[:1].detach()
            empty_device_cache(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                soft_mu = (initial_mu + mu_delta).transpose(1, 2).contiguous()
                soft_cond = initial_feat.transpose(1, 2).contiguous()
                prompt_x = model.flow.decoder.noise_slice(
                    0,
                    initial_mu.shape[1],
                    device=device,
                    dtype=source_x.dtype,
                )
                student = estimator.forward_soft_prompt_source_until_layer(
                    prompt_x=prompt_x,
                    prompt_mu=soft_mu,
                    prompt_cond=soft_cond,
                    prompt_spks=soft_spk,
                    source_x=source_x,
                    source_mu=source_mu,
                    t=t,
                    source_spks=soft_spk,
                    source_cond=source_cond,
                    distill_layer=config.distill_layer,
                    streaming=True,
                    source_mel_offset=source_mel_offset,
                    checkpoint_segments=checkpoint_segments,
                )[:1]
                hidden_loss = F.mse_loss(student.float(), teacher.to(device=student.device, dtype=student.dtype).float())
                delta_l2 = mu_delta.float().pow(2).mean()
                smoothness = (mu_delta[:, 1:] - mu_delta[:, :-1]).float().pow(2).mean()
                loss = (
                    float(config.hidden_mse_weight) * hidden_loss
                    + float(config.prompt_delta_l2_weight) * delta_l2
                    + float(config.prompt_smoothness_weight) * smoothness
                )
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([mu_delta], float(config.gradient_clip_norm))
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            ema_loss = last_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * last_loss
            if step == 1 or step == config.steps or step % max(1, int(config.log_every)) == 0:
                validation_loss = None
                if validation_windows and (
                    step == 1
                    or step == config.steps
                    or step % max(1, int(config.validation_every)) == 0
                ):
                    validation_loss = evaluate_soft_prompt_validation_loss(
                        model,
                        estimator,
                        branch_records,
                        branch_weights,
                        initial_mu=initial_mu,
                        initial_feat=initial_feat,
                        mu_delta=mu_delta,
                        soft_spk=soft_spk,
                        validation_windows=validation_windows,
                        prompt_inputs=prompt_inputs,
                        config=config,
                        checkpoint_segments=checkpoint_segments,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                    )
                    last_validation_loss = validation_loss
                    if best_validation_loss is None or validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_validation_step = step
                        best_delta = mu_delta.detach().clone()
                    empty_device_cache(device)
                validation_text = (
                    ""
                    if validation_loss is None
                    else (
                        f" validation_hidden_mse={validation_loss:.6f}"
                        f" best_validation_hidden_mse={best_validation_loss:.6f}"
                        f" best_step={best_validation_step}"
                    )
                )
                log_fn(
                    "soft_prompt_step={step}/{total} loss={loss:.6f} hidden_mse={hidden:.6f} "
                    "ema_loss={ema:.6f} delta_l2={delta:.6f} smoothness={smooth:.6f} "
                    "source_mel={source_mel}{validation}".format(
                        step=step,
                        total=config.steps,
                        loss=last_loss,
                        hidden=float(hidden_loss.detach().cpu().item()),
                        ema=float(ema_loss),
                        delta=float(delta_l2.detach().cpu().item()),
                        smooth=float(smoothness.detach().cpu().item()),
                        source_mel=source_len,
                        validation=validation_text,
                    )
                )
            del teacher, student, loss, hidden_loss, delta_l2, smoothness
            empty_device_cache(device)

    if best_delta is not None and best_validation_loss is not None:
        log_fn(
            "soft_prompt_selected_step={step} best_validation_hidden_mse={loss:.6f}".format(
                step=best_validation_step,
                loss=best_validation_loss,
            )
        )
        return best_validation_loss, best_delta
    return (last_validation_loss if last_validation_loss is not None else last_loss), mu_delta.detach()


def prepare_branch_teacher_records(
    model: Any,
    prompt_inputs: VoicePromptInputs,
    *,
    active_indices: list[int],
    target_device: torch.device,
    target_frames: int,
) -> list[dict[str, Any]]:
    records = []
    for branch_index in active_indices:
        branch = prompt_inputs.branches[branch_index]
        token = branch.prompt_token.to(device=target_device)
        feat = branch.prompt_feat.to(device=target_device, dtype=torch.float32)
        feat = fit_frames(feat, target_frames)
        mu = branch_prompt_mu(model, token, target_frames=target_frames).transpose(1, 2).contiguous()
        embedding = F.normalize(branch.embedding.to(device=target_device, dtype=torch.float32), dim=1)
        spk = model.flow.spk_embed_affine_layer(embedding)
        records.append(
            {
                "branch_index": branch_index,
                "mu": mu.detach(),
                "cond": feat.transpose(1, 2).contiguous().detach(),
                "spk": spk.detach(),
            }
        )
    return records


def multi_reference_soft_prompt_initialization(
    model: Any,
    prompt_inputs: VoicePromptInputs,
    *,
    active_indices: list[int],
    branch_weights: torch.Tensor,
    target_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_accum = None
    feat_accum = None
    for position, branch_index in enumerate(active_indices):
        branch = prompt_inputs.branches[branch_index]
        weight = branch_weights[position].to(device=model.device, dtype=torch.float32)
        branch_mu = branch_prompt_mu(
            model,
            branch.prompt_token.to(model.device),
            target_frames=target_frames,
        )
        branch_feat = fit_frames(
            branch.prompt_feat.to(device=model.device, dtype=torch.float32),
            target_frames,
        )
        mu_accum = branch_mu * weight if mu_accum is None else mu_accum + branch_mu * weight
        feat_accum = branch_feat * weight if feat_accum is None else feat_accum + branch_feat * weight
    if mu_accum is None or feat_accum is None:
        raise ValueError("cannot initialize soft prompt without active references")
    return mu_accum.to(dtype=torch.float32), feat_accum.to(dtype=torch.float32)


def branch_prompt_mu(model: Any, prompt_token: torch.Tensor, target_frames: int) -> torch.Tensor:
    prompt_token = prompt_token.to(device=model.device, dtype=torch.int64)
    token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=model.device)
    mask = torch.ones((1, prompt_token.shape[1], 1), dtype=torch.float32, device=model.device)
    if token_len.numel() > 0:
        token_embed = model.flow.input_embedding(torch.clamp(prompt_token, min=0)) * mask
    else:
        token_embed = torch.zeros(1, 0, model.flow.input_size, device=model.device)
    h = model.flow.pre_lookahead_layer(token_embed)
    h = h.repeat_interleave(model.token_mel_ratio, dim=1)
    return fit_frames(h.to(dtype=torch.float32), target_frames)


def source_token_mu(model: Any, source_token: torch.Tensor) -> torch.Tensor:
    source_token = source_token.to(device=model.device, dtype=torch.int64)
    token_embed = model.flow.input_embedding(torch.clamp(source_token, min=0))
    h = model.flow.pre_lookahead_layer(token_embed)
    h = h.repeat_interleave(model.token_mel_ratio, dim=1)
    return h.transpose(1, 2).contiguous().to(dtype=torch.float32)


def fit_frames(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected a [1, T, C] tensor, got {tuple(tensor.shape)}")
    target_frames = int(target_frames)
    if tensor.shape[1] == target_frames:
        return tensor.to(dtype=torch.float32)
    if tensor.shape[1] <= 0:
        return torch.zeros(1, target_frames, tensor.shape[2], device=tensor.device, dtype=torch.float32)
    resized = F.interpolate(
        tensor.transpose(1, 2).to(dtype=torch.float32),
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2).contiguous()


def frame_fit_policy(source_frames: int, target_frames: int) -> str:
    source_frames = int(source_frames)
    target_frames = int(target_frames)
    if source_frames == target_frames:
        return "exact"
    if source_frames <= 0:
        return "zero_pad_empty"
    if source_frames > target_frames:
        return "linear_resample_compress"
    return "linear_resample_stretch"


def align_soft_prompt_mel_frames(model: Any, seconds: float) -> int:
    raw_frames = max(1, int(round(float(seconds) * TOKEN_RATE * TOKEN_MEL_RATIO)))
    static_chunk = int(getattr(model.flow.decoder.estimator, "static_chunk_size", 0))
    if static_chunk <= 0:
        return raw_frames
    return max(static_chunk, int(math.ceil(raw_frames / static_chunk) * static_chunk))


def sample_source_window(
    prompt_inputs: VoicePromptInputs,
    active_indices: list[int],
    min_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    weights = prompt_inputs.sharpened_weights()
    active_weights = [weights[index] for index in active_indices]
    branch_index = random.choices(active_indices, weights=active_weights, k=1)[0]
    branch = prompt_inputs.branches[branch_index]
    token = branch.prompt_token
    if token.shape[1] <= 0:
        raise ValueError("cannot sample a source window from an empty prompt token branch")
    window_tokens = random.randint(min_tokens, max_tokens)
    window_tokens = min(max(1, window_tokens), token.shape[1])
    start = 0 if token.shape[1] <= window_tokens else random.randint(0, token.shape[1] - window_tokens)
    return {
        "branch_index": branch_index,
        "token_start": start,
        "token": token[:, start:start + window_tokens],
    }


def build_validation_windows(
    prompt_inputs: VoicePromptInputs,
    active_indices: list[int],
    min_tokens: int,
    max_tokens: int,
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(0)
    windows = []
    weights = prompt_inputs.sharpened_weights()
    active_weights = [weights[index] for index in active_indices]
    for _ in range(int(count)):
        branch_index = rng.choices(active_indices, weights=active_weights, k=1)[0]
        branch = prompt_inputs.branches[branch_index]
        token = branch.prompt_token
        if token.shape[1] <= 0:
            continue
        window_tokens = rng.randint(min_tokens, max_tokens)
        window_tokens = min(max(1, window_tokens), token.shape[1])
        start = 0 if token.shape[1] <= window_tokens else rng.randint(0, token.shape[1] - window_tokens)
        windows.append(
            {
                "branch_index": branch_index,
                "token_start": start,
                "token": token[:, start:start + window_tokens].clone(),
                "t": 0.15 + 0.7 * (len(windows) + 1) / max(1, int(count) + 1),
            }
        )
    return windows


@torch.no_grad()
def evaluate_soft_prompt_validation_loss(
    model: Any,
    estimator: torch.nn.Module,
    branch_records: list[dict[str, Any]],
    branch_weights: torch.Tensor,
    *,
    initial_mu: torch.Tensor,
    initial_feat: torch.Tensor,
    mu_delta: torch.Tensor,
    soft_spk: torch.Tensor,
    validation_windows: list[dict[str, Any]],
    prompt_inputs: VoicePromptInputs,
    config: SoftPromptTrainingConfig,
    checkpoint_segments: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    if not validation_windows:
        return 0.0
    device = model.device
    ratio = int(getattr(model, "token_mel_ratio", TOKEN_MEL_RATIO))
    dominant_index = prompt_inputs.dominant_branch_index()
    dominant_position = next(
        (index for index, record in enumerate(branch_records) if record.get("branch_index") == dominant_index),
        0,
    )
    losses = []
    for window in validation_windows:
        source_token = window["token"].to(device=device, dtype=torch.int64)
        source_mel_offset = int(window["token_start"]) * ratio
        source_mu = source_token_mu(model, source_token)
        source_len = source_mu.shape[2]
        source_cond = torch.zeros_like(source_mu)
        source_x = model.flow.decoder.noise_slice(
            source_mel_offset,
            source_len,
            device=device,
            dtype=torch.float32,
        )
        t = torch.tensor(float(window["t"]), device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            teacher = estimator.forward_grouped_prompt_source_until_layer(
                branch_prompt_x=[
                    model.flow.decoder.noise_slice(0, record["mu"].shape[2], device=device, dtype=torch.float32)
                    for record in branch_records
                ],
                branch_prompt_mu=[record["mu"].to(device=device) for record in branch_records],
                branch_prompt_cond=[record["cond"].to(device=device) for record in branch_records],
                branch_spks=[record["spk"].to(device=device) for record in branch_records],
                source_x=source_x,
                source_mu=source_mu,
                t=t,
                source_spks=soft_spk,
                source_cond=source_cond,
                branch_weights=branch_weights.to(device=device),
                distill_layer=config.distill_layer,
                dominant_branch_position=dominant_position,
                streaming=True,
                source_mel_offset=source_mel_offset,
                attention_temperature=float(prompt_inputs.metadata.get("attention_temperature", 1.0)),
            )[:1]
            student = estimator.forward_soft_prompt_source_until_layer(
                prompt_x=model.flow.decoder.noise_slice(
                    0,
                    initial_mu.shape[1],
                    device=device,
                    dtype=source_x.dtype,
                ),
                prompt_mu=(initial_mu + mu_delta).transpose(1, 2).contiguous(),
                prompt_cond=initial_feat.transpose(1, 2).contiguous(),
                prompt_spks=soft_spk,
                source_x=source_x,
                source_mu=source_mu,
                t=t,
                source_spks=soft_spk,
                source_cond=source_cond,
                distill_layer=config.distill_layer,
                streaming=True,
                source_mel_offset=source_mel_offset,
                checkpoint_segments=0 if checkpoint_segments <= 0 else checkpoint_segments,
            )[:1]
        losses.append(F.mse_loss(student.float(), teacher.float()).detach())
    return float(torch.stack(losses).mean().cpu().item())


def choose_teacher_records(
    branch_records: list[dict[str, Any]],
    branch_weights: torch.Tensor,
    subset_size: int,
    dominant_index: int,
) -> dict[str, Any]:
    if subset_size <= 0 or subset_size >= len(branch_records):
        dominant_position = next(
            (index for index, record in enumerate(branch_records) if record.get("branch_index") == dominant_index),
            0,
        )
        return {"records": branch_records, "weights": branch_weights, "dominant_position": dominant_position}
    weights_cpu = branch_weights.detach().cpu().float().numpy()
    indices = np.random.choice(
        len(branch_records),
        size=max(1, int(subset_size)),
        replace=False,
        p=weights_cpu / weights_cpu.sum(),
    )
    indices = sorted(int(index) for index in indices)
    selected_weights = branch_weights[indices]
    selected_weights = selected_weights / selected_weights.sum()
    dominant_position = next(
        (
            position
            for position, record_index in enumerate(indices)
            if branch_records[record_index].get("branch_index") == dominant_index
        ),
        0,
    )
    return {
        "records": [branch_records[index] for index in indices],
        "weights": selected_weights,
        "dominant_position": dominant_position,
    }


def resolve_checkpoint_segments(
    model: Any,
    *,
    mode: str,
    segments: int,
    low_memory_free_gb: float,
) -> int:
    if mode == "off":
        return 0
    if mode == "on":
        return max(1, int(segments))
    device = model.device
    if device.type != "cuda":
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return max(1, int(segments))
    free_gb = free_bytes / (1024 ** 3)
    return max(1, int(segments)) if free_gb < float(low_memory_free_gb) else 0


@contextmanager
def freeze_module_parameters(module: torch.nn.Module):
    parameters = list(module.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    training = module.training
    try:
        module.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, value in zip(parameters, requires_grad):
            parameter.requires_grad_(value)
        module.train(training)


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _noop(message: str) -> None:
    pass
