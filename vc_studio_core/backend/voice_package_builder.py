from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .environment import configure_environment

configure_environment()

import numpy as np
import soundfile as sf
import torch

from cosyvoice.vc.audio import mel_spectrogram_24000, read_mono
from cosyvoice.vc.device import empty_cache as empty_device_cache
from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat
from cosyvoice.vc.soft_prompt import SoftPromptTrainingConfig, distill_soft_prompt_v1
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
    VoicePromptBranch,
    VoicePromptInputs,
    l2_normalize_array,
    model_compatibility_fields,
    new_package_id,
    save_voice_package,
    sha256_file,
    sharpen_weights,
    utc_now_iso,
)

from .callbacks import _noop_log, _noop_status
from .device import sync_device


MappingLike = Mapping[str, Any]


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
            learning_rate=float(options_dict.get("soft_prompt_learning_rate", 5e-4)),
            hidden_mse_weight=float(options_dict.get("soft_prompt_hidden_mse_weight", 1.0)),
            prompt_delta_l2_weight=float(options_dict.get("soft_prompt_delta_l2_weight", 1e-4)),
            prompt_smoothness_weight=float(options_dict.get("soft_prompt_smoothness_weight", 1e-4)),
            gradient_clip_norm=float(options_dict.get("soft_prompt_gradient_clip_norm", 1.0)),
            validation_windows=int(options_dict.get("soft_prompt_validation_windows", 8)),
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
