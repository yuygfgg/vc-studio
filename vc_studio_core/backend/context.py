from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .environment import configure_environment

configure_environment()

import torch

from cosyvoice.vc.model import VCOnlyModel
from cosyvoice.vc.voice_package import TOKEN_MEL_RATIO, TOKEN_RATE, VoicePromptInputs

from .device import sync_device
from .prompt_cache import (
    grouped_prompt_cache_enabled,
    grouped_prompt_enabled,
    normalize_prompt_runtime_policy,
    prepare_grouped_prompt_cache,
    prepare_grouped_prompt_runtime_inputs,
    prepare_prompt_inputs_from_package,
    prepare_prompt_inputs_from_wav,
    prompt_cache_dtype_bytes,
    prompt_cache_offload_kv_to_cpu,
    prompt_cache_storage_dtype,
    choose_full_prompt_cache_frames,
    optimize_prompt_cache_storage,
    select_runtime_prompt_inputs,
    select_soft_prompt_runtime_inputs,
    trim_cache_mel_frames_to_static,
)
from .timing import (
    align_audio_blend_samples,
    align_audio_declick_samples,
    align_delayed_commit_tokens,
    align_history_tokens,
    align_overlap_tokens,
    is_static_cache_aligned,
)
from .types import PreparedStreamContext, StreamSettings


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
