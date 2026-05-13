from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .environment import configure_environment

configure_environment()

import torch

from cosyvoice.vc.model import VCOnlyModel

from .device import sync_device


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
        source_cache_len = history_target_mel if use_history_kv_cache and active_cache_steps is not None else 0
        source_cache_end = history_mel + current_mel
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
