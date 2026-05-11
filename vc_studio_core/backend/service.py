from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .environment import configure_environment

configure_environment()

import torch

from cosyvoice.vc.audio import write_wav
from cosyvoice.vc.model import VCOnlyModel

from .callbacks import _noop_log, _noop_metrics, _noop_status
from .context import prepare_stream_context, stream_context_log_lines
from .lavasr import (
    add_lavasr_row_stats,
    add_realtime_resample_row_stats,
    create_lavasr_extender,
    realtime_output_sample_rate,
    resample_realtime_output,
)
from .player import RealtimeAudioPlayer
from .streaming import run_window_stream, write_rows
from .summary import offline_summary_lines
from .tokenizers import AsyncSourceTokenizer, MicrophoneSourceTokenizer
from .types import BackendConfig
from .vad import create_vad_gate
from .voice_package_builder import create_voice_package


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
