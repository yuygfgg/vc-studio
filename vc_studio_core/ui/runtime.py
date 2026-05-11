from __future__ import annotations

import importlib.util
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

from PyQt6 import QtGui, QtWidgets

from vc_studio_core.backend import BackendConfig, StreamSettings

from .state import TextValue


class RuntimeMixin:
    def _start_offline(self) -> None:
        if self._is_running():
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return
        try:
            config = self._snapshot_config(require_source=True)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Invalid settings", str(error))
            return
        self.offline_stop_event = threading.Event()
        self._set_running("offline", True)
        self.worker_thread = threading.Thread(target=self._offline_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop_offline(self) -> None:
        if self.offline_stop_event is not None:
            self.offline_stop_event.set()
        self._post("status", "Stopping offline job after the current chunk...")

    def _start_realtime(self) -> None:
        if self._is_running():
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return
        if importlib.util.find_spec("sounddevice") is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing dependency",
                "Realtime audio requires sounddevice. Install requirements.txt, then restart the GUI.",
            )
            return
        try:
            config = self._snapshot_config(require_source=False)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Invalid settings", str(error))
            return
        config = replace(
            config,
            input_device=self.input_device_map.get(self.input_device_var.get()),
            output_device=self.output_device_map.get(self.output_device_var.get()),
        )
        self.live_stop_event = threading.Event()
        self._set_running("realtime", True)
        self.worker_thread = threading.Thread(target=self._realtime_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop_realtime(self) -> None:
        if self.live_stop_event is not None:
            self.live_stop_event.set()
        self.backend.stop_realtime()
        self._post("status", "Stopping live stream...")

    def _snapshot_config(self, require_source: bool) -> BackendConfig:
        model_dir = self.model_dir_var.get().strip()
        voice_package = self.voice_package_var.get().strip()
        prompt = self.prompt_var.get().strip()
        if not model_dir:
            raise ValueError("Model directory is required.")
        if not Path(model_dir).expanduser().is_dir():
            raise ValueError("Model directory does not exist.")
        if not voice_package and not prompt:
            raise ValueError("Voice package is required. Use Legacy prompt WAV only from advanced settings.")
        if voice_package and not Path(voice_package).expanduser().is_file():
            raise ValueError("Voice package does not exist.")
        if prompt and not Path(prompt).expanduser().is_file():
            raise ValueError("Legacy prompt WAV does not exist.")
        if require_source and not self.source_var.get().strip():
            raise ValueError("Source wav is required for offline benchmark.")
        if require_source and not Path(self.source_var.get().strip()).expanduser().is_file():
            raise ValueError("Source wav does not exist.")
        settings = self._settings_from_form()
        source = self.source_var.get().strip()
        output = self.output_var.get().strip() or "out/vc_streaming.wav"
        csv_path = self.csv_var.get().strip()
        coreml_cache = self.coreml_cache_var.get().strip()
        return BackendConfig(
            model_dir=str(Path(model_dir).expanduser()),
            prompt=str(Path(prompt).expanduser()) if prompt else "",
            voice_package=str(Path(voice_package).expanduser()) if voice_package else "",
            source=str(Path(source).expanduser()) if source else "",
            output=str(Path(output).expanduser()),
            csv=str(Path(csv_path).expanduser()) if csv_path else "",
            device=self.device_var.get(),
            ort_provider=self.ort_provider_var.get(),
            coreml_cache_dir=str(Path(coreml_cache).expanduser()) if coreml_cache else None,
            settings=settings,
        )

    def _snapshot_model_config(self) -> BackendConfig:
        model_dir = self.model_dir_var.get().strip()
        if not model_dir:
            raise ValueError("Model directory is required.")
        if not Path(model_dir).expanduser().is_dir():
            raise ValueError("Model directory does not exist.")
        coreml_cache = self.coreml_cache_var.get().strip()
        return BackendConfig(
            model_dir=str(Path(model_dir).expanduser()),
            prompt="",
            voice_package="",
            source="",
            output="",
            csv="",
            device=self.device_var.get(),
            ort_provider=self.ort_provider_var.get(),
            coreml_cache_dir=str(Path(coreml_cache).expanduser()) if coreml_cache else None,
            settings=self._settings_from_form(),
        )

    def _settings_from_form(self) -> StreamSettings:
        chunk_sec = self._positive_float(self.chunk_sec_var, "Chunk sec")
        tokenizer_chunk_sec = self._nonnegative_float(self.tokenizer_chunk_sec_var, "Tokenizer chunk")
        effective_tokenizer_chunk_sec = tokenizer_chunk_sec if tokenizer_chunk_sec > 0 else chunk_sec
        tokenizer_left_context_sec = self._nonnegative_float(self.tokenizer_left_context_sec_var, "Tokenizer left context")
        tokenizer_right_context_sec = self._nonnegative_float(self.tokenizer_right_context_sec_var, "Tokenizer right context")
        if effective_tokenizer_chunk_sec + tokenizer_left_context_sec + tokenizer_right_context_sec > 30:
            raise ValueError("Tokenizer chunk plus left/right context must be 30 seconds or less.")
        vad_enabled = self.vad_enabled_var.get()
        vad_threshold = self._float_in_range(self.vad_threshold_var, "VAD threshold", 0.0, 1.0)
        vad_min_speech_ms = self._nonnegative_float(self.vad_min_speech_ms_var, "VAD min speech ms")
        vad_min_silence_ms = self._nonnegative_float(self.vad_min_silence_ms_var, "VAD min silence ms")
        vad_speech_pad_ms = self._nonnegative_float(self.vad_speech_pad_ms_var, "VAD speech pad ms")
        lavasr_enabled = self.lavasr_enabled_var.get()
        lavasr_lowpass_hz = self._float_in_range(self.lavasr_lowpass_hz_var, "LavaSR lowpass Hz", 1.0, 8000.0)
        if lavasr_enabled:
            if importlib.util.find_spec("LavaSR") is None:
                raise ValueError(
                    "LavaSR BWE is enabled, but the optional LavaSR package is not installed. "
                    "Install requirements.txt, then restart the GUI."
                )
        if vad_enabled:
            if importlib.util.find_spec("silero_vad") is None:
                raise ValueError(
                    "Silero VAD is enabled, but the optional silero-vad package is not installed. "
                    "Install requirements.txt or run `pip install silero-vad`."
                )
        prompt_cache_max_mb = self._nonnegative_float(self.prompt_cache_max_mb_var, "Prompt cache MiB")
        prompt_cache_max_seconds = self._nonnegative_float(self.prompt_cache_max_seconds_var, "Prompt cache sec")
        prompt_cache_dtype = self.prompt_cache_dtype_var.get()
        if prompt_cache_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Prompt cache dtype must be auto, float32, float16, or bfloat16.")
        prompt_cache_storage = self.prompt_cache_storage_var.get()
        if prompt_cache_storage not in {"device", "cpu_offload"}:
            raise ValueError("Prompt cache storage must be device or cpu_offload.")
        prompt_runtime_policy = self.prompt_runtime_policy_var.get()
        if prompt_runtime_policy not in {"auto", "soft", "grouped", "dominant"}:
            raise ValueError("Prompt mode must be auto, soft, grouped, or dominant.")
        return StreamSettings(
            chunk_sec=chunk_sec,
            tokenizer_chunk_sec=tokenizer_chunk_sec if tokenizer_chunk_sec > 0 else None,
            tokenizer_left_context_sec=tokenizer_left_context_sec,
            tokenizer_right_context_sec=tokenizer_right_context_sec,
            history_sec=self._nonnegative_float(self.history_sec_var, "History sec"),
            mel_overlap_sec=self._nonnegative_float(self.mel_overlap_sec_var, "Mel overlap sec"),
            delayed_commit_sec=self._nonnegative_float(self.delayed_commit_sec_var, "Delayed commit sec"),
            audio_declick_ms=self._nonnegative_float(self.audio_declick_ms_var, "De-click ms"),
            audio_blend_ms=self._nonnegative_float(self.audio_blend_ms_var, "Audio blend ms"),
            vad_enabled=vad_enabled,
            vad_threshold=vad_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            flow_context=self.flow_context_var.get(),
            hift_mode=self.hift_mode_var.get(),
            disable_prompt_kv_cache=not self.prompt_cache_var.get(),
            disable_history_kv_cache=not self.history_cache_var.get(),
            prompt_cache_max_mb=prompt_cache_max_mb,
            prompt_cache_max_seconds=prompt_cache_max_seconds,
            prompt_cache_dtype=prompt_cache_dtype,
            prompt_cache_storage=prompt_cache_storage,
            prompt_runtime_policy=prompt_runtime_policy,
            enable_grouped_prompt=prompt_runtime_policy == "grouped",
            enable_grouped_prompt_cache=prompt_runtime_policy == "grouped",
            lavasr_enabled=lavasr_enabled,
            lavasr_lowpass_hz=lavasr_lowpass_hz,
        )

    def _positive_float(self, variable: TextValue, name: str) -> float:
        value = self._float(variable, name)
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")
        return value

    def _nonnegative_float(self, variable: TextValue, name: str) -> float:
        value = self._float(variable, name)
        if value < 0:
            raise ValueError(f"{name} must be 0 or greater.")
        return value

    def _float_in_range(self, variable: TextValue, name: str, minimum: float, maximum: float) -> float:
        value = self._float(variable, name)
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
        return value

    def _float(self, variable: TextValue, name: str) -> float:
        try:
            return float(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error

    def _offline_worker(self, config: BackendConfig) -> None:
        try:
            self.backend.run_offline(
                config,
                stop_event=self.offline_stop_event,
                status_fn=lambda message: self._post("status", message),
                log_fn=lambda message: self._post("log", message),
                metrics_fn=lambda row, player_stats=None: self._post_metrics(row, player_stats),
            )
        except Exception as error:
            self._post_exception("Offline benchmark failed", error)
        finally:
            self._post("finished", "offline")

    def _package_worker(
        self,
        config: BackendConfig,
        references: list[str],
        output_path: str,
        options: dict,
    ) -> None:
        try:
            self.backend.create_voice_package(
                config,
                references,
                output_path,
                options,
                status_fn=lambda message: self._post("status", message),
                log_fn=lambda message: self._post("log", message),
            )
            self._post("voice_package_created", output_path)
        except Exception as error:
            self._post_exception("Voice package creation failed", error)
        finally:
            self._post("finished", "package")

    def _realtime_worker(self, config: BackendConfig) -> None:
        try:
            self.backend.run_realtime(
                config,
                stop_event=self.live_stop_event,
                status_fn=lambda message: self._post("status", message),
                log_fn=lambda message: self._post("log", message),
                metrics_fn=lambda row, player_stats=None: self._post_metrics(row, player_stats),
            )
        except Exception as error:
            self._post_exception("Live stream failed", error)
        finally:
            self._post("finished", "realtime")

    def _post_metrics(self, row: dict, player_stats: dict | None = None) -> None:
        input_clock = row["end_token"] / 25.0
        lag = row["wall_end_seconds"] - input_clock
        payload = {
            "chunk": str(row["chunk"]),
            "rtf": f"{row['chunk_rtf']:.2f}",
            "lag": f"{lag:.2f}s",
            "buffer": "-",
            "underflows": "-",
        }
        if player_stats is not None:
            payload["buffer"] = f"{player_stats['buffer_seconds']:.2f}s"
            payload["underflows"] = str(player_stats["underflows"])
        self._post("metrics", payload)

    def _post_exception(self, title: str, error: Exception) -> None:
        import traceback

        self._post("log", f"{title}: {error}")
        self._post("log", traceback.format_exc())
        self._post("status", title)

    def _is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _set_running(self, mode: str, running: bool) -> None:
        self.active_mode = mode if running else None
        self.start_live_button.configure(state="disabled" if running else "normal")
        self.run_offline_button.configure(state="disabled" if running else "normal")
        if hasattr(self, "create_package_button"):
            self.create_package_button.configure(state="disabled" if running else "normal")
        self.stop_live_button.configure(state="normal" if running and mode == "realtime" else "disabled")
        self.stop_offline_button.configure(state="normal" if running and mode == "offline" else "disabled")
        self.status_var.set("Starting..." if running else "Ready")

    def _post(self, kind: str, payload: object) -> None:
        self.ui_queue.put((kind, payload))

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._log(str(payload))
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "metrics":
                metrics = payload
                self.metric_chunk_var.set(metrics["chunk"])
                self.metric_rtf_var.set(metrics["rtf"])
                self.metric_lag_var.set(metrics["lag"])
                self.metric_buffer_var.set(metrics["buffer"])
                self.metric_underflow_var.set(metrics["underflows"])
            elif kind == "voice_package_created":
                self.voice_package_var.set(str(payload))
                self._inspect_package_current()
            elif kind == "finished":
                self._set_running(str(payload), False)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(f"[{timestamp}] {message}\n")
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self.live_stop_event is not None:
            self.live_stop_event.set()
        if self.offline_stop_event is not None:
            self.offline_stop_event.set()
        self.backend.shutdown()

    def _on_close(self) -> None:
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown()
        event.accept()
