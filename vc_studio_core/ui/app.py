from __future__ import annotations

import argparse
import queue
import sys
import threading

from PyQt6 import QtCore, QtWidgets

from vc_studio_core.backend import VCStudioBackend

from .controls import ControlMixin
from .package_panel import PackagePanelMixin
from .runtime import RuntimeMixin
from .state import BoolValue, TextValue
from .style import StyleMixin
from .views import ViewBuilderMixin
from .widgets import PALETTE


def launch_gui(args: argparse.Namespace) -> None:
    qt_app = QtWidgets.QApplication.instance()
    created_app = qt_app is None
    if qt_app is None:
        qt_app = QtWidgets.QApplication(sys.argv[:1])
    qt_app.setApplicationName("CosyVoice VC Studio")
    qt_app.setStyle("Fusion")
    window = VCStudioApp(args, qt_app)
    qt_app._vc_studio_window = window
    window.show()
    if created_app:
        qt_app.exec()


class VCStudioApp(
    StyleMixin,
    ViewBuilderMixin,
    ControlMixin,
    PackagePanelMixin,
    RuntimeMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self, args: argparse.Namespace, qt_app: QtWidgets.QApplication):
        super().__init__()
        self.qt_app = qt_app
        self.colors = PALETTE.copy()
        self.setWindowTitle("CosyVoice VC Studio")
        self.resize(1240, 860)
        self.setMinimumSize(1040, 720)
        self.setObjectName("MainWindow")

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.active_mode: str | None = None
        self.offline_stop_event: threading.Event | None = None
        self.live_stop_event: threading.Event | None = None
        self.backend = VCStudioBackend()
        self.input_device_map: dict[str, int | None] = {"Default": None}
        self.output_device_map: dict[str, int | None] = {"Default": None}
        self._shutdown_done = False

        self._create_variables(args)
        self._configure_style()
        self._build_ui()
        self._refresh_audio_devices()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._poll_ui_queue)
        self.poll_timer.start(100)

    def mainloop(self) -> None:
        self.show()
        self.qt_app.exec()

    def _create_variables(self, args: argparse.Namespace) -> None:
        self.model_dir_var = TextValue(args.model_dir)
        self.voice_package_var = TextValue(args.voice_package)
        self.prompt_var = TextValue(args.prompt)
        self.source_var = TextValue(args.source)
        self.output_var = TextValue(args.output)
        self.csv_var = TextValue(args.csv)
        self.device_var = TextValue(args.device)
        self.ort_provider_var = TextValue(args.ort_provider)
        self.coreml_cache_var = TextValue(args.coreml_cache_dir)
        if getattr(args, "prompt_runtime_policy", "auto") != "auto":
            prompt_runtime_policy = args.prompt_runtime_policy
        elif getattr(args, 'enable_grouped_prompt', False) or getattr(args, 'enable_grouped_prompt_cache', False):
            prompt_runtime_policy = "grouped"
        else:
            prompt_runtime_policy = "auto"
        self.prompt_runtime_policy_var = TextValue(prompt_runtime_policy)
        self.chunk_sec_var = TextValue(f"{args.chunk_sec:g}")
        self.tokenizer_chunk_sec_var = TextValue(f"{args.tokenizer_chunk_sec:g}")
        self.tokenizer_left_context_sec_var = TextValue(f"{args.tokenizer_left_context_sec:g}")
        self.tokenizer_right_context_sec_var = TextValue(f"{args.tokenizer_right_context_sec:g}")
        self.history_sec_var = TextValue(f"{args.history_sec:g}")
        self.mel_overlap_sec_var = TextValue(f"{args.mel_overlap_sec:g}")
        self.delayed_commit_sec_var = TextValue(f"{args.delayed_commit_sec:g}")
        self.audio_declick_ms_var = TextValue(f"{args.audio_declick_ms:g}")
        self.audio_blend_ms_var = TextValue(f"{args.audio_blend_ms:g}")
        self.lavasr_enabled_var = BoolValue(not getattr(args, "disable_lavasr", False))
        self.lavasr_lowpass_hz_var = TextValue(f"{getattr(args, 'lavasr_lowpass_hz', 7800.0):g}")
        self.vad_enabled_var = BoolValue(args.enable_vad)
        self.vad_threshold_var = TextValue(f"{args.vad_threshold:g}")
        self.vad_min_speech_ms_var = TextValue(f"{args.vad_min_speech_ms:g}")
        self.vad_min_silence_ms_var = TextValue(f"{args.vad_min_silence_ms:g}")
        self.vad_speech_pad_ms_var = TextValue(f"{args.vad_speech_pad_ms:g}")
        self.flow_context_var = TextValue(args.flow_context)
        self.hift_mode_var = TextValue(args.hift_mode)
        self.prompt_cache_var = BoolValue(not args.disable_prompt_kv_cache)
        self.history_cache_var = BoolValue(not args.disable_history_kv_cache)
        self.prompt_cache_max_mb_var = TextValue(f"{getattr(args, 'prompt_cache_max_mb', 1024.0):g}")
        self.prompt_cache_max_seconds_var = TextValue(f"{getattr(args, 'prompt_cache_max_seconds', 0.0):g}")
        self.prompt_cache_dtype_var = TextValue(getattr(args, 'prompt_cache_dtype', "auto"))
        self.prompt_cache_storage_var = TextValue(getattr(args, 'prompt_cache_storage', "device"))
        self.input_device_var = TextValue("Default")
        self.output_device_var = TextValue("Default")
        self.status_var = TextValue("Ready")
        self.metric_chunk_var = TextValue("-")
        self.metric_rtf_var = TextValue("-")
        self.metric_lag_var = TextValue("-")
        self.metric_buffer_var = TextValue("-")
        self.metric_underflow_var = TextValue("-")
        self.package_output_var = TextValue("out/voice.cvvoice")
        self.package_portrait_var = TextValue("")
        self.package_display_name_var = TextValue("")
        self.package_short_description_var = TextValue("")
        self.package_fusion_mode_var = TextValue("equal_weight")
        self.package_branch_gamma_var = TextValue("1.0")
        self.package_attention_temperature_var = TextValue("1.0")
        self.package_canonical_seconds_var = TextValue("10.0")
        self.package_soft_prompt_var = BoolValue(getattr(args, "enable_soft_prompt", True))
        self.package_soft_prompt_seconds_var = TextValue(f"{getattr(args, 'soft_prompt_seconds', 15.0):g}")
        self.package_soft_prompt_steps_var = TextValue(f"{getattr(args, 'soft_prompt_steps', 300):g}")
        self.package_soft_prompt_teacher_var = TextValue(getattr(args, "soft_prompt_teacher_mode", "grouped_branch_attention"))
        self.package_soft_prompt_checkpointing_var = TextValue(
            getattr(args, "soft_prompt_activation_checkpointing", "auto")
        )
        self.package_soft_prompt_segments_var = TextValue(f"{getattr(args, 'soft_prompt_checkpoint_segments', 3):g}")
