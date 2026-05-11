from __future__ import annotations

import argparse
import importlib.util
import math
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from cosyvoice.vc.voice_package import (
    read_voice_package_metadata,
    validate_model_compatibility,
)
from vc_studio_backend import (
    BackendConfig,
    StreamSettings,
    VCStudioBackend,
)


PALETTE = {
    "bg": "#FFF0F5",
    "surface": "#FFE4E1",
    "surface_alt": "#F0F8FF",
    "panel": "#FFF5EE",
    "card": "#FFFFFF",
    "field": "#FFFDF8",
    "text": "#5C4B51",
    "muted": "#9A8C98",
    "line": "#FAD2E1",
    "pink": "#FFC2D1",
    "pink_strong": "#FF8FAB",
    "mint": "#C1FBA4",
    "mint_strong": "#7BF1A8",
    "sky": "#A2D2FF",
    "sky_soft": "#C7CEEA",
    "cream": "#FFDAC1",
    "yellow": "#FDFD96",
    "purple": "#E2C6FF",
    "purple_strong": "#CDB4DB",
    "danger": "#FFB4A2",
    "danger_hover": "#E5989B",
    "disabled": "#E5E5E5",
}


class TextValue(QtCore.QObject):
    changed = QtCore.pyqtSignal(str)

    def __init__(self, value: str = ""):
        super().__init__()
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        value = str(value)
        if value == self._value:
            return
        self._value = value
        self.changed.emit(value)


class BoolValue(QtCore.QObject):
    changed = QtCore.pyqtSignal(bool)

    def __init__(self, value: bool = False):
        super().__init__()
        self._value = bool(value)

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        value = bool(value)
        if value == self._value:
            return
        self._value = value
        self.changed.emit(value)


class KawaiiBackdrop(QtWidgets.QWidget):
    def __init__(self, colors: dict[str, str]):
        super().__init__()
        self.colors = colors
        self.phase = 0.0
        self.elements = [
            ('petal', 0.06, 0.08, 0.16, 12, colors["pink"]),
            ('star', 0.18, 0.62, 0.10, 10, colors["cream"]),
            ('heart', 0.34, 0.22, 0.14, 15, colors["pink_strong"]),
            ('star', 0.52, 0.78, 0.12, 14, colors["mint"]),
            ('petal', 0.71, 0.14, 0.09, 10, colors["sky"]),
            ('heart', 0.84, 0.54, 0.13, 16, colors["purple"]),
            ('star', 0.94, 0.30, 0.11, 12, colors["yellow"]),
        ]
        self.sparkles = [
            (0.12, 0.35, colors["sky"]),
            (0.27, 0.12, colors["mint"]),
            (0.48, 0.48, colors["purple"]),
            (0.69, 0.28, colors["pink"]),
            (0.88, 0.75, colors["cream"]),
        ]
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def _tick(self) -> None:
        self.phase = (self.phase + 0.003) % 1.0
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(self.colors["bg"]))

        width = self.width()
        height = self.height()
        self._draw_wave(painter, width, height, self.colors["surface"], 0.08, 0.23, 0.0)
        self._draw_wave(painter, width, height, self.colors["surface_alt"], 0.16, 0.30, 0.5)

        for el_type, x_frac, y_frac, speed, size, color in self.elements:
            x = x_frac * width + 20 * math.sin((self.phase * 5 + x_frac) * math.tau)
            y = (y_frac + self.phase * speed * 2) % 1.1 * height - 30
            angle = self.phase * 360 + x_frac * 180
            
            if el_type == 'petal':
                self._draw_petal(painter, x, y, size, color, angle)
            elif el_type == 'star':
                self._draw_star(painter, x, y, size, color, 180, angle)
            elif el_type == 'heart':
                self._draw_heart(painter, x, y, size, color, 160, angle)

        for x_frac, y_frac, color in self.sparkles:
            alpha = 70 + int(85 * (1 + math.sin((self.phase * 4 + x_frac) * math.tau)))
            self._draw_sparkle(painter, x_frac * width, y_frac * height, color, alpha)

    def _draw_wave(
        self,
        painter: QtGui.QPainter,
        width: int,
        height: int,
        color: str,
        top: float,
        depth: float,
        phase_offset: float,
    ) -> None:
        path = QtGui.QPainterPath()
        path.moveTo(0, top * height)
        p1y = depth * height + 15 * math.sin((self.phase * 2 + phase_offset) * math.tau)
        p2y = top * height + 15 * math.cos((self.phase * 2 + phase_offset) * math.tau)
        p3y = depth * height + 15 * math.sin((self.phase * 2 + phase_offset + 0.5) * math.tau)
        path.cubicTo(width * 0.22, p1y, width * 0.46, p2y, width * 0.68, p3y)
        path.cubicTo(width * 0.84, p2y, width, p1y, width, top * height)
        path.lineTo(width, 0)
        path.lineTo(0, 0)
        path.closeSubpath()
        fill = QtGui.QColor(color)
        fill.setAlpha(180)
        painter.fillPath(path, fill)

    def _draw_petal(self, painter: QtGui.QPainter, x: float, y: float, size: float, color: str, angle: float) -> None:
        import math
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        fill = QtGui.QColor(color)
        fill.setAlpha(160)
        painter.setBrush(fill)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(QtCore.QRectF(-size * 0.42, -size * 0.18, size * 0.84, size * 1.28))
        painter.restore()

    def _draw_star(self, painter: QtGui.QPainter, x: float, y: float, size: float, color: str, alpha: int, angle: float) -> None:
        import math
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        fill = QtGui.QColor(color)
        fill.setAlpha(alpha)
        painter.setBrush(fill)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        path = QtGui.QPainterPath()
        points = 5
        outer_radius = size
        inner_radius = size * 0.4
        for i in range(points * 2):
            radius = outer_radius if i % 2 == 0 else inner_radius
            theta = math.pi * i / points - math.pi / 2
            px = radius * math.cos(theta)
            py = radius * math.sin(theta)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        path.closeSubpath()
        painter.drawPath(path)
        painter.restore()

    def _draw_heart(self, painter: QtGui.QPainter, x: float, y: float, size: float, color: str, alpha: int, angle: float) -> None:
        import math
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        fill = QtGui.QColor(color)
        fill.setAlpha(alpha)
        painter.setBrush(fill)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        path = QtGui.QPainterPath()
        path.moveTo(0, -size * 0.2)
        path.cubicTo(-size * 0.8, -size * 0.8, -size, size * 0.3, 0, size * 0.8)
        path.moveTo(0, -size * 0.2)
        path.cubicTo(size * 0.8, -size * 0.8, size, size * 0.3, 0, size * 0.8)
        painter.drawPath(path)
        painter.restore()

    def _draw_sparkle(self, painter: QtGui.QPainter, x: float, y: float, color: str, alpha: int) -> None:
        pen_color = QtGui.QColor(color)
        pen_color.setAlpha(alpha)
        painter.setPen(QtGui.QPen(pen_color, 2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        painter.drawLine(QtCore.QPointF(x - 5, y), QtCore.QPointF(x + 5, y))
        painter.drawLine(QtCore.QPointF(x, y - 5), QtCore.QPointF(x, y + 5))


class CuteButton(QtWidgets.QPushButton):
    def __init__(
        self,
        text: str,
        command,
        *,
        base: str,
        hover: str,
        disabled: str,
        foreground: str = "#2B3A55",
        radius: int = 15,
        min_width: int = 96,
        height: int = 42,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent)
        self.base = base
        self.hover = hover
        self.pressed = self._mix(base, "#CDB4DB", 0.28)
        self.disabled = disabled
        self.foreground = foreground
        self.radius = radius
        self._hovered = False
        self._pressed = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setMinimumSize(min_width, height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.clicked.connect(command)

        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(8)
        self.shadow.setOffset(0, 4)
        self.shadow.setColor(QtGui.QColor(214, 159, 183, 85))
        self.setGraphicsEffect(self.shadow)
        self.shadow_animation = QtCore.QPropertyAnimation(self.shadow, b"blurRadius", self)
        self.shadow_animation.setDuration(150)
        self.shadow_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._apply_style()

    def configure(self, cnf=None, **kwargs) -> None:
        if cnf:
            kwargs.update(cnf)
        if "state" in kwargs:
            self.setEnabled(kwargs.pop("state") != "disabled")
        if "text" in kwargs:
            self.setText(str(kwargs.pop("text")))
        self._apply_style()

    config = configure

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = True
        self._animate_shadow(18)
        self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = False
        self._pressed = False
        self._animate_shadow(8)
        self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._pressed = True
        self._animate_shadow(4)
        self._apply_style()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._pressed = False
        self._animate_shadow(18 if self._hovered else 8)
        self._apply_style()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.EnabledChange:
            self._apply_style()

    def _animate_shadow(self, radius: int) -> None:
        self.shadow_animation.stop()
        self.shadow_animation.setStartValue(self.shadow.blurRadius())
        self.shadow_animation.setEndValue(radius)
        self.shadow_animation.start()

    def _apply_style(self) -> None:
        if not self.isEnabled():
            bg = self.disabled
            color = "#A0A0A0"
            border_color = self._mix(bg, "#FFFFFF", 0.3)
        elif self._pressed:
            bg = self.pressed
            color = self.foreground
            border_color = self._mix(bg, "#2B3A55", 0.15)
        elif self._hovered:
            bg = self.hover
            color = self.foreground
            border_color = self._mix(bg, "#2B3A55", 0.10)
        else:
            bg = self.base
            color = self.foreground
            border_color = self._mix(bg, "#2B3A55", 0.05)
        
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg};
                color: {color};
                border: 2px solid {border_color};
                border-radius: {self.radius}px;
                font-weight: 800;
                padding: 8px 18px;
            }}
            QPushButton:disabled {{
                color: #A0A0A0;
            }}
            """
        )

    @staticmethod
    def _mix(left: str, right: str, amount: float) -> str:
        left_color = QtGui.QColor(left)
        right_color = QtGui.QColor(right)
        inv = 1.0 - amount
        return "#{:02X}{:02X}{:02X}".format(
            round(left_color.red() * inv + right_color.red() * amount),
            round(left_color.green() * inv + right_color.green() * amount),
            round(left_color.blue() * inv + right_color.blue() * amount),
        )


class MetricCard(QtWidgets.QFrame):
    def __init__(self, name: str, variable: TextValue, colors: dict[str, str], accent: str):
        super().__init__()
        self.colors = colors
        self.accent = accent
        self.setObjectName("MetricCard")
        self.setMinimumHeight(68)
        self.setMaximumHeight(76)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(7)
        self.shadow.setOffset(0, 3)
        shadow_color = QtGui.QColor(accent)
        shadow_color.setAlpha(78)
        self.shadow.setColor(shadow_color)
        self.shadow.setEnabled(True)
        self.setGraphicsEffect(self.shadow)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(3)
        title = QtWidgets.QLabel(name)
        title.setObjectName("MetricName")
        self.value_label = QtWidgets.QLabel(variable.get())
        self.value_label.setObjectName("MetricValue")
        layout.addWidget(title)
        layout.addWidget(self.value_label)

        self.pulse = QtCore.QPropertyAnimation(self.shadow, b"blurRadius", self)
        self.pulse.setDuration(260)
        self.pulse.setKeyValueAt(0.0, 7)
        self.pulse.setKeyValueAt(0.45, 17)
        self.pulse.setKeyValueAt(1.0, 7)
        self.pulse.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        variable.changed.connect(self.set_value)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
        self.pulse.stop()
        self.pulse.start()


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


class VCStudioApp(QtWidgets.QMainWindow):
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

    def _configure_style(self) -> None:
        self.qt_app.setFont(QtGui.QFont("Nunito", 11))
        self.qt_app.setStyleSheet(
            f"""
            * {{
                color: {self.colors["text"]};
                font-family: "Nunito", "Quicksand", "Varela Round", "Comic Sans MS", "Avenir Next Rounded", "Helvetica Neue", Arial;
                font-size: 13px;
                letter-spacing: 0px;
            }}
            QMainWindow#MainWindow {{
                background: {self.colors["bg"]};
            }}
            QLabel#Title {{
                font-size: 28px;
                font-weight: 900;
                color: {self.colors["pink_strong"]};
            }}
            QLabel#Subtitle, QLabel#Muted {{
                color: {self.colors["muted"]};
                font-weight: 600;
            }}
            QLabel#SectionTitle {{
                font-size: 16px;
                font-weight: 900;
                color: {self.colors["text"]};
            }}
            QLabel#SmallTitle {{
                font-size: 14px;
                font-weight: 800;
            }}
            QLabel#ParamHelp {{
                color: {self.colors["muted"]};
                font-size: 12px;
                font-weight: 600;
            }}
            QToolButton#DisclosureButton {{
                background-color: {self.colors["surface_alt"]};
                border: 2px solid {self.colors["sky_soft"]};
                border-radius: 16px;
                padding: 10px 14px;
                color: {self.colors["text"]};
                font-weight: 800;
                text-align: left;
            }}
            QToolButton#DisclosureButton:hover {{
                background-color: {self.colors["purple"]};
                border-color: {self.colors["purple_strong"]};
            }}
            QFrame#AdvancedBody {{
                background-color: rgba(255, 255, 255, 180);
                border: 2px dashed {self.colors["line"]};
                border-radius: 20px;
            }}
            QLabel#StatusText {{
                font-weight: 800;
                color: {self.colors["text"]};
            }}
            QFrame#HeroPanel {{
                background-color: rgba(255, 255, 255, 160);
                border: 2px solid {self.colors["pink"]};
                border-radius: 30px;
            }}
            QFrame#SidePanel, QFrame#LogPanel {{
                background-color: rgba(255, 255, 255, 160);
                border: 2px solid {self.colors["line"]};
                border-radius: 26px;
            }}
            QFrame#Card {{
                background-color: rgba(255, 255, 255, 170);
                border: 2px solid {self.colors["line"]};
                border-radius: 24px;
            }}
            QFrame#MetricCard {{
                background-color: rgba(255, 255, 255, 180);
                border: 2px solid {self.colors["pink"]};
                border-radius: 20px;
                margin: 4px;
            }}
            QLabel#MetricName {{
                color: {self.colors["muted"]};
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }}
            QLabel#MetricValue {{
                color: {self.colors["pink_strong"]};
                font-size: 22px;
                font-weight: 900;
            }}
            QFrame#StatusPill {{
                background-color: {self.colors["mint"]};
                border: 2px solid {self.colors["mint_strong"]};
                border-radius: 20px;
            }}
            QFrame#AccentStrip {{
                background-color: {self.colors["pink_strong"]};
                border-radius: 6px;
            }}
            QLineEdit, QComboBox {{
                background-color: {self.colors["field"]};
                border: 2px solid {self.colors["line"]};
                border-radius: 16px;
                padding: 8px 14px;
                min-height: 22px;
                selection-background-color: {self.colors["pink"]};
                font-weight: 600;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 2px solid {self.colors["sky"]};
                background-color: #FFFFFF;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
            }}
            QComboBox::down-arrow {{
                image: none;
            }}
            QFrame#ComboPopupContainer {{
                background: transparent;
                border: none;
            }}
            QComboBox QAbstractItemView, QListView#ComboPopupView {{
                background-color: {self.colors["field"]};
                border: 2px solid {self.colors["line"]};
                border-radius: 12px;
                padding: 8px;
                selection-background-color: {self.colors["pink"]};
                outline: none;
            }}
            QPushButton#GhostButton {{
                background-color: {self.colors["surface_alt"]};
                border: 2px solid {self.colors["sky_soft"]};
                border-radius: 16px;
                padding: 8px 16px;
                font-weight: 800;
                color: {self.colors["text"]};
            }}
            QPushButton#GhostButton:hover {{
                background-color: {self.colors["sky"]};
                border-color: {self.colors["sky"]};
                color: #FFFFFF;
            }}
            QPushButton#GhostButton:pressed {{
                background-color: {self.colors["sky_soft"]};
            }}
            QPushButton#GhostButton:disabled {{
                background-color: {self.colors["disabled"]};
                border-color: {self.colors["disabled"]};
                color: #FFFFFF;
            }}
            QTabWidget::pane {{
                border: 2px solid {self.colors["line"]};
                border-radius: 26px;
                background-color: rgba(255, 255, 255, 160);
                margin-top: -2px;
            }}
            QTabBar::tab {{
                background-color: {self.colors["surface_alt"]};
                border: 2px solid {self.colors["line"]};
                border-bottom: none;
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
                padding: 12px 24px;
                margin-right: 8px;
                color: {self.colors["muted"]};
                font-weight: 800;
            }}
            QTabBar::tab:selected {{
                background-color: {self.colors["pink"]};
                border-color: {self.colors["pink_strong"]};
                color: {self.colors["text"]};
                margin-top: -2px;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {self.colors["purple"]};
                border-color: {self.colors["purple_strong"]};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea#TabScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#TabViewport {{
                background: transparent;
            }}
            QTextEdit {{
                background-color: {self.colors["field"]};
                border: 2px solid {self.colors["line"]};
                border-radius: 20px;
                padding: 14px;
                selection-background-color: {self.colors["pink"]};
                font-family: "Menlo", "Consolas", monospace;
                font-size: 12px;
                color: {self.colors["text"]};
            }}
            QCheckBox {{
                spacing: 12px;
                font-weight: 700;
                color: {self.colors["muted"]};
            }}
            QCheckBox:checked {{
                color: {self.colors["text"]};
            }}
            QCheckBox::indicator {{
                width: 22px;
                height: 22px;
                border-radius: 11px;
                border: 2px solid {self.colors["line"]};
                background-color: {self.colors["field"]};
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {self.colors["pink"]};
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.colors["mint"]};
                border: 2px solid {self.colors["mint_strong"]};
                image: none;
            }}
            QSplitter::handle {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 14px;
                margin: 8px 2px 8px 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.colors["pink"]};
                border-radius: 5px;
                min-height: 32px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {self.colors["pink_strong"]};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

    def _build_ui(self) -> None:
        root = KawaiiBackdrop(self.colors)
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        layout.addWidget(self._build_header())

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        self.model_panel = QtWidgets.QFrame()
        self.model_panel.setObjectName("SidePanel")
        self.model_panel.setFixedWidth(390)
        self._build_model_panel(self.model_panel)
        body.addWidget(self.model_panel)

        self.notebook = QtWidgets.QTabWidget()
        self.notebook.setDocumentMode(True)
        self._build_voice_package_tab(self.notebook)
        self._build_realtime_tab(self.notebook)
        self._build_offline_tab(self.notebook)
        self._build_parameters_tab(self.notebook)
        body.addWidget(self.notebook, 1)
        layout.addLayout(body, 1)

        layout.addWidget(self._build_log_panel())

    def _build_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header.setObjectName("HeroPanel")
        shadow = QtWidgets.QGraphicsDropShadowEffect(header)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 8)
        shadow.setColor(QtGui.QColor(224, 177, 203, 65))
        header.setGraphicsEffect(shadow)

        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(16)

        accent = QtWidgets.QFrame()
        accent.setObjectName("AccentStrip")
        accent.setFixedSize(8, 54)
        layout.addWidget(accent)

        title_block = QtWidgets.QVBoxLayout()
        title_block.setSpacing(4)
        title = QtWidgets.QLabel("CosyVoice VC Studio")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("Realtime voice conversion and offline benchmark in one soft control room.")
        subtitle.setObjectName("Subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        layout.addLayout(title_block, 1)

        status_pill = QtWidgets.QFrame()
        status_pill.setObjectName("StatusPill")
        status_layout = QtWidgets.QHBoxLayout(status_pill)
        status_layout.setContentsMargins(16, 7, 16, 7)
        status_layout.setSpacing(8)
        dot = QtWidgets.QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background-color: {self.colors['mint_strong']}; border-radius: 5px;")
        status_text = QtWidgets.QLabel(self.status_var.get())
        status_text.setObjectName("StatusText")
        self.status_var.changed.connect(status_text.setText)
        status_layout.addWidget(dot)
        status_layout.addWidget(status_text)
        layout.addWidget(status_pill, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return header

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("LogPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(9)
        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(self._section_title("📝 Run Log", self.colors["sky"]))
        title_row.addStretch(1)
        layout.addLayout(title_row)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(128)
        layout.addWidget(self.log_text)
        return panel

    def _build_model_panel(self, parent: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("🌸 Model", self.colors["pink"]))

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(16)
        form.setColumnStretch(1, 1)
        row = 0
        row = self._path_row(form, row, "Model dir", self.model_dir_var, "directory")
        row = self._path_row(form, row, "Voice package", self.voice_package_var, "open_cvvoice")
        row = self._combo_row(form, row, "Prompt mode", self.prompt_runtime_policy_var, ["auto", "soft", "grouped", "dominant"])
        row = self._combo_row(form, row, "Torch device", self.device_var, ["auto", "cpu", "cuda", "mps"])
        row = self._combo_row(form, row, "ORT provider", self.ort_provider_var, ["auto", "cpu", "cuda", "coreml"])
        self._path_row(form, row, "CoreML cache", self.coreml_cache_var, "directory")
        layout.addLayout(form)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {self.colors['line']};")
        layout.addWidget(divider)

        layout.addWidget(self._section_title("✨ Metrics", self.colors["mint"]))
        metrics = QtWidgets.QGridLayout()
        metrics.setSpacing(16)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        self._metric_card(metrics, 0, 0, "Chunk", self.metric_chunk_var, self.colors["pink"])
        self._metric_card(metrics, 0, 1, "RTF", self.metric_rtf_var, self.colors["sky"])
        self._metric_card(metrics, 1, 0, "Lag", self.metric_lag_var, self.colors["cream"])
        self._metric_card(metrics, 1, 1, "Buffer", self.metric_buffer_var, self.colors["mint"])
        self._metric_card(metrics, 2, 0, "Underflows", self.metric_underflow_var, self.colors["purple"], 2)
        layout.addLayout(metrics)
        layout.addStretch(1)

    def _build_voice_package_tab(self, notebook: QtWidgets.QTabWidget) -> None:
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        create_card = self._card()
        create_layout = QtWidgets.QVBoxLayout(create_card)
        create_layout.setContentsMargins(18, 18, 18, 18)
        create_layout.setSpacing(12)
        create_layout.addWidget(self._section_title("🎁 Create Voice Package", self.colors["pink"]))

        self.reference_table = QtWidgets.QTableWidget(0, 3)
        self.reference_table.setHorizontalHeaderLabels(["Reference WAV", "Raw weight", "Normalized"])
        self.reference_table.verticalHeader().setVisible(False)
        self.reference_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.reference_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.reference_table.horizontalHeader().setStretchLastSection(False)
        self.reference_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.reference_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.reference_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.reference_table.setMinimumHeight(185)
        self.reference_table.itemChanged.connect(lambda _item: self._update_reference_weights())
        create_layout.addWidget(self.reference_table)

        ref_buttons = QtWidgets.QHBoxLayout()
        ref_buttons.setSpacing(8)
        for text, command in [
            ("Add WAV", self._add_reference_files),
            ("Add Folder", self._add_reference_folder),
            ("Remove", self._remove_selected_references),
            ("Up", lambda: self._move_selected_reference(-1)),
            ("Down", lambda: self._move_selected_reference(1)),
        ]:
            ref_buttons.addWidget(self._ghost_button(text, command))
        ref_buttons.addStretch(1)
        create_layout.addLayout(ref_buttons)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self._combo_form_row(
            form,
            "Fusion mode",
            self.package_fusion_mode_var,
            ["equal_weight", "duration_weight", "manual_weight"],
            "Controls how accepted reference branches are weighted before normalization.",
        )
        self.package_fusion_mode_var.changed.connect(lambda _value: self._update_reference_weights())
        self._number_row(form, "Branch gamma", self.package_branch_gamma_var, "Sharpens cross-branch weights after normalization.")
        self._number_row(
            form,
            "Attention temp",
            self.package_attention_temperature_var,
            "Stored with the package for grouped prompt attention compatibility.",
        )
        self._number_row(
            form,
            "Canonical sec",
            self.package_canonical_seconds_var,
            "Stable source position base used by package metadata.",
        )
        self._checkbox_row(
            form,
            "Soft prompt",
            self.package_soft_prompt_var,
            "Distills all references into one fixed-length continuous prompt for constant-cost package runtime.",
        )
        self._number_row(
            form,
            "Soft prompt sec",
            self.package_soft_prompt_seconds_var,
            "Target soft prompt length. The value is converted to mel frames and aligned for prompt caching.",
        )
        self._number_row(
            form,
            "Soft prompt steps",
            self.package_soft_prompt_steps_var,
            "Offline optimization steps. 200 to 500 is the intended first budget; 0 stores initialization only.",
        )
        self._combo_form_row(
            form,
            "Soft teacher",
            self.package_soft_prompt_teacher_var,
            ["grouped_branch_attention", "init_only"],
            "Teacher for offline distillation. init_only skips training and stores the weighted reference initialization.",
        )
        self._combo_form_row(
            form,
            "Soft checkpoint",
            self.package_soft_prompt_checkpointing_var,
            ["auto", "on", "off"],
            "Activation checkpointing policy for soft prompt training only.",
        )
        self._number_row(
            form,
            "Soft segments",
            self.package_soft_prompt_segments_var,
            "Checkpoint segments across the distillation layers when checkpointing is enabled.",
        )
        self._path_form_row(
            form,
            "Portrait",
            self.package_portrait_var,
            "open_image",
            "Optional PNG, JPEG, or WEBP portrait stored inside the package.",
        )
        self._path_form_row(
            form,
            "Output",
            self.package_output_var,
            "save_cvvoice",
            "Destination .cvvoice file.",
        )
        self._text_form_row(form, "Display name", self.package_display_name_var, "Name shown in package inspection.")
        self._text_form_row(form, "Short note", self.package_short_description_var, "Brief package description.")
        create_layout.addLayout(form)

        long_label = self._field_label("Long description")
        self.package_long_description_edit = QtWidgets.QTextEdit()
        self.package_long_description_edit.setMinimumHeight(95)
        self.package_long_description_edit.setAcceptRichText(False)
        create_layout.addWidget(long_label)
        create_layout.addWidget(self.package_long_description_edit)

        action_row = QtWidgets.QHBoxLayout()
        self.create_package_button = CuteButton(
            "Create Package",
            self._start_package_create,
            base=self.colors["mint"],
            hover=self.colors["mint_strong"],
            disabled=self.colors["disabled"],
            min_width=142,
        )
        action_row.addWidget(self.create_package_button)
        action_row.addStretch(1)
        create_layout.addLayout(action_row)
        layout.addWidget(create_card, 3)

        inspect_card = self._card()
        inspect_layout = QtWidgets.QVBoxLayout(inspect_card)
        inspect_layout.setContentsMargins(18, 18, 18, 18)
        inspect_layout.setSpacing(12)
        inspect_layout.addWidget(self._section_title("🔎 Inspect Package", self.colors["sky"]))
        inspect_form = QtWidgets.QGridLayout()
        inspect_form.setHorizontalSpacing(8)
        inspect_form.setVerticalSpacing(10)
        inspect_form.setColumnStretch(1, 1)
        self._path_row(inspect_form, 0, "Package", self.voice_package_var, "open_cvvoice")
        inspect_layout.addLayout(inspect_form)
        inspect_button = self._ghost_button("Inspect", self._inspect_package_current)
        inspect_layout.addWidget(inspect_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.package_inspect_text = QtWidgets.QTextEdit()
        self.package_inspect_text.setReadOnly(True)
        self.package_inspect_text.setMinimumHeight(360)
        inspect_layout.addWidget(self.package_inspect_text, 1)
        layout.addWidget(inspect_card, 2)

        notebook.addTab(self._scroll(content), "🎁 Voice Package")

    def _build_realtime_tab(self, notebook: QtWidgets.QTabWidget) -> None:
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        audio_card = self._card()
        audio_layout = QtWidgets.QGridLayout(audio_card)
        audio_layout.setContentsMargins(18, 18, 18, 18)
        audio_layout.setHorizontalSpacing(10)
        audio_layout.setVerticalSpacing(12)
        audio_layout.setColumnStretch(1, 1)
        audio_layout.addWidget(self._section_title("🎧 Audio I/O", self.colors["sky"]), 0, 0, 1, 3)
        audio_layout.addWidget(self._field_label("Input"), 1, 0)
        self.input_device_combo = QtWidgets.QComboBox()
        self._bind_combo(self.input_device_combo, self.input_device_var, ["Default"])
        audio_layout.addWidget(self.input_device_combo, 1, 1)
        audio_layout.addWidget(self._field_label("Output"), 2, 0)
        self.output_device_combo = QtWidgets.QComboBox()
        self._bind_combo(self.output_device_combo, self.output_device_var, ["Default"])
        audio_layout.addWidget(self.output_device_combo, 2, 1)
        refresh_button = self._ghost_button("Refresh", self._refresh_audio_devices)
        audio_layout.addWidget(refresh_button, 1, 2, 2, 1)
        layout.addWidget(audio_card)

        control_card = self._card()
        control_layout = QtWidgets.QVBoxLayout(control_card)
        control_layout.setContentsMargins(18, 18, 18, 18)
        control_layout.setSpacing(14)
        control_layout.addWidget(self._section_title("🎀 Live Control", self.colors["pink"]))
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(10)
        self.start_live_button = CuteButton(
            "✨ Start Live",
            self._start_realtime,
            base=self.colors["mint"],
            hover=self.colors["mint_strong"],
            disabled=self.colors["disabled"],
            min_width=132,
        )
        self.stop_live_button = CuteButton(
            "🛑 Stop",
            self._stop_realtime,
            base=self.colors["danger"],
            hover=self.colors["danger_hover"],
            disabled=self.colors["disabled"],
            foreground="#FFF8F8",
            min_width=92,
        )
        self.stop_live_button.configure(state="disabled")
        buttons.addWidget(self.start_live_button)
        buttons.addWidget(self.stop_live_button)
        buttons.addStretch(1)
        control_layout.addLayout(buttons)
        hint = QtWidgets.QLabel(
            "Use headphones to avoid feedback. Smaller chunks reduce latency; more context can improve stability."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        control_layout.addWidget(hint)
        layout.addWidget(control_card)
        layout.addStretch(1)

        notebook.addTab(self._scroll(content), "🎙️ Realtime")

    def _build_offline_tab(self, notebook: QtWidgets.QTabWidget) -> None:
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        job_card = self._card()
        grid = QtWidgets.QGridLayout(job_card)
        grid.setContentsMargins(18, 18, 18, 18)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.addWidget(self._section_title("🚀 Benchmark Job", self.colors["purple"]), 0, 0, 1, 3)
        row = 1
        row = self._path_row(grid, row, "Source wav", self.source_var, "open_wav")
        row = self._path_row(grid, row, "Output wav", self.output_var, "save_wav")
        row = self._path_row(grid, row, "CSV report", self.csv_var, "save_csv")
        layout.addWidget(job_card)

        controls = self._card()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(10)
        self.run_offline_button = CuteButton(
            "🚀 Run Benchmark",
            self._start_offline,
            base=self.colors["sky"],
            hover=self.colors["sky_soft"],
            disabled=self.colors["disabled"],
            min_width=158,
        )
        self.stop_offline_button = CuteButton(
            "🛑 Stop",
            self._stop_offline,
            base=self.colors["danger"],
            hover=self.colors["danger_hover"],
            disabled=self.colors["disabled"],
            foreground="#FFF8F8",
            min_width=92,
        )
        self.stop_offline_button.configure(state="disabled")
        controls_layout.addWidget(self.run_offline_button)
        controls_layout.addWidget(self.stop_offline_button)
        controls_layout.addStretch(1)
        layout.addWidget(controls)
        layout.addStretch(1)

        notebook.addTab(self._scroll(content), "📊 Offline Benchmark")

    def _build_parameters_tab(self, notebook: QtWidgets.QTabWidget) -> None:
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        timing = self._card()
        timing_layout = QtWidgets.QVBoxLayout(timing)
        timing_layout.setContentsMargins(18, 18, 18, 18)
        timing_layout.setSpacing(12)
        timing_layout.addWidget(self._section_title("⏱️ Timing", self.colors["cream"]))
        timing_form = QtWidgets.QFormLayout()
        timing_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        timing_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        timing_form.setHorizontalSpacing(16)
        timing_form.setVerticalSpacing(14)
        for label, var, description in [
            (
                "Chunk sec",
                self.chunk_sec_var,
                "Committed source duration per inference step. Higher values can improve continuity, "
                "but add latency and more work per update; lower values feel faster but may sound less stable.",
            ),
            (
                "Tokenizer chunk",
                self.tokenizer_chunk_sec_var,
                "Speech-tokenizer step size; 0 follows Chunk sec. Higher values reduce boundary churn "
                "and overhead, but increase wait time; lower values update sooner with more edge risk.",
            ),
            (
                "Tokenizer left ctx",
                self.tokenizer_left_context_sec_var,
                "Past audio supplied only to stabilize token boundaries. Raising it can smooth consonants "
                "and phrase starts, but increases tokenizer compute.",
            ),
            (
                "Tokenizer right ctx",
                self.tokenizer_right_context_sec_var,
                "Future audio lookahead for tokenizer decisions. Raising it often improves endings and "
                "boundary quality, but directly increases latency.",
            ),
            (
                "History sec",
                self.history_sec_var,
                "Past converted tokens prepended to each flow window. Higher values improve prosody and "
                "speaker continuity, but raise attention/vocoder load.",
            ),
            (
                "Mel overlap sec",
                self.mel_overlap_sec_var,
                "Extra mel context blended across neighboring chunks. Higher values smooth joins and can "
                "improve quality, but add compute and may soften timing.",
            ),
            (
                "Delayed commit sec",
                self.delayed_commit_sec_var,
                "Holds output until extra future context is available. Raising it can improve transitions "
                "and prosody, but increases output latency.",
            ),
        ]:
            self._number_row(timing_form, label, var, description)
        timing_layout.addLayout(timing_form)
        timing_layout.addStretch(1)
        layout.addWidget(timing, 1)

        quality = self._card()
        quality_layout = QtWidgets.QVBoxLayout(quality)
        quality_layout.setContentsMargins(18, 16, 18, 16)
        quality_layout.setSpacing(9)
        quality_layout.addWidget(self._section_title("💎 Quality / Runtime", self.colors["mint"]))
        quality_form = QtWidgets.QFormLayout()
        quality_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        quality_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        quality_form.setHorizontalSpacing(16)
        quality_form.setVerticalSpacing(12)
        self._number_row(
            quality_form,
            "De-click ms",
            self.audio_declick_ms_var,
            "Short fade at waveform boundaries. Raising it masks clicks, but can soften transients; "
            "0 leaves boundaries untouched.",
        )
        self._number_row(
            quality_form,
            "Audio blend ms",
            self.audio_blend_ms_var,
            "Crossfades adjacent audio chunks. Raising it smooths joins, but can smear attacks and adds "
            "a small post-processing cost.",
        )
        self._checkbox_row(
            quality_form,
            "LavaSR BWE",
            self.lavasr_enabled_var,
            "Expands converted 24 kHz speech by lowpass/resampling it to 16 kHz, then running LavaSR to "
            "produce 48 kHz output with synthesized high-band detail.",
        )
        self._number_row(
            quality_form,
            "LavaSR lowpass Hz",
            self.lavasr_lowpass_hz_var,
            "Cutoff used before the 16 kHz LavaSR input and as the low/high-band merge point. Higher values "
            "preserve more original VC output; lower values let LavaSR replace more upper spectrum.",
        )

        self._checkbox_row(
            quality_form,
            "Silero VAD gate",
            self.vad_enabled_var,
            "Detects non-speech and fades matching output to silence. Enabling it reduces room noise "
            "being converted into voice, but strict settings can mute quiet or short speech.",
        )
        self._number_row(
            quality_form,
            "VAD threshold",
            self.vad_threshold_var,
            "Speech probability cutoff. Raising it is stricter and suppresses more noise, but can miss "
            "soft speech; lowering it catches quieter speech with more false positives.",
        )
        self._number_row(
            quality_form,
            "VAD min speech ms",
            self.vad_min_speech_ms_var,
            "Minimum speech duration accepted by VAD. Raising it ignores brief noises, but can drop short "
            "words; lowering it reacts to shorter speech.",
        )
        self._number_row(
            quality_form,
            "VAD min silence ms",
            self.vad_min_silence_ms_var,
            "Silence duration required before closing a speech segment. Raising it avoids choppy gating, "
            "but holds noise longer; lowering it cuts faster.",
        )
        self._number_row(
            quality_form,
            "VAD speech pad ms",
            self.vad_speech_pad_ms_var,
            "Extra padding around detected speech. Raising it preserves starts and endings, but passes "
            "more room tone; lowering it gates tighter and may clip edges.",
        )
        self._checkbox_row(
            quality_form,
            "Prompt KV cache",
            self.prompt_cache_var,
            "Caches target-speaker prompt attention in streaming flow. Enabling it reduces repeated compute "
            "and latency; disabling can help diagnose cache-related artifacts.",
        )
        self._number_row(
            quality_form,
            "Prompt cache MiB",
            self.prompt_cache_max_mb_var,
            "Upper bound for prepared prompt KV cache memory. If the full prompt does not fit, cache is disabled "
            "instead of truncating the prompt. Set 0 for no automatic budget limit.",
        )
        self._number_row(
            quality_form,
            "Prompt cache sec",
            self.prompt_cache_max_seconds_var,
            "Maximum full prompt duration allowed in the KV cache. If the prompt is longer, cache is disabled "
            "and quality is preserved. Set 0 to follow the memory budget.",
        )
        quality_layout.addLayout(quality_form)

        advanced_panel, advanced_form = self._advanced_panel()
        self._path_form_row(
            advanced_form,
            "Legacy prompt WAV",
            self.prompt_var,
            "open_wav",
            "Used only when the Voice package field is empty.",
        )
        self._combo_form_row(
            advanced_form,
            "Flow context",
            self.flow_context_var,
            ["streaming", "window-full"],
            "Attention mode inside each flow window. window-full can improve local quality, but disables "
            "streaming caches and costs more; streaming is faster.",
        )
        self._combo_form_row(
            advanced_form,
            "HiFT mode",
            self.hift_mode_var,
            ["stateful", "window"],
            "Vocoder state strategy. stateful reuses caches for lower compute and real-time smoothness; "
            "window recomputes bounded context and is safer for quality debugging.",
        )
        self._combo_form_row(
            advanced_form,
            "Prompt cache dtype",
            self.prompt_cache_dtype_var,
            ["auto", "float32", "float16", "bfloat16"],
            "Storage dtype for cached K/V tensors. auto uses half precision on GPU/MPS and float32 on CPU.",
        )
        self._combo_form_row(
            advanced_form,
            "Prompt cache storage",
            self.prompt_cache_storage_var,
            ["device", "cpu_offload"],
            "Keep cached K/V on the active device, or store it in CPU memory and transfer per step.",
        )
        self._checkbox_row(
            advanced_form,
            "History KV cache",
            self.history_cache_var,
            "Caches reusable history attention when alignment allows. Enabling it lowers compute for longer "
            "context; disabling is slower but simpler.",
        )
        quality_layout.addWidget(advanced_panel)
        quality_layout.addStretch(1)
        layout.addWidget(quality, 1)

        notebook.addTab(self._scroll(content), "⚙️ Parameters")

    def _card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        shadow = QtWidgets.QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 5)
        shadow.setColor(QtGui.QColor(188, 174, 196, 42))
        card.setGraphicsEffect(shadow)
        return card

    def _scroll(self, content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("TabScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.viewport().setObjectName("TabViewport")
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setWidget(content)
        return scroll

    def _fill_widget(self, widget: QtWidgets.QWidget, color: str) -> None:
        palette = widget.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(color))
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)

    def _advanced_panel(self) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(8)

        button = QtWidgets.QToolButton()
        button.setObjectName("DisclosureButton")
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        button.setText("Advanced model/runtime controls")
        button.setCheckable(True)
        button.setChecked(False)
        button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        layout.addWidget(button)

        body = QtWidgets.QFrame()
        body.setObjectName("AdvancedBody")
        body.setVisible(False)
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        body_layout.addLayout(form)
        layout.addWidget(body)

        button.toggled.connect(lambda checked: self._set_disclosure_open(button, body, checked))
        return panel, form

    def _set_disclosure_open(
        self,
        button: QtWidgets.QToolButton,
        body: QtWidgets.QWidget,
        checked: bool,
    ) -> None:
        body.setVisible(checked)
        button.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)

    def _section_title(self, text: str, color: str) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        dot = QtWidgets.QFrame()
        dot.setFixedSize(12, 12)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
        label = QtWidgets.QLabel(text)
        label.setObjectName("SectionTitle")
        layout.addWidget(dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        return holder

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("SmallTitle")
        return label

    def _number_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        variable: TextValue,
        description: str,
    ) -> None:
        field = QtWidgets.QLineEdit()
        field.setMaximumWidth(190)
        self._bind_line_edit(field, variable)
        form.addRow(self._field_label(label), self._described_control(field, description))

    def _text_form_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        variable: TextValue,
        description: str,
    ) -> None:
        field = QtWidgets.QLineEdit()
        self._bind_line_edit(field, variable)
        form.addRow(self._field_label(label), self._described_control(field, description))

    def _path_form_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        variable: TextValue,
        kind: str,
        description: str,
    ) -> None:
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        field = QtWidgets.QLineEdit()
        self._bind_line_edit(field, variable)
        button = self._ghost_button("Browse", lambda: self._browse_path(variable, kind))
        row.addWidget(field, 1)
        row.addWidget(button)
        form.addRow(self._field_label(label), self._described_control(holder, description))

    def _path_row(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        label: str,
        variable: TextValue,
        kind: str,
    ) -> int:
        field = QtWidgets.QLineEdit()
        self._bind_line_edit(field, variable)
        button = self._ghost_button("Browse", lambda: self._browse_path(variable, kind))
        grid.addWidget(self._field_label(label), row, 0)
        grid.addWidget(field, row, 1)
        grid.addWidget(button, row, 2)
        return row + 1

    def _combo_row(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        label: str,
        variable: TextValue,
        values: list[str],
    ) -> int:
        combo = QtWidgets.QComboBox()
        self._bind_combo(combo, variable, values)
        grid.addWidget(self._field_label(label), row, 0)
        grid.addWidget(combo, row, 1, 1, 2)
        return row + 1

    def _combo_form_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        variable: TextValue,
        values: list[str],
        description: str,
    ) -> None:
        combo = QtWidgets.QComboBox()
        combo.setMaximumWidth(220)
        self._bind_combo(combo, variable, values)
        form.addRow(self._field_label(label), self._described_control(combo, description))

    def _checkbox_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        variable: BoolValue,
        description: str,
    ) -> None:
        checkbox = QtWidgets.QCheckBox()
        self._bind_checkbox(checkbox, variable)
        self._update_checkbox_text(checkbox, variable.get())
        checkbox.toggled.connect(lambda checked, widget=checkbox: self._update_checkbox_text(widget, checked))
        variable.changed.connect(lambda checked, widget=checkbox: self._update_checkbox_text(widget, checked))
        form.addRow(self._field_label(label), self._described_control(checkbox, description))

    def _update_checkbox_text(self, checkbox: QtWidgets.QCheckBox, checked: bool) -> None:
        checkbox.setText("On" if checked else "Off")

    def _described_control(self, control: QtWidgets.QWidget, description: str) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        help_label = QtWidgets.QLabel(description)
        help_label.setObjectName("ParamHelp")
        help_label.setWordWrap(True)
        help_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(control)
        layout.addWidget(help_label)
        return holder

    def _metric_card(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        column: int,
        name: str,
        variable: TextValue,
        accent: str,
        column_span: int = 1,
    ) -> None:
        card = MetricCard(name, variable, self.colors, accent)
        grid.addWidget(card, row, column, 1, column_span)

    def _ghost_button(self, text: str, command) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setObjectName("GhostButton")
        button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        button.clicked.connect(command)
        button.setMinimumHeight(38)
        return button

    def _bind_line_edit(self, field: QtWidgets.QLineEdit, variable: TextValue) -> None:
        field.setText(variable.get())
        field.textChanged.connect(variable.set)
        variable.changed.connect(lambda value, widget=field: self._set_line_text(widget, value))

    def _style_combo_popup(self, view: QtWidgets.QAbstractItemView) -> None:
        view.setObjectName("ComboPopupView")
        view.setStyleSheet(
            f"""
            QListView#ComboPopupView {{
                background-color: {self.colors["field"]};
                border: 2px solid {self.colors["line"]};
                border-radius: 12px;
                padding: 8px;
                selection-background-color: {self.colors["pink"]};
                outline: none;
            }}
            QListView#ComboPopupView::item {{
                background: transparent;
                border: none;
                min-height: 28px;
                padding: 8px 12px;
            }}
            QListView#ComboPopupView::item:hover {{
                background-color: {self.colors["surface"]};
            }}
            QListView#ComboPopupView::item:selected {{
                background-color: {self.colors["pink"]};
                border: 1px solid {self.colors["pink_strong"]};
            }}
            """
        )

    def _bind_combo(self, combo: QtWidgets.QComboBox, variable: TextValue, values: list[str]) -> None:
        view = combo.view()
        if view:
            self._style_combo_popup(view)
            view.setContentsMargins(0, 0, 0, 0)
            popup = view.window()
            if popup:
                popup.setObjectName("ComboPopupContainer")
                popup.setStyleSheet("QFrame#ComboPopupContainer { background: transparent; border: none; }")
                if isinstance(popup, QtWidgets.QFrame):
                    popup.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
                    popup.setLineWidth(0)
                    popup.setMidLineWidth(0)
                popup.setContentsMargins(0, 0, 0, 0)
                layout = popup.layout()
                if layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(0)
                popup.setWindowFlags(
                    QtCore.Qt.WindowType.Popup
                    | QtCore.Qt.WindowType.FramelessWindowHint
                    | QtCore.Qt.WindowType.NoDropShadowWindowHint
                )
                popup.setAutoFillBackground(False)
                popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        combo.addItems(values)
        if variable.get() in values:
            combo.setCurrentText(variable.get())
        combo.currentTextChanged.connect(variable.set)
        variable.changed.connect(lambda value, widget=combo: self._set_combo_text(widget, value))

    def _bind_checkbox(self, checkbox: QtWidgets.QCheckBox, variable: BoolValue) -> None:
        checkbox.setChecked(variable.get())
        checkbox.toggled.connect(variable.set)
        variable.changed.connect(lambda value, widget=checkbox: self._set_checkbox_value(widget, value))

    def _set_line_text(self, field: QtWidgets.QLineEdit, value: str) -> None:
        if field.text() == value:
            return
        blocker = QtCore.QSignalBlocker(field)
        field.setText(value)
        del blocker

    def _set_combo_text(self, combo: QtWidgets.QComboBox, value: str) -> None:
        if combo.currentText() == value:
            return
        index = combo.findText(value)
        if index < 0:
            return
        blocker = QtCore.QSignalBlocker(combo)
        combo.setCurrentIndex(index)
        del blocker

    def _set_checkbox_value(self, checkbox: QtWidgets.QCheckBox, value: bool) -> None:
        if checkbox.isChecked() == value:
            return
        blocker = QtCore.QSignalBlocker(checkbox)
        checkbox.setChecked(value)
        del blocker

    def _replace_combo_items(self, combo: QtWidgets.QComboBox, variable: TextValue, values: list[str]) -> None:
        current = variable.get()
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        combo.addItems(values)
        if current in values:
            combo.setCurrentText(current)
        elif values:
            combo.setCurrentText(values[0])
        selected = combo.currentText()
        del blocker
        if selected != current:
            variable.set(selected)

    def _browse_path(self, variable: TextValue, kind: str) -> None:
        initial = self._initial_dir(variable.get())
        if kind == "directory":
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", initial)
        elif kind == "open_cvvoice":
            value, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open Voice Package",
                initial,
                "CosyVoice package (*.cvvoice);;All files (*.*)",
            )
        elif kind == "save_cvvoice":
            value, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Voice Package",
                variable.get() or str(Path(initial) / "voice.cvvoice"),
                "CosyVoice package (*.cvvoice);;All files (*.*)",
            )
        elif kind == "open_image":
            value, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open Portrait",
                initial,
                "Images (*.png *.jpg *.jpeg *.webp);;All files (*.*)",
            )
        elif kind == "save_wav":
            value, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save WAV",
                variable.get() or str(Path(initial) / "vc_streaming.wav"),
                "WAV audio (*.wav);;All files (*.*)",
            )
        elif kind == "save_csv":
            value, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save CSV",
                variable.get() or str(Path(initial) / "vc_report.csv"),
                "CSV (*.csv);;All files (*.*)",
            )
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open WAV",
                initial,
                "WAV audio (*.wav);;All files (*.*)",
            )
        if value:
            variable.set(value)

    def _initial_dir(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.is_dir():
            return str(path)
        if path.parent.exists():
            return str(path.parent)
        return str(Path.cwd())

    def _refresh_audio_devices(self) -> None:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as error:
            self._log(f"Audio device refresh failed: {error}")
            self.input_device_map = {"Default": None}
            self.output_device_map = {"Default": None}
        else:
            self.input_device_map = {"Default": None}
            self.output_device_map = {"Default": None}
            for index, device in enumerate(devices):
                hostapi = hostapis[device["hostapi"]]["name"] if hostapis else ""
                label = f"{index}: {device['name']} [{hostapi}]"
                if device.get("max_input_channels", 0) > 0:
                    self.input_device_map[label] = index
                if device.get("max_output_channels", 0) > 0:
                    self.output_device_map[label] = index
        if hasattr(self, "input_device_combo"):
            self._replace_combo_items(self.input_device_combo, self.input_device_var, list(self.input_device_map.keys()))
        if hasattr(self, "output_device_combo"):
            self._replace_combo_items(self.output_device_combo, self.output_device_var, list(self.output_device_map.keys()))

    def _add_reference_files(self) -> None:
        initial = self._initial_dir(self.model_dir_var.get())
        values, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add Reference WAV Files",
            initial,
            "WAV audio (*.wav);;All files (*.*)",
        )
        for value in values:
            self._append_reference_row(value, "1.0")
        self._update_reference_weights()

    def _add_reference_folder(self) -> None:
        initial = self._initial_dir(self.model_dir_var.get())
        value = QtWidgets.QFileDialog.getExistingDirectory(self, "Add Reference Folder", initial)
        if not value:
            return
        for path in sorted(Path(value).glob("*.wav")):
            self._append_reference_row(str(path), "1.0")
        self._update_reference_weights()

    def _append_reference_row(self, path: str, weight: str) -> None:
        if not path:
            return
        existing = {self.reference_table.item(row, 0).text() for row in range(self.reference_table.rowCount())}
        if path in existing:
            return
        row = self.reference_table.rowCount()
        self.reference_table.insertRow(row)
        path_item = QtWidgets.QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        weight_item = QtWidgets.QTableWidgetItem(weight)
        normalized_item = QtWidgets.QTableWidgetItem("-")
        normalized_item.setFlags(normalized_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.reference_table.setItem(row, 0, path_item)
        self.reference_table.setItem(row, 1, weight_item)
        self.reference_table.setItem(row, 2, normalized_item)

    def _remove_selected_references(self) -> None:
        rows = sorted({index.row() for index in self.reference_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.reference_table.removeRow(row)
        self._update_reference_weights()

    def _move_selected_reference(self, direction: int) -> None:
        rows = sorted({index.row() for index in self.reference_table.selectedIndexes()})
        if not rows:
            return
        if direction < 0:
            rows_iter = rows
        else:
            rows_iter = list(reversed(rows))
        for row in rows_iter:
            target = row + direction
            if target < 0 or target >= self.reference_table.rowCount():
                continue
            self._swap_reference_rows(row, target)
        self.reference_table.clearSelection()
        for row in [max(0, min(self.reference_table.rowCount() - 1, row + direction)) for row in rows]:
            self.reference_table.selectRow(row)
        self._update_reference_weights()

    def _swap_reference_rows(self, left: int, right: int) -> None:
        values_left = [self.reference_table.item(left, col).text() for col in range(3)]
        values_right = [self.reference_table.item(right, col).text() for col in range(3)]
        for col, value in enumerate(values_right):
            self.reference_table.item(left, col).setText(value)
        for col, value in enumerate(values_left):
            self.reference_table.item(right, col).setText(value)

    def _reference_paths_and_weights(self) -> tuple[list[str], list[float]]:
        paths = []
        weights = []
        manual_mode = self.package_fusion_mode_var.get() == "manual_weight"
        for row in range(self.reference_table.rowCount()):
            path_item = self.reference_table.item(row, 0)
            weight_item = self.reference_table.item(row, 1)
            path = path_item.text().strip() if path_item is not None else ""
            if not path:
                continue
            paths.append(path)
            if manual_mode:
                try:
                    weights.append(float((weight_item.text() if weight_item is not None else "1.0").strip()))
                except ValueError as error:
                    raise ValueError(f"Raw weight for row {row + 1} must be a number.") from error
            else:
                weights.append(1.0)
        return paths, weights

    def _update_reference_weights(self) -> None:
        if not hasattr(self, "reference_table"):
            return
        mode = self.package_fusion_mode_var.get()
        row_count = self.reference_table.rowCount()
        raw_weights = []
        for row in range(row_count):
            if mode == "equal_weight":
                raw = 1.0
            elif mode == "duration_weight":
                raw = self._reference_duration_seconds(row)
            else:
                item = self.reference_table.item(row, 1)
                try:
                    raw = float(item.text()) if item is not None else 1.0
                except ValueError:
                    raw = 0.0
            raw_weights.append(max(0.0, raw))
        total = sum(weight for weight in raw_weights if weight > 0)
        blocker = QtCore.QSignalBlocker(self.reference_table)
        try:
            for row, raw in enumerate(raw_weights):
                raw_item = self.reference_table.item(row, 1)
                normalized_item = self.reference_table.item(row, 2)
                if raw_item is None or normalized_item is None:
                    continue
                if mode != "manual_weight":
                    raw_item.setText(f"{raw:.6g}")
                    raw_item.setFlags(raw_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                else:
                    raw_item.setFlags(raw_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                normalized = raw / total if raw > 0 and total > 0 else 0.0
                suffix = " (masked)" if normalized <= 0.0 and row_count > 0 else ""
                normalized_item.setText(f"{normalized:.4f}{suffix}")
        finally:
            del blocker

    def _reference_duration_seconds(self, row: int) -> float:
        item = self.reference_table.item(row, 0)
        if item is None:
            return 0.0
        try:
            import soundfile as sf

            info = sf.info(item.text())
            if info.samplerate <= 0:
                return 0.0
            return max(0.0, float(info.frames) / float(info.samplerate))
        except Exception:
            return 0.0

    def _start_package_create(self) -> None:
        if self._is_running():
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return
        try:
            config = self._snapshot_model_config()
            references, manual_weights = self._reference_paths_and_weights()
            if not references:
                raise ValueError("At least one reference WAV is required.")
            output_path = self.package_output_var.get().strip()
            if not output_path:
                raise ValueError("Output .cvvoice path is required.")
            if not output_path.lower().endswith(".cvvoice"):
                output_path += ".cvvoice"
                self.package_output_var.set(output_path)
            options = self._package_options(manual_weights)
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Invalid package settings", str(error))
            return
        self._set_running("package", True)
        self.worker_thread = threading.Thread(
            target=self._package_worker,
            args=(config, references, output_path, options),
            daemon=True,
        )
        self.worker_thread.start()

    def _package_options(self, manual_weights: list[float]) -> dict:
        if any(weight < 0 for weight in manual_weights):
            raise ValueError("Manual raw weights must be non-negative.")
        branch_gamma = self._positive_float(self.package_branch_gamma_var, "Branch gamma")
        attention_temperature = self._positive_float(self.package_attention_temperature_var, "Attention temp")
        canonical_seconds = self._positive_float(self.package_canonical_seconds_var, "Canonical sec")
        soft_prompt_seconds = self._positive_float(self.package_soft_prompt_seconds_var, "Soft prompt sec")
        soft_prompt_steps = int(self._nonnegative_float(self.package_soft_prompt_steps_var, "Soft prompt steps"))
        soft_prompt_teacher = self.package_soft_prompt_teacher_var.get()
        if soft_prompt_teacher not in {"grouped_branch_attention", "init_only"}:
            raise ValueError("Soft teacher must be grouped_branch_attention or init_only.")
        soft_checkpointing = self.package_soft_prompt_checkpointing_var.get()
        if soft_checkpointing not in {"auto", "on", "off"}:
            raise ValueError("Soft checkpoint must be auto, on, or off.")
        soft_segments = int(self._positive_float(self.package_soft_prompt_segments_var, "Soft segments"))
        options = {
            "fusion_mode": self.package_fusion_mode_var.get(),
            "manual_weights": manual_weights,
            "branch_weight_gamma": branch_gamma,
            "attention_temperature": attention_temperature,
            "canonical_prompt_length_seconds": canonical_seconds,
            "enable_soft_prompt": self.package_soft_prompt_var.get(),
            "soft_prompt_seconds": soft_prompt_seconds,
            "soft_prompt_steps": soft_prompt_steps,
            "soft_prompt_teacher_mode": soft_prompt_teacher,
            "soft_prompt_activation_checkpointing": soft_checkpointing,
            "soft_prompt_checkpoint_segments": soft_segments,
            "display_name": self.package_display_name_var.get().strip(),
            "short_description": self.package_short_description_var.get().strip(),
            "long_description": self.package_long_description_edit.toPlainText(),
        }
        portrait = self.package_portrait_var.get().strip()
        if portrait:
            options["portrait_path"] = str(Path(portrait).expanduser())
        return options

    def _inspect_package_current(self) -> None:
        package_path = self.voice_package_var.get().strip()
        if not package_path:
            QtWidgets.QMessageBox.information(self, "No package", "Choose a .cvvoice package first.")
            return
        try:
            metadata = read_voice_package_metadata(package_path)
            compatibility = "unknown (model directory not set)"
            model_dir = self.model_dir_var.get().strip()
            if model_dir:
                try:
                    validate_model_compatibility(metadata, Path(model_dir).expanduser())
                    compatibility = "compatible"
                except Exception as error:
                    compatibility = f"incompatible: {error}"
            self.package_inspect_text.setPlainText(self._format_package_metadata(metadata, compatibility))
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Package inspection failed", str(error))

    def _format_package_metadata(self, metadata: dict, compatibility: str) -> str:
        lines = [
            f"name: {metadata.get('display_name') or metadata.get('package_id')}",
            f"package_id: {metadata.get('package_id')}",
            f"format_version: {metadata.get('format_version')}",
            f"model: {metadata.get('model_family')} / {metadata.get('model_dir_name')}",
            f"compatibility: {compatibility}",
            f"size: {self._format_bytes(int(metadata.get('package_bytes', 0)))}",
            f"prompt_seconds: {float(metadata.get('prompt_seconds', 0.0)):.3f}",
            f"reference_count: {metadata.get('reference_count')}",
            f"branch_count: {metadata.get('branch_count')}",
            f"feature_dtype: {metadata.get('feature_dtype')}",
            f"fusion_mode: {metadata.get('fusion_mode')}",
            f"prompt_fusion: {metadata.get('prompt_fusion_algorithm')}",
            f"tail_policy: {metadata.get('flow_token_tail_fusion_policy')}",
            f"branch_weight_gamma: {metadata.get('branch_weight_gamma')}",
            f"attention_temperature: {metadata.get('attention_temperature')}",
            f"source_position_policy: {metadata.get('source_position_policy')}",
            f"canonical_prompt_length_seconds: {metadata.get('canonical_prompt_length_seconds')}",
            f"tensor_sha256: {metadata.get('tensor_sha256')}",
        ]
        if metadata.get("prompt_fusion_algorithm") == "soft_prompt_v1":
            lines.extend(
                [
                    f"soft_prompt_version: {metadata.get('soft_prompt_version')}",
                    f"soft_prompt_seconds: {float(metadata.get('soft_prompt_seconds', 0.0)):.3f}",
                    f"soft_prompt_mel_frames: {metadata.get('soft_prompt_mel_frames')}",
                    f"soft_prompt_training_steps: {metadata.get('soft_prompt_training_steps')}",
                    f"soft_prompt_final_loss: {metadata.get('soft_prompt_final_loss')}",
                    f"soft_prompt_activation_checkpointing: {metadata.get('soft_prompt_activation_checkpointing')}",
                    f"soft_prompt_checkpoint_segments: {metadata.get('soft_prompt_checkpoint_segments')}",
                ]
            )
        if metadata.get("portrait_path"):
            lines.extend(
                [
                    f"portrait: {metadata.get('portrait_path')} ({metadata.get('portrait_mime_type')})",
                    f"portrait_size: {metadata.get('portrait_width')}x{metadata.get('portrait_height')}",
                ]
            )
        lines.append("")
        lines.append("sources:")
        for source in metadata.get("prompt_sources", []):
            lines.append(
                "  branch={branch} file={file} seconds={seconds:.3f} tokens={tokens} "
                "raw={raw:.4f} normalized={normalized:.4f} masked={masked} sha256={sha}".format(
                    branch=source.get("branch_index"),
                    file=source.get("path_basename"),
                    seconds=float(source.get("accepted_seconds", 0.0)),
                    tokens=source.get("token_frames"),
                    raw=float(source.get("fusion_weight_raw", 0.0)),
                    normalized=float(source.get("fusion_weight_normalized", 0.0)),
                    masked=source.get("is_masked"),
                    sha=source.get("file_sha256"),
                )
            )
        return "\n".join(lines)

    def _format_bytes(self, value: int) -> str:
        amount = float(value)
        for unit in ["B", "KiB", "MiB", "GiB"]:
            if amount < 1024.0 or unit == "GiB":
                return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
            amount /= 1024.0
        return f"{value} B"

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
