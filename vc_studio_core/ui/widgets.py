from __future__ import annotations

import math

from PyQt6 import QtCore, QtGui, QtWidgets

from .state import TextValue


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
