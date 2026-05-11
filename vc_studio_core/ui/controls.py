from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from .state import BoolValue, TextValue
from .widgets import MetricCard


class ControlMixin:
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
