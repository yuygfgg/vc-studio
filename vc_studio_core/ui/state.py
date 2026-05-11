from __future__ import annotations

from PyQt6 import QtCore


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
