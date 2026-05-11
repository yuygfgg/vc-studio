from __future__ import annotations

from PyQt6 import QtGui


class StyleMixin:
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
