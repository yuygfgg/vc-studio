from __future__ import annotations

import argparse
import importlib.util
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from vc_studio_backend import (
    BackendConfig,
    StreamSettings,
    VCStudioBackend,
)


def rounded_rect(canvas, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs) -> int:
    radius = min(radius, max(0, (x2 - x1) // 2), max(0, (y2 - y1) // 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=18, **kwargs)


class RoundedButton:
    def __init__(
        self,
        parent,
        text: str,
        command: Callable[[], None],
        *,
        width: int = 128,
        height: int = 38,
        radius: int = 16,
        fill: str,
        hover_fill: str,
        disabled_fill: str,
        foreground: str,
        background: str,
        font: tuple[str, int, str] = ("Helvetica", 11, "bold"),
    ):
        import tkinter as tk

        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=background,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.radius = radius
        self.fill = fill
        self.hover_fill = hover_fill
        self.disabled_fill = disabled_fill
        self.foreground = foreground
        self.background = background
        self.font = font
        self.state = "normal"
        self.hover = False
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Button-1>", self._on_click)
        self._draw()

    def grid(self, *args, **kwargs):
        return self.canvas.grid(*args, **kwargs)

    def pack(self, *args, **kwargs):
        return self.canvas.pack(*args, **kwargs)

    def configure(self, cnf=None, **kwargs) -> None:
        if cnf:
            kwargs.update(cnf)
        if "state" in kwargs:
            self.state = kwargs.pop("state")
            self.canvas.configure(cursor="" if self.state == "disabled" else "hand2")
            self._draw()
        if "text" in kwargs:
            self.text = kwargs.pop("text")
            self._draw()
        if kwargs:
            self.canvas.configure(**kwargs)

    config = configure

    def _draw(self) -> None:
        self.canvas.delete("all")
        fill = self.disabled_fill if self.state == "disabled" else self.hover_fill if self.hover else self.fill
        text_fill = "#f8faf8" if self.state != "disabled" else "#d5d3cb"
        rounded_rect(self.canvas, 1, 1, self.width - 1, self.height - 1, self.radius, fill=fill, outline="")
        self.canvas.create_text(
            self.width // 2,
            self.height // 2,
            text=self.text,
            fill=text_fill if self.foreground == "#ffffff" else self.foreground,
            font=self.font,
        )

    def _on_enter(self, event) -> None:
        self.hover = True
        self._draw()

    def _on_leave(self, event) -> None:
        self.hover = False
        self._draw()

    def _on_click(self, event) -> None:
        if self.state != "disabled":
            self.command()


class MetricCard:
    def __init__(
        self,
        parent,
        name: str,
        variable,
        *,
        colors: dict[str, str],
        width: int = 142,
        height: int = 78,
    ):
        import tkinter as tk

        self.variable = variable
        self.name = name
        self.colors = colors
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=colors["panel"],
            highlightthickness=0,
            bd=0,
        )
        self.variable.trace_add("write", lambda *_: self._draw())
        self.canvas.bind("<Configure>", self._on_configure)
        self._draw()

    def grid(self, *args, **kwargs):
        return self.canvas.grid(*args, **kwargs)

    def _on_configure(self, event) -> None:
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        rounded_rect(
            self.canvas,
            1,
            1,
            self.width - 1,
            self.height - 1,
            14,
            fill=self.colors["card"],
            outline=self.colors["line"],
            width=1,
        )
        self.canvas.create_text(
            16,
            18,
            text=self.name,
            fill=self.colors["muted"],
            anchor="w",
            font=("Helvetica", 10),
        )
        self.canvas.create_text(
            16,
            48,
            text=self.variable.get(),
            fill=self.colors["text"],
            anchor="w",
            font=("Helvetica", 19, "bold"),
        )


def launch_gui(args: argparse.Namespace) -> None:
    app = VCStudioApp(args)
    app.mainloop()


class VCStudioApp:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("CosyVoice VC Studio")
        self.root.geometry("1180x820")
        self.root.minsize(1040, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.active_mode: str | None = None
        self.offline_stop_event: threading.Event | None = None
        self.live_stop_event: threading.Event | None = None
        self.backend = VCStudioBackend()
        self.input_device_map: dict[str, int | None] = {"Default": None}
        self.output_device_map: dict[str, int | None] = {"Default": None}

        self._create_variables(args)
        self._configure_style()
        self._build_ui()
        self._refresh_audio_devices()
        self.root.after(100, self._poll_ui_queue)

    def mainloop(self) -> None:
        self.root.mainloop()

    def _create_variables(self, args: argparse.Namespace) -> None:
        tk = self.tk
        self.model_dir_var = tk.StringVar(value=args.model_dir)
        self.prompt_var = tk.StringVar(value=args.prompt)
        self.source_var = tk.StringVar(value=args.source)
        self.output_var = tk.StringVar(value=args.output)
        self.csv_var = tk.StringVar(value=args.csv)
        self.device_var = tk.StringVar(value=args.device)
        self.ort_provider_var = tk.StringVar(value=args.ort_provider)
        self.coreml_cache_var = tk.StringVar(value=args.coreml_cache_dir)
        self.chunk_sec_var = tk.StringVar(value=f"{args.chunk_sec:g}")
        self.tokenizer_chunk_sec_var = tk.StringVar(value=f"{args.tokenizer_chunk_sec:g}")
        self.tokenizer_left_context_sec_var = tk.StringVar(value=f"{args.tokenizer_left_context_sec:g}")
        self.tokenizer_right_context_sec_var = tk.StringVar(value=f"{args.tokenizer_right_context_sec:g}")
        self.history_sec_var = tk.StringVar(value=f"{args.history_sec:g}")
        self.mel_overlap_sec_var = tk.StringVar(value=f"{args.mel_overlap_sec:g}")
        self.delayed_commit_sec_var = tk.StringVar(value=f"{args.delayed_commit_sec:g}")
        self.audio_declick_ms_var = tk.StringVar(value=f"{args.audio_declick_ms:g}")
        self.audio_blend_ms_var = tk.StringVar(value=f"{args.audio_blend_ms:g}")
        self.vad_enabled_var = tk.BooleanVar(value=args.enable_vad)
        self.vad_threshold_var = tk.StringVar(value=f"{args.vad_threshold:g}")
        self.vad_min_speech_ms_var = tk.StringVar(value=f"{args.vad_min_speech_ms:g}")
        self.vad_min_silence_ms_var = tk.StringVar(value=f"{args.vad_min_silence_ms:g}")
        self.vad_speech_pad_ms_var = tk.StringVar(value=f"{args.vad_speech_pad_ms:g}")
        self.flow_context_var = tk.StringVar(value=args.flow_context)
        self.hift_mode_var = tk.StringVar(value=args.hift_mode)
        self.prompt_cache_var = tk.BooleanVar(value=not args.disable_prompt_kv_cache)
        self.history_cache_var = tk.BooleanVar(value=not args.disable_history_kv_cache)
        self.input_device_var = tk.StringVar(value="Default")
        self.output_device_var = tk.StringVar(value="Default")
        self.status_var = tk.StringVar(value="Ready")
        self.metric_chunk_var = tk.StringVar(value="-")
        self.metric_rtf_var = tk.StringVar(value="-")
        self.metric_lag_var = tk.StringVar(value="-")
        self.metric_buffer_var = tk.StringVar(value="-")
        self.metric_underflow_var = tk.StringVar(value="-")

    def _configure_style(self) -> None:
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        bg = "#f4f1ea"
        panel = "#fbf7ef"
        card = "#ffffff"
        text = "#252a31"
        muted = "#747b72"
        accent = "#2f7d75"
        accent_hover = "#398f86"
        danger = "#b85852"
        danger_hover = "#c8645e"
        line = "#ded6c9"
        field = "#fffdf8"
        tab = "#ebe4d7"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=text, fieldbackground=field, bordercolor=line)
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("Panel.TLabel", background=panel, foreground=text)
        style.configure("Card.TLabel", background=card, foreground=text)
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Helvetica", 22, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Helvetica", 12))
        style.configure("Metric.TLabel", background=card, foreground=text, font=("Helvetica", 18, "bold"))
        style.configure("MetricName.TLabel", background=card, foreground=muted, font=("Helvetica", 10))
        style.configure("TButton", padding=(12, 7), background="#e8dfd0", foreground=text, bordercolor=line)
        style.map("TButton", background=[("active", "#efe7da"), ("disabled", "#ded8ce")])
        style.configure("Accent.TButton", background=accent, foreground="#ffffff", bordercolor=accent)
        style.map("Accent.TButton", background=[("active", accent_hover), ("disabled", "#b9c7c2")])
        style.configure("Danger.TButton", background=danger, foreground="#ffffff", bordercolor=danger)
        style.map("Danger.TButton", background=[("active", danger_hover), ("disabled", "#d8c0bb")])
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), background=tab, foreground=muted)
        style.map("TNotebook.Tab", background=[("selected", card)], foreground=[("selected", text)])
        self.colors = {
            "bg": bg,
            "panel": panel,
            "card": card,
            "text": text,
            "muted": muted,
            "accent": accent,
            "accent_hover": accent_hover,
            "danger": danger,
            "danger_hover": danger_hover,
            "line": line,
            "field": field,
            "disabled": "#cfc7bb",
        }

    def _build_ui(self) -> None:
        ttk = self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="CosyVoice VC Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Realtime microphone conversion and offline benchmark in one control room.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Subtitle.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(root, padding=(18, 6, 18, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        left.columnconfigure(1, weight=1)
        self._build_model_panel(left)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._build_realtime_tab(notebook)
        self._build_offline_tab(notebook)
        self._build_parameters_tab(notebook)

        bottom = ttk.Frame(root, padding=(18, 0, 18, 16))
        bottom.grid(row=2, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)
        ttk.Label(bottom, text="Run Log", style="Subtitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log_text = self.tk.Text(
            bottom,
            height=9,
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["line"],
            highlightcolor=self.colors["accent"],
            padx=12,
            pady=10,
            wrap="word",
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")

    def _build_model_panel(self, parent) -> None:
        ttk = self.ttk
        row = 0
        ttk.Label(parent, text="Model", style="Panel.TLabel", font=("Helvetica", 14, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 12))
        row += 1
        row = self._path_row(parent, row, "Model dir", self.model_dir_var, "directory")
        row = self._path_row(parent, row, "Prompt wav", self.prompt_var, "open_wav")
        ttk.Label(parent, text="Torch device", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=self.device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=12).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1
        ttk.Label(parent, text="ORT provider", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=self.ort_provider_var, values=["auto", "cpu", "cuda", "coreml"], state="readonly", width=12).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1
        row = self._path_row(parent, row, "CoreML cache", self.coreml_cache_var, "directory", optional=True)
        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=14)
        row += 1
        ttk.Label(parent, text="Metrics", style="Panel.TLabel", font=("Helvetica", 14, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1
        metrics = ttk.Frame(parent, style="Panel.TFrame")
        metrics.grid(row=row, column=0, columnspan=3, sticky="ew")
        metrics.columnconfigure((0, 1), weight=1)
        self._metric_card(metrics, 0, 0, "Chunk", self.metric_chunk_var)
        self._metric_card(metrics, 0, 1, "RTF", self.metric_rtf_var)
        self._metric_card(metrics, 1, 0, "Lag", self.metric_lag_var)
        self._metric_card(metrics, 1, 1, "Buffer", self.metric_buffer_var)
        self._metric_card(metrics, 2, 0, "Underflows", self.metric_underflow_var)

    def _build_realtime_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure(0, weight=1)
        notebook.add(tab, text="Realtime")
        ttk.Label(tab, text="Audio I/O", font=("Helvetica", 15, "bold")).grid(row=0, column=0, sticky="w")
        device_frame = ttk.Frame(tab)
        device_frame.grid(row=1, column=0, sticky="ew", pady=(12, 16))
        device_frame.columnconfigure(1, weight=1)
        ttk.Label(device_frame, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, state="readonly")
        self.input_device_combo.grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(device_frame, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var, state="readonly")
        self.output_device_combo.grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(device_frame, text="Refresh", command=self._refresh_audio_devices).grid(row=0, column=2, rowspan=2, sticky="ns", padx=(10, 0), pady=5)

        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, sticky="w", pady=(4, 20))
        self.start_live_button = RoundedButton(
            buttons,
            "Start Live",
            self._start_realtime,
            width=132,
            fill=self.colors["accent"],
            hover_fill=self.colors["accent_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.start_live_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_live_button = RoundedButton(
            buttons,
            "Stop",
            self._stop_realtime,
            width=92,
            fill=self.colors["danger"],
            hover_fill=self.colors["danger_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.stop_live_button.configure(state="disabled")
        self.stop_live_button.grid(row=0, column=1)

        hint = (
            "Use headphones to avoid feeding generated audio back into the microphone. "
            "Lower chunk and overlap values reduce latency; larger context usually improves stability."
        )
        ttk.Label(tab, text=hint, style="Muted.TLabel", wraplength=680).grid(row=3, column=0, sticky="w")

    def _build_offline_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text="Offline Benchmark")
        ttk.Label(tab, text="Benchmark Job", font=("Helvetica", 15, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        row = 1
        row = self._path_row(tab, row, "Source wav", self.source_var, "open_wav")
        row = self._path_row(tab, row, "Output wav", self.output_var, "save_wav")
        row = self._path_row(tab, row, "CSV report", self.csv_var, "save_csv", optional=True)
        buttons = ttk.Frame(tab)
        buttons.grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 0))
        self.run_offline_button = RoundedButton(
            buttons,
            "Run Benchmark",
            self._start_offline,
            width=158,
            fill=self.colors["accent"],
            hover_fill=self.colors["accent_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.run_offline_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_offline_button = RoundedButton(
            buttons,
            "Stop",
            self._stop_offline,
            width=92,
            fill=self.colors["danger"],
            hover_fill=self.colors["danger_hover"],
            disabled_fill=self.colors["disabled"],
            foreground="#ffffff",
            background=self.colors["bg"],
        )
        self.stop_offline_button.configure(state="disabled")
        self.stop_offline_button.grid(row=0, column=1)

    def _build_parameters_tab(self, notebook) -> None:
        ttk = self.ttk
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure((0, 1), weight=1)
        notebook.add(tab, text="Parameters")

        left = ttk.Frame(tab, style="Card.TFrame", padding=14)
        right = ttk.Frame(tab, style="Card.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ttk.Label(left, text="Timing", style="Card.TLabel", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        timing = [
            ("Chunk sec", self.chunk_sec_var),
            ("Tokenizer chunk", self.tokenizer_chunk_sec_var),
            ("Tokenizer left ctx", self.tokenizer_left_context_sec_var),
            ("Tokenizer right ctx", self.tokenizer_right_context_sec_var),
            ("History sec", self.history_sec_var),
            ("Mel overlap sec", self.mel_overlap_sec_var),
            ("Delayed commit sec", self.delayed_commit_sec_var),
        ]
        for index, (label, var) in enumerate(timing, start=1):
            self._number_row(left, index, label, var)

        ttk.Label(right, text="Quality / Runtime", style="Card.TLabel", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._number_row(right, 1, "De-click ms", self.audio_declick_ms_var)
        self._number_row(right, 2, "Audio blend ms", self.audio_blend_ms_var)
        ttk.Checkbutton(right, text="Silero VAD gate", variable=self.vad_enabled_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 2))
        self._number_row(right, 4, "VAD threshold", self.vad_threshold_var)
        self._number_row(right, 5, "VAD min speech ms", self.vad_min_speech_ms_var)
        self._number_row(right, 6, "VAD min silence ms", self.vad_min_silence_ms_var)
        self._number_row(right, 7, "VAD speech pad ms", self.vad_speech_pad_ms_var)
        ttk.Label(right, text="Flow context", style="Card.TLabel").grid(row=8, column=0, sticky="w", pady=6)
        ttk.Combobox(right, textvariable=self.flow_context_var, values=["streaming", "window-full"], state="readonly").grid(row=8, column=1, sticky="ew", pady=6)
        ttk.Label(right, text="HiFT mode", style="Card.TLabel").grid(row=9, column=0, sticky="w", pady=6)
        ttk.Combobox(right, textvariable=self.hift_mode_var, values=["stateful", "window"], state="readonly").grid(row=9, column=1, sticky="ew", pady=6)
        ttk.Checkbutton(right, text="Prompt KV cache", variable=self.prompt_cache_var).grid(row=10, column=0, columnspan=2, sticky="w", pady=(12, 2))
        ttk.Checkbutton(right, text="History KV cache", variable=self.history_cache_var).grid(row=11, column=0, columnspan=2, sticky="w", pady=2)
        left.columnconfigure(1, weight=1)
        right.columnconfigure(1, weight=1)

    def _number_row(self, parent, row: int, label: str, variable) -> None:
        self.ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        self.ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", pady=6)

    def _path_row(self, parent, row: int, label: str, variable, kind: str, optional: bool = False) -> int:
        ttk = self.ttk
        style = "Panel.TLabel" if str(parent.cget("style")) == "Panel.TFrame" else "TLabel"
        ttk.Label(parent, text=label, style=style).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 8))
        ttk.Entry(parent, textvariable=variable, width=36).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_path(variable, kind)).grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=5)
        if optional:
            variable.set(variable.get())
        return row + 1

    def _metric_card(self, parent, row: int, column: int, name: str, variable) -> None:
        card = MetricCard(parent, name, variable, colors=self.colors)
        card.grid(row=row, column=column, sticky="ew", padx=4, pady=4)

    def _browse_path(self, variable, kind: str) -> None:
        if kind == "directory":
            value = self.filedialog.askdirectory(initialdir=self._initial_dir(variable.get()))
        elif kind == "save_wav":
            value = self.filedialog.asksaveasfilename(
                initialdir=self._initial_dir(variable.get()),
                defaultextension=".wav",
                filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            )
        elif kind == "save_csv":
            value = self.filedialog.asksaveasfilename(
                initialdir=self._initial_dir(variable.get()),
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
        else:
            value = self.filedialog.askopenfilename(
                initialdir=self._initial_dir(variable.get()),
                filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
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
        self.input_device_combo["values"] = list(self.input_device_map.keys())
        self.output_device_combo["values"] = list(self.output_device_map.keys())
        if self.input_device_var.get() not in self.input_device_map:
            self.input_device_var.set("Default")
        if self.output_device_var.get() not in self.output_device_map:
            self.output_device_var.set("Default")

    def _start_offline(self) -> None:
        if self._is_running():
            self.messagebox.showinfo("Busy", "A job is already running.")
            return
        try:
            config = self._snapshot_config(require_source=True)
        except ValueError as error:
            self.messagebox.showerror("Invalid settings", str(error))
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
            self.messagebox.showinfo("Busy", "A job is already running.")
            return
        if importlib.util.find_spec("sounddevice") is None:
            self.messagebox.showerror(
                "Missing dependency",
                "Realtime audio requires sounddevice. Install requirements.txt, then restart the GUI.",
            )
            return
        try:
            config = self._snapshot_config(require_source=False)
        except ValueError as error:
            self.messagebox.showerror("Invalid settings", str(error))
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
        prompt = self.prompt_var.get().strip()
        if not model_dir:
            raise ValueError("Model directory is required.")
        if not prompt:
            raise ValueError("Prompt wav is required.")
        if not Path(model_dir).expanduser().is_dir():
            raise ValueError("Model directory does not exist.")
        if not Path(prompt).expanduser().is_file():
            raise ValueError("Prompt wav does not exist.")
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
            prompt=str(Path(prompt).expanduser()),
            source=str(Path(source).expanduser()) if source else "",
            output=str(Path(output).expanduser()),
            csv=str(Path(csv_path).expanduser()) if csv_path else "",
            device=self.device_var.get(),
            ort_provider=self.ort_provider_var.get(),
            coreml_cache_dir=str(Path(coreml_cache).expanduser()) if coreml_cache else None,
            settings=settings,
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
        if vad_enabled:
            if importlib.util.find_spec("silero_vad") is None:
                raise ValueError(
                    "Silero VAD is enabled, but the optional silero-vad package is not installed. "
                    "Install requirements.txt or run `pip install silero-vad`."
                )
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
        )

    def _positive_float(self, variable, name: str) -> float:
        value = self._float(variable, name)
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")
        return value

    def _nonnegative_float(self, variable, name: str) -> float:
        value = self._float(variable, name)
        if value < 0:
            raise ValueError(f"{name} must be 0 or greater.")
        return value

    def _float_in_range(self, variable, name: str, minimum: float, maximum: float) -> float:
        value = self._float(variable, name)
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
        return value

    def _float(self, variable, name: str) -> float:
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
            elif kind == "finished":
                self._set_running(str(payload), False)
        self.root.after(100, self._poll_ui_queue)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _on_close(self) -> None:
        if self.live_stop_event is not None:
            self.live_stop_event.set()
        if self.offline_stop_event is not None:
            self.offline_stop_event.set()
        self.backend.shutdown()
        self.root.destroy()
