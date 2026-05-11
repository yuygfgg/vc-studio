from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from .widgets import CuteButton, KawaiiBackdrop


class ViewBuilderMixin:
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
