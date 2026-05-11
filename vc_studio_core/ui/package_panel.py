from __future__ import annotations

import threading
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from cosyvoice.vc.voice_package import read_voice_package_metadata, validate_model_compatibility


class PackagePanelMixin:
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
