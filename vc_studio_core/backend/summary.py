from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import configure_environment

configure_environment()

from cosyvoice.vc.model import VCOnlyModel


def offline_summary_lines(
    model: VCOnlyModel,
    source_stats: dict,
    rows: list[dict],
    output: str | Path,
    output_sample_rate: int | None = None,
    postprocess_stats: dict[str, Any] | None = None,
) -> list[str]:
    source_tokens = source_stats["tokens"]
    source_seconds = source_tokens / 25.0
    infer_compute_seconds = sum(row["compute_seconds"] for row in rows)
    pipeline_compute_seconds = infer_compute_seconds + source_stats["tokenize_seconds"]
    token_wait_seconds = sum(row["token_wait_seconds"] for row in rows)
    wall_pipeline_seconds = rows[-1]["wall_end_seconds"] if rows else source_stats["wall_seconds"]
    max_chunk_seconds = max((row["compute_seconds"] for row in rows), default=0.0)
    avg_infer_rtf = infer_compute_seconds / source_seconds if source_seconds > 0 else 0.0
    avg_pipeline_rtf = pipeline_compute_seconds / source_seconds if source_seconds > 0 else 0.0
    wall_rtf = wall_pipeline_seconds / source_seconds if source_seconds > 0 else 0.0
    output_sample_rate = output_sample_rate or model.sample_rate
    lines = [
        f"source_tokens={source_tokens} source_seconds={source_seconds:.3f}",
        f"source_tokenize_chunks={source_stats['chunks']}",
        f"source_tokenize_audio_seconds={source_stats['audio_seconds']:.3f}",
        f"source_tokenize_window_audio_seconds={source_stats['window_audio_seconds']:.3f}",
        f"source_tokenize_read_seconds={source_stats['read_seconds']:.3f}",
        f"source_tokenize_compute_seconds={source_stats['tokenize_seconds']:.3f}",
        f"source_tokenize_wall_seconds={source_stats['wall_seconds']:.3f}",
        f"source_vad_enabled={source_stats['vad_enabled']}",
        f"source_vad_speech_chunks={source_stats['vad_speech_chunks']}",
        f"source_vad_silence_chunks={source_stats['vad_silence_chunks']}",
        f"source_vad_silence_seconds={source_stats['vad_silence_seconds']:.3f}",
        f"source_vad_compute_seconds={source_stats['vad_compute_seconds']:.3f}",
        f"stream_token_wait_seconds={token_wait_seconds:.3f}",
        f"stream_infer_compute_seconds={infer_compute_seconds:.3f}",
        f"stream_pipeline_compute_seconds={pipeline_compute_seconds:.3f}",
        f"stream_wall_seconds={wall_pipeline_seconds:.3f}",
        f"avg_infer_compute_rtf={avg_infer_rtf:.3f}",
        f"avg_pipeline_compute_rtf={avg_pipeline_rtf:.3f}",
        f"wall_stream_rtf={wall_rtf:.3f}",
        f"max_chunk_compute_seconds={max_chunk_seconds:.3f}",
        f"chunks={len(rows)}",
        f"output_sample_rate={output_sample_rate}",
        f"output={Path(output).resolve()}",
    ]
    if postprocess_stats is not None:
        lines.extend(
            [
                f"lavasr_output_sample_rate={postprocess_stats['lavasr_output_sample_rate']}",
                f"lavasr_lowpass_hz={postprocess_stats['lavasr_lowpass_hz']:g}",
                f"lavasr_compute_seconds={postprocess_stats['lavasr_seconds']:.3f}",
                f"lavasr_rtf={postprocess_stats['lavasr_seconds'] / source_seconds if source_seconds > 0 else 0.0:.3f}",
            ]
        )
    return lines
