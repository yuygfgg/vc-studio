from __future__ import annotations

import zipfile
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosyvoice.vc.voice_package import (
    APP_VERSION,
    FORMAT_NAME,
    FORMAT_VERSION,
    MEL_BINS,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
    TOKENIZER_SAMPLE_RATE,
    TOKEN_MEL_RATIO,
    TOKEN_RATE,
    VoicePackageCompatibilityError,
    VoicePackageValidationError,
    load_voice_package,
    model_compatibility_fields,
    save_voice_package,
    sharpen_weights,
    validate_metadata,
)


def test_voice_package_round_trip_preserves_long_description(tmp_path: Path) -> None:
    tensors = _sample_tensors(branch_count=1)
    metadata = _sample_metadata(tmp_path, branch_count=1, long_description="line 1\nline 2" * 100)
    output = tmp_path / "voice.cvvoice"

    final_metadata = save_voice_package(output, tensors, metadata)
    assert zipfile.is_zipfile(output)
    assert final_metadata["long_description"].startswith("line 1")

    loaded = load_voice_package(output, model=tmp_path / "model")
    assert loaded.metadata["format_name"] == FORMAT_NAME
    assert loaded.metadata["long_description"] == final_metadata["long_description"]
    assert len(loaded.branches) == 1
    assert loaded.branches[0].prompt_token.shape == (1, 3)
    assert loaded.branches[0].prompt_feat.shape == (1, 6, MEL_BINS)
    assert loaded.fused_embedding.shape == (1, SPEAKER_EMBEDDING_DIM)


def test_voice_package_portrait_payload_metadata(tmp_path: Path) -> None:
    tensors = _sample_tensors(branch_count=1)
    metadata = _sample_metadata(tmp_path, branch_count=1)
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (1).to_bytes(4, "big") + (2).to_bytes(4, "big"))
    output = tmp_path / "voice.cvvoice"

    final_metadata = save_voice_package(output, tensors, metadata, portrait_path=portrait)
    assert final_metadata["portrait_path"] == "portrait.png"
    assert final_metadata["portrait_mime_type"] == "image/png"
    assert final_metadata["portrait_width"] == 1
    assert final_metadata["portrait_height"] == 2

    with zipfile.ZipFile(output, "r") as archive:
        assert "portrait.png" in archive.namelist()


def test_metadata_validation_rejects_missing_required_key(tmp_path: Path) -> None:
    metadata = _sample_metadata(tmp_path, branch_count=1)
    metadata.pop("prompt_sources")
    with pytest.raises(VoicePackageValidationError, match="missing metadata keys"):
        validate_metadata(metadata)


def test_tensor_validation_rejects_wrong_dtype(tmp_path: Path) -> None:
    tensors = _sample_tensors(branch_count=1)
    tensors["branch_0_prompt_token"] = tensors["branch_0_prompt_token"].astype(np.int64)
    metadata = _sample_metadata(tmp_path, branch_count=1)

    with pytest.raises(VoicePackageValidationError, match="branch_0_prompt_token must use int32"):
        save_voice_package(tmp_path / "voice.cvvoice", tensors, metadata)


def test_model_hash_mismatch_is_hard_error(tmp_path: Path) -> None:
    tensors = _sample_tensors(branch_count=1)
    metadata = _sample_metadata(tmp_path, branch_count=1)
    output = tmp_path / "voice.cvvoice"
    save_voice_package(output, tensors, metadata)

    (tmp_path / "model" / "flow.pt").write_bytes(b"different")
    with pytest.raises(VoicePackageCompatibilityError, match="flow_hash"):
        load_voice_package(output, model=tmp_path / "model")


def test_sharpen_weights_masks_zero_weight_branch() -> None:
    weights = sharpen_weights([0.25, 0.75, 0.0], gamma=2.0)
    assert weights[2] == 0.0
    assert pytest.approx(sum(weights), abs=1e-6) == 1.0
    assert weights[1] > weights[0]


def _sample_tensors(branch_count: int) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for index in range(branch_count):
        tensors[f"branch_{index}_speaker_embedding"] = _unit_embedding(index)
        tensors[f"branch_{index}_prompt_token"] = np.array([[1, 2, 3]], dtype=np.int32)
        tensors[f"branch_{index}_prompt_feat"] = np.ones((1, 6, MEL_BINS), dtype=np.float32) * (index + 1)
    tensors["fused_speaker_embedding"] = _unit_embedding(0)
    return tensors


def _sample_metadata(tmp_path: Path, branch_count: int, **extra) -> dict:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    for name in ["flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx"]:
        path = model_dir / name
        if not path.exists():
            path.write_bytes(name.encode("utf-8"))
    prompt_sources = []
    for index in range(branch_count):
        prompt_sources.append(
            {
                "source_index": index,
                "branch_index": index,
                "display_name": f"ref {index}",
                "path_basename": f"ref_{index}.wav",
                "file_sha256": "0" * 64,
                "original_sample_rate": TOKENIZER_SAMPLE_RATE,
                "duration_seconds": 0.12,
                "accepted_seconds": 3 / TOKEN_RATE,
                "token_frames": 3,
                "mel_frames": 6,
                "embedding_norm": 1.0,
                "fusion_weight_raw": 1.0,
                "fusion_weight_normalized": 1.0 / branch_count,
                "branch_weight_after_gamma": 1.0 / branch_count,
                "is_masked": False,
            }
        )
    metadata = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "package_id": "test-package",
        "created_at": "2026-01-01T00:00:00+00:00",
        "created_by": "pytest",
        "app_version": APP_VERSION,
        **model_compatibility_fields(model_dir),
        "branch_count": branch_count,
        "prompt_seconds": branch_count * 3 / TOKEN_RATE,
        "prompt_token_frames": branch_count * 3,
        "prompt_mel_frames": branch_count * 6,
        "fused_speaker_embedding_norm": 1.0,
        "tensor_sha256": "",
        "reference_count": branch_count,
        "total_reference_seconds": branch_count * 0.12,
        "accepted_reference_seconds": branch_count * 3 / TOKEN_RATE,
        "prompt_sources": prompt_sources,
        "fusion_mode": "equal_weight",
        "speaker_embedding_fusion_algorithm": "l2_normalize_each_then_weighted_average_then_l2_normalize",
        "prompt_fusion_algorithm": "grouped_branch_attention_output_mix",
        "experimental_prompt_fusion_algorithms": [
            "concat_branch_prompt_kv_with_attention_logit_bias",
            "concat_branch_prompt_kv_with_value_scaling",
        ],
        "fusion_weight_sum_raw": float(branch_count),
        "fusion_weight_normalization": "divide_by_sum_of_positive_raw_weights",
        "attention_weight_zero_policy": "mask_branch",
        "branch_weight_gamma": 1.0,
        "attention_temperature": 1.0,
        "single_speaker_package": True,
        "source_position_policy": "canonical_prompt_length",
        "canonical_prompt_length_seconds": 10.0,
        "canonical_prompt_length_mel_frames": 500,
        "prompt_length_normalization_policy": "reject_over_limit_until_vad_segmentation",
        "flow_token_tail_fusion_policy": "dominant_branch",
        "dominant_branch_tie_breaker": "lowest_branch_index",
        "sample_rate": SAMPLE_RATE,
        "tokenizer_sample_rate": TOKENIZER_SAMPLE_RATE,
        "token_rate": TOKEN_RATE,
        "token_mel_ratio": TOKEN_MEL_RATIO,
        "speaker_embedding_dim": SPEAKER_EMBEDDING_DIM,
        "mel_bins": MEL_BINS,
        "feature_dtype": "float32",
    }
    metadata.update(extra)
    return metadata


def _unit_embedding(seed: int) -> np.ndarray:
    embedding = np.zeros((1, SPEAKER_EMBEDDING_DIM), dtype=np.float32)
    embedding[0, seed % SPEAKER_EMBEDDING_DIM] = 1.0
    return embedding
