from __future__ import annotations

import io
import json
import struct
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


FORMAT_NAME = "vc_studio_voice_package"
FORMAT_VERSION = 1
APP_VERSION = "0.1.0"
MODEL_FAMILY = "cosyvoice3"
TOKENIZER_SAFE_SECONDS = 30.0
TOKEN_RATE = 25
TOKENIZER_SAMPLE_RATE = 16000
SAMPLE_RATE = 24000
TOKEN_MEL_RATIO = 2
SPEAKER_EMBEDDING_DIM = 192
MEL_BINS = 80
SOFT_PROMPT_ALGORITHM = "soft_prompt_v1"
SOFT_PROMPT_VERSION = 1
DEFAULT_SOFT_PROMPT_SECONDS = 15.0
DEFAULT_SOFT_PROMPT_DISTILL_LAYER = 6
DEFAULT_SOFT_PROMPT_CHECKPOINT_SEGMENTS = 3

MODEL_HASH_FILES = {
    "flow_hash": "flow.pt",
    "hift_hash": "hift.pt",
    "campplus_hash": "campplus.onnx",
    "speech_tokenizer_hash": "speech_tokenizer_v3.onnx",
}

REQUIRED_METADATA_KEYS = {
    "format_name",
    "format_version",
    "package_id",
    "created_at",
    "created_by",
    "app_version",
    "model_family",
    "model_dir_name",
    "flow_hash",
    "hift_hash",
    "campplus_hash",
    "speech_tokenizer_hash",
    "sample_rate",
    "tokenizer_sample_rate",
    "token_rate",
    "token_mel_ratio",
    "speaker_embedding_dim",
    "mel_bins",
    "feature_dtype",
    "branch_count",
    "prompt_seconds",
    "prompt_token_frames",
    "prompt_mel_frames",
    "fused_speaker_embedding_norm",
    "tensor_sha256",
    "reference_count",
    "total_reference_seconds",
    "accepted_reference_seconds",
    "prompt_sources",
    "fusion_mode",
    "speaker_embedding_fusion_algorithm",
    "prompt_fusion_algorithm",
    "experimental_prompt_fusion_algorithms",
    "fusion_weight_sum_raw",
    "fusion_weight_normalization",
    "attention_weight_zero_policy",
    "branch_weight_gamma",
    "attention_temperature",
    "single_speaker_package",
    "source_position_policy",
    "canonical_prompt_length_seconds",
    "canonical_prompt_length_mel_frames",
    "prompt_length_normalization_policy",
    "flow_token_tail_fusion_policy",
    "dominant_branch_tie_breaker",
}

REQUIRED_SOURCE_KEYS = {
    "source_index",
    "branch_index",
    "display_name",
    "path_basename",
    "file_sha256",
    "original_sample_rate",
    "duration_seconds",
    "accepted_seconds",
    "token_frames",
    "mel_frames",
    "embedding_norm",
    "fusion_weight_raw",
    "fusion_weight_normalized",
    "branch_weight_after_gamma",
    "is_masked",
}

SOFT_PROMPT_METADATA_KEYS = {
    "soft_prompt_version",
    "soft_prompt_mel_frames",
    "soft_prompt_seconds",
    "soft_prompt_init",
    "soft_prompt_training_steps",
    "soft_prompt_training_loss",
    "soft_prompt_teacher",
    "soft_prompt_distill_layer",
    "soft_speaker_embedding_init",
    "soft_speaker_embedding_trainable",
    "soft_prompt_activation_checkpointing",
    "soft_prompt_checkpoint_segments",
}

SOFT_PROMPT_TENSOR_KEYS = {
    "soft_prompt_mu",
    "soft_prompt_feat",
    "soft_speaker_embedding",
}


class VoicePackageError(ValueError):
    pass


class VoicePackageValidationError(VoicePackageError):
    pass


class VoicePackageCompatibilityError(VoicePackageError):
    pass


@dataclass(frozen=True)
class VoicePackage:
    metadata: dict[str, Any]
    tensors: dict[str, np.ndarray]
    package_path: str | None = None
    portrait_path: str | None = None


@dataclass(frozen=True)
class VoicePromptBranch:
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    weight_raw: float
    weight_normalized: float
    metadata: dict[str, Any]

    @property
    def is_masked(self) -> bool:
        return self.weight_normalized <= 0.0


@dataclass(frozen=True)
class VoiceSoftPrompt:
    prompt_mu: torch.Tensor
    prompt_feat: torch.Tensor
    speaker_embedding: torch.Tensor
    metadata: dict[str, Any]

    @property
    def mel_frames(self) -> int:
        return int(self.prompt_mu.shape[1])

    @property
    def seconds(self) -> float:
        return self.mel_frames / float(TOKEN_RATE * TOKEN_MEL_RATIO)


@dataclass(frozen=True)
class VoicePromptInputs:
    branches: list[VoicePromptBranch]
    fused_embedding: torch.Tensor
    metadata: dict[str, Any]
    package_path: str | None = None
    soft_prompt: VoiceSoftPrompt | None = None

    def active_branch_indices(self) -> list[int]:
        return [index for index, branch in enumerate(self.branches) if branch.weight_normalized > 0.0]

    def sharpened_weights(self) -> list[float]:
        gamma = float(self.metadata.get("branch_weight_gamma", 1.0))
        if gamma <= 0:
            raise VoicePackageValidationError("branch_weight_gamma must be greater than 0")
        raw = [
            0.0 if branch.weight_normalized <= 0 else float(branch.weight_normalized) ** gamma
            for branch in self.branches
        ]
        total = sum(raw)
        if total <= 0:
            raise VoicePackageValidationError("voice package has no active prompt branches")
        return [value / total for value in raw]

    def dominant_branch_index(self) -> int:
        weights = self.sharpened_weights()
        best_weight = max(weights)
        for index, weight in enumerate(weights):
            if weight == best_weight:
                return index
        return 0

    def dominant_branch(self) -> VoicePromptBranch:
        return self.branches[self.dominant_branch_index()]

    def has_soft_prompt(self) -> bool:
        return self.soft_prompt is not None and self.metadata.get("prompt_fusion_algorithm") == SOFT_PROMPT_ALGORITHM


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_package_id() -> str:
    return str(uuid.uuid4())


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).expanduser().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes(model_or_dir: Any) -> dict[str, str]:
    model_dir = model_dir_from(model_or_dir)
    hashes = {}
    for key, filename in MODEL_HASH_FILES.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing model compatibility file: {path}")
        hashes[key] = sha256_file(path)
    return hashes


def model_dir_from(model_or_dir: Any) -> Path:
    if hasattr(model_or_dir, "model_dir"):
        return Path(getattr(model_or_dir, "model_dir")).expanduser()
    return Path(model_or_dir).expanduser()


def model_compatibility_fields(model_or_dir: Any) -> dict[str, Any]:
    model_dir = model_dir_from(model_or_dir)
    fields: dict[str, Any] = {
        "model_family": MODEL_FAMILY,
        "model_dir_name": model_dir.name,
        "sample_rate": int(getattr(model_or_dir, "sample_rate", SAMPLE_RATE)),
        "tokenizer_sample_rate": TOKENIZER_SAMPLE_RATE,
        "token_rate": TOKEN_RATE,
        "token_mel_ratio": int(getattr(model_or_dir, "token_mel_ratio", TOKEN_MEL_RATIO)),
        "speaker_embedding_dim": SPEAKER_EMBEDDING_DIM,
        "mel_bins": MEL_BINS,
        "feature_dtype": "float32",
    }
    fields.update(model_hashes(model_dir))
    return fields


def validate_model_compatibility(metadata: Mapping[str, Any], model_or_dir: Any) -> None:
    expected = model_compatibility_fields(model_or_dir)
    hard_fields = [
        "model_family",
        "sample_rate",
        "tokenizer_sample_rate",
        "token_rate",
        "token_mel_ratio",
        "speaker_embedding_dim",
        "mel_bins",
        "feature_dtype",
        *MODEL_HASH_FILES.keys(),
    ]
    mismatches = []
    for key in hard_fields:
        if metadata.get(key) != expected.get(key):
            mismatches.append(f"{key}: package={metadata.get(key)!r} model={expected.get(key)!r}")
    if mismatches:
        raise VoicePackageCompatibilityError("incompatible voice package:\n" + "\n".join(mismatches))


def save_voice_package(
    output_path: str | Path,
    tensors: Mapping[str, np.ndarray | torch.Tensor],
    metadata: Mapping[str, Any],
    portrait_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".cvvoice":
        raise VoicePackageValidationError("voice package output path must use the .cvvoice extension")
    arrays = _normalize_tensor_arrays(tensors)
    npz_payload = _npz_bytes(arrays)
    package_metadata = dict(metadata)
    package_metadata["tensor_sha256"] = sha256_bytes(npz_payload)

    portrait_name = None
    portrait_payload = None
    if portrait_path:
        portrait_name, portrait_payload, portrait_fields = _load_portrait_payload(portrait_path)
        package_metadata.update(portrait_fields)

    validate_metadata(package_metadata)
    validate_tensors(arrays, package_metadata, tensor_payload=npz_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = json.dumps(package_metadata, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("metadata.json", metadata_payload)
        archive.writestr("tensors.npz", npz_payload)
        if portrait_name is not None and portrait_payload is not None:
            archive.writestr(portrait_name, portrait_payload)
    return package_metadata


def load_voice_package(
    package_path: str | Path,
    model: Any | None = None,
    device: torch.device | str | None = None,
) -> VoicePromptInputs:
    path = Path(package_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"voice package does not exist: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            if "metadata.json" not in names:
                raise VoicePackageValidationError("malformed voice package: missing metadata.json")
            if "tensors.npz" not in names:
                raise VoicePackageValidationError("malformed voice package: missing tensors.npz")
            try:
                metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            except json.JSONDecodeError as error:
                raise VoicePackageValidationError("malformed voice package: metadata.json is not valid JSON") from error
            tensor_payload = archive.read("tensors.npz")
    except zipfile.BadZipFile as error:
        raise VoicePackageValidationError("malformed voice package: not a valid compressed ZIP container") from error

    arrays = _load_npz_arrays(tensor_payload)
    validate_metadata(metadata)
    validate_tensors(arrays, metadata, tensor_payload=tensor_payload)
    if model is not None:
        validate_model_compatibility(metadata, model)

    target_device = device
    if target_device is None and model is not None and hasattr(model, "device"):
        target_device = getattr(model, "device")
    return tensors_to_prompt_inputs(arrays, metadata, package_path=str(path), device=target_device)


def read_voice_package_metadata(package_path: str | Path) -> dict[str, Any]:
    path = Path(package_path).expanduser()
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if "metadata.json" not in set(archive.namelist()):
                raise VoicePackageValidationError("malformed voice package: missing metadata.json")
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    except zipfile.BadZipFile as error:
        raise VoicePackageValidationError("malformed voice package: not a valid compressed ZIP container") from error
    validate_metadata(metadata)
    metadata = dict(metadata)
    metadata["package_bytes"] = path.stat().st_size
    metadata["package_sha256"] = sha256_file(path)
    return metadata


def tensors_to_prompt_inputs(
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    package_path: str | None = None,
    device: torch.device | str | None = None,
) -> VoicePromptInputs:
    branch_count = int(metadata["branch_count"])
    source_by_branch = {
        int(source["branch_index"]): dict(source)
        for source in metadata.get("prompt_sources", [])
        if "branch_index" in source
    }
    branches = []
    for index in range(branch_count):
        branch_metadata = source_by_branch.get(index, {"branch_index": index})
        prompt_token = torch.from_numpy(np.asarray(tensors[f"branch_{index}_prompt_token"], dtype=np.int32))
        prompt_feat = torch.from_numpy(np.asarray(tensors[f"branch_{index}_prompt_feat"], dtype=np.float32))
        embedding = torch.from_numpy(np.asarray(tensors[f"branch_{index}_speaker_embedding"], dtype=np.float32))
        if device is not None:
            prompt_token = prompt_token.to(device=device, dtype=torch.int32)
            prompt_feat = prompt_feat.to(device=device, dtype=torch.float32)
            embedding = embedding.to(device=device, dtype=torch.float32)
        branches.append(
            VoicePromptBranch(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                weight_raw=float(branch_metadata.get("fusion_weight_raw", 1.0)),
                weight_normalized=float(branch_metadata.get("fusion_weight_normalized", 1.0)),
                metadata=branch_metadata,
            )
        )
    fused_embedding = torch.from_numpy(np.asarray(tensors["fused_speaker_embedding"], dtype=np.float32))
    if device is not None:
        fused_embedding = fused_embedding.to(device=device, dtype=torch.float32)
    soft_prompt = None
    if _metadata_declares_soft_prompt(metadata) or SOFT_PROMPT_TENSOR_KEYS <= set(tensors.keys()):
        prompt_mu = torch.from_numpy(np.asarray(tensors["soft_prompt_mu"], dtype=np.float32))
        prompt_feat = torch.from_numpy(np.asarray(tensors["soft_prompt_feat"], dtype=np.float32))
        speaker_embedding = torch.from_numpy(np.asarray(tensors["soft_speaker_embedding"], dtype=np.float32))
        if device is not None:
            prompt_mu = prompt_mu.to(device=device, dtype=torch.float32)
            prompt_feat = prompt_feat.to(device=device, dtype=torch.float32)
            speaker_embedding = speaker_embedding.to(device=device, dtype=torch.float32)
        soft_prompt = VoiceSoftPrompt(
            prompt_mu=prompt_mu,
            prompt_feat=prompt_feat,
            speaker_embedding=speaker_embedding,
            metadata={key: metadata.get(key) for key in SOFT_PROMPT_METADATA_KEYS if key in metadata},
        )
    return VoicePromptInputs(
        branches=branches,
        fused_embedding=fused_embedding,
        metadata=dict(metadata),
        package_path=package_path,
        soft_prompt=soft_prompt,
    )


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise VoicePackageValidationError("malformed voice package: metadata must be a JSON object")
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata.keys()))
    if missing:
        raise VoicePackageValidationError("malformed voice package: missing metadata keys: " + ", ".join(missing))
    if metadata.get("format_name") != FORMAT_NAME:
        raise VoicePackageValidationError(f"unsupported voice package format: {metadata.get('format_name')!r}")
    if int(metadata.get("format_version")) != FORMAT_VERSION:
        raise VoicePackageValidationError(f"unsupported voice package format_version: {metadata.get('format_version')!r}")
    branch_count = _require_positive_int(metadata, "branch_count")
    if bool(metadata.get("single_speaker_package")) is not True:
        raise VoicePackageValidationError("voice package must declare single_speaker_package=true")
    if metadata.get("feature_dtype") != "float32":
        raise VoicePackageValidationError("voice package prompt features must use float32")
    if int(metadata.get("token_mel_ratio")) <= 0:
        raise VoicePackageValidationError("token_mel_ratio must be positive")
    prompt_sources = metadata.get("prompt_sources")
    if not isinstance(prompt_sources, list):
        raise VoicePackageValidationError("prompt_sources must be a list")
    branch_indices = set()
    for source in prompt_sources:
        if not isinstance(source, Mapping):
            raise VoicePackageValidationError("each prompt_sources entry must be an object")
        missing_source = sorted(REQUIRED_SOURCE_KEYS - set(source.keys()))
        if missing_source:
            raise VoicePackageValidationError(
                "malformed voice package: missing prompt source keys: " + ", ".join(missing_source)
            )
        branch_index = int(source["branch_index"])
        if branch_index < 0 or branch_index >= branch_count:
            raise VoicePackageValidationError(f"prompt source has invalid branch_index: {branch_index}")
        branch_indices.add(branch_index)
        if float(source["fusion_weight_raw"]) < 0:
            raise VoicePackageValidationError("prompt source fusion_weight_raw must be non-negative")
    expected = set(range(branch_count))
    if branch_indices != expected:
        raise VoicePackageValidationError(
            f"prompt_sources must contain exactly one entry for each branch; got {sorted(branch_indices)}"
        )
    if _metadata_declares_soft_prompt(metadata):
        missing_soft = sorted(SOFT_PROMPT_METADATA_KEYS - set(metadata.keys()))
        if missing_soft:
            raise VoicePackageValidationError(
                "malformed soft prompt package: missing metadata keys: " + ", ".join(missing_soft)
            )
        if int(metadata.get("soft_prompt_version")) != SOFT_PROMPT_VERSION:
            raise VoicePackageValidationError(
                f"unsupported soft_prompt_version: {metadata.get('soft_prompt_version')!r}"
            )
        _require_positive_int(metadata, "soft_prompt_mel_frames")
        _require_positive_int(metadata, "soft_prompt_distill_layer")
        _require_positive_int(metadata, "soft_prompt_checkpoint_segments")
        try:
            soft_prompt_training_steps = int(metadata.get("soft_prompt_training_steps", -1))
        except (TypeError, ValueError) as error:
            raise VoicePackageValidationError("soft_prompt_training_steps must be an integer") from error
        if soft_prompt_training_steps < 0:
            raise VoicePackageValidationError("soft_prompt_training_steps must be 0 or greater")
        if float(metadata.get("soft_prompt_seconds", 0.0)) <= 0:
            raise VoicePackageValidationError("soft_prompt_seconds must be greater than 0")
        if metadata.get("soft_prompt_training_loss") != "layer_hidden_distill_v1":
            raise VoicePackageValidationError("soft_prompt_training_loss must be layer_hidden_distill_v1")
        if metadata.get("soft_prompt_teacher") != "grouped_branch_attention_hidden_state":
            raise VoicePackageValidationError("soft_prompt_teacher must be grouped_branch_attention_hidden_state")
        if metadata.get("soft_speaker_embedding_init") != "weighted_fused_embedding":
            raise VoicePackageValidationError("soft_speaker_embedding_init must be weighted_fused_embedding")
        if bool(metadata.get("soft_speaker_embedding_trainable")) is not False:
            raise VoicePackageValidationError("soft_speaker_embedding_trainable must be false")
        if metadata.get("soft_prompt_activation_checkpointing") not in {"off", "auto", "on"}:
            raise VoicePackageValidationError("soft_prompt_activation_checkpointing must be off, auto, or on")


def validate_tensors(
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    tensor_payload: bytes | None = None,
) -> None:
    if tensor_payload is not None:
        actual_hash = sha256_bytes(tensor_payload)
        if metadata.get("tensor_sha256") != actual_hash:
            raise VoicePackageValidationError(
                f"tensors.npz sha256 mismatch: package={metadata.get('tensor_sha256')} actual={actual_hash}"
            )
    branch_count = int(metadata["branch_count"])
    embedding_dim = int(metadata.get("speaker_embedding_dim", SPEAKER_EMBEDDING_DIM))
    mel_bins = int(metadata.get("mel_bins", MEL_BINS))
    ratio = int(metadata.get("token_mel_ratio", TOKEN_MEL_RATIO))
    required = {"fused_speaker_embedding"}
    for index in range(branch_count):
        required.update(
            {
                f"branch_{index}_speaker_embedding",
                f"branch_{index}_prompt_token",
                f"branch_{index}_prompt_feat",
            }
        )
    missing = sorted(required - set(tensors.keys()))
    if missing:
        raise VoicePackageValidationError("malformed voice package: missing tensor keys: " + ", ".join(missing))

    has_soft_tensors = bool(SOFT_PROMPT_TENSOR_KEYS & set(tensors.keys()))
    if _metadata_declares_soft_prompt(metadata):
        missing_soft = sorted(SOFT_PROMPT_TENSOR_KEYS - set(tensors.keys()))
        if missing_soft:
            raise VoicePackageValidationError(
                "malformed soft prompt package: missing tensor keys: " + ", ".join(missing_soft)
            )
    elif has_soft_tensors:
        raise VoicePackageValidationError(
            "soft prompt tensors are present but prompt_fusion_algorithm is not soft_prompt_v1"
        )

    fused = np.asarray(tensors["fused_speaker_embedding"])
    _require_shape_dtype(fused, "fused_speaker_embedding", (1, embedding_dim), np.float32)
    for index in range(branch_count):
        embedding = np.asarray(tensors[f"branch_{index}_speaker_embedding"])
        token = np.asarray(tensors[f"branch_{index}_prompt_token"])
        feat = np.asarray(tensors[f"branch_{index}_prompt_feat"])
        _require_shape_dtype(embedding, f"branch_{index}_speaker_embedding", (1, embedding_dim), np.float32)
        if token.dtype != np.int32:
            raise VoicePackageValidationError(f"branch_{index}_prompt_token must use int32, got {token.dtype}")
        if token.ndim != 2 or token.shape[0] != 1 or token.shape[1] <= 0:
            raise VoicePackageValidationError(
                f"branch_{index}_prompt_token must have shape [1, T] with T > 0, got {token.shape}"
            )
        if feat.dtype != np.float32:
            raise VoicePackageValidationError(f"branch_{index}_prompt_feat must use float32, got {feat.dtype}")
        expected_feat_shape = (1, token.shape[1] * ratio, mel_bins)
        if feat.shape != expected_feat_shape:
            raise VoicePackageValidationError(
                f"branch_{index}_prompt_feat must have shape {expected_feat_shape}, got {feat.shape}"
            )

    if _metadata_declares_soft_prompt(metadata):
        soft_frames = int(metadata["soft_prompt_mel_frames"])
        soft_mu = np.asarray(tensors["soft_prompt_mu"])
        soft_feat = np.asarray(tensors["soft_prompt_feat"])
        soft_speaker = np.asarray(tensors["soft_speaker_embedding"])
        _require_shape_dtype(soft_mu, "soft_prompt_mu", (1, soft_frames, mel_bins), np.float32)
        _require_shape_dtype(soft_feat, "soft_prompt_feat", (1, soft_frames, mel_bins), np.float32)
        _require_shape_dtype(soft_speaker, "soft_speaker_embedding", (1, embedding_dim), np.float32)


def branch_tensor_keys(branch_index: int) -> tuple[str, str, str]:
    return (
        f"branch_{branch_index}_speaker_embedding",
        f"branch_{branch_index}_prompt_token",
        f"branch_{branch_index}_prompt_feat",
    )


def l2_normalize_array(array: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, eps)


def sharpen_weights(weights: list[float], gamma: float) -> list[float]:
    if gamma <= 0:
        raise VoicePackageValidationError("branch_weight_gamma must be greater than 0")
    sharpened = [0.0 if weight <= 0 else float(weight) ** gamma for weight in weights]
    total = sum(sharpened)
    if total <= 0:
        raise VoicePackageValidationError("at least one branch weight must be positive")
    return [value / total for value in sharpened]


def _metadata_declares_soft_prompt(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("prompt_fusion_algorithm") == SOFT_PROMPT_ALGORITHM


def _normalize_tensor_arrays(tensors: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    return arrays


def _npz_bytes(tensors: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **dict(tensors))
    return buffer.getvalue()


def _load_npz_arrays(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    except Exception as error:
        raise VoicePackageValidationError("malformed voice package: tensors.npz is not a valid NumPy archive") from error


def _require_positive_int(metadata: Mapping[str, Any], key: str) -> int:
    try:
        value = int(metadata[key])
    except (TypeError, ValueError) as error:
        raise VoicePackageValidationError(f"{key} must be an integer") from error
    if value <= 0:
        raise VoicePackageValidationError(f"{key} must be greater than 0")
    return value


def _require_shape_dtype(array: np.ndarray, key: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
    if array.dtype != dtype:
        raise VoicePackageValidationError(f"{key} must use {np.dtype(dtype).name}, got {array.dtype}")
    if array.shape != shape:
        raise VoicePackageValidationError(f"{key} must have shape {shape}, got {array.shape}")


def _load_portrait_payload(path: str | Path) -> tuple[str, bytes, dict[str, Any]]:
    portrait_path = Path(path).expanduser()
    suffix = portrait_path.suffix.lower()
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    if suffix not in mime_by_suffix:
        raise VoicePackageValidationError("portrait image must be PNG, JPG, JPEG, or WEBP")
    payload = portrait_path.read_bytes()
    width, height = _image_dimensions(payload, suffix)
    fields = {
        "portrait_path": f"portrait{suffix}",
        "portrait_mime_type": mime_by_suffix[suffix],
        "portrait_sha256": sha256_bytes(payload),
        "portrait_width": width,
        "portrait_height": height,
        "portrait_bytes": len(payload),
    }
    return f"portrait{suffix}", payload, fields


def _image_dimensions(payload: bytes, suffix: str) -> tuple[int | None, int | None]:
    if suffix == ".png" and payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        return struct.unpack(">II", payload[16:24])
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(payload)
    if suffix == ".webp":
        return _webp_dimensions(payload)
    return None, None


def _jpeg_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    if not payload.startswith(b"\xff\xd8"):
        return None, None
    index = 2
    while index + 9 < len(payload):
        if payload[index] != 0xFF:
            index += 1
            continue
        marker = payload[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(payload):
            break
        length = struct.unpack(">H", payload[index:index + 2])[0]
        if length < 2 or index + length > len(payload):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = struct.unpack(">H", payload[index + 3:index + 5])[0]
            width = struct.unpack(">H", payload[index + 5:index + 7])[0]
            return width, height
        index += length
    return None, None


def _webp_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 30 or not payload.startswith(b"RIFF") or payload[8:12] != b"WEBP":
        return None, None
    chunk = payload[12:16]
    if chunk == b"VP8X" and len(payload) >= 30:
        width = int.from_bytes(payload[24:27], "little") + 1
        height = int.from_bytes(payload[27:30], "little") + 1
        return width, height
    if chunk == b"VP8 " and len(payload) >= 30:
        width = struct.unpack("<H", payload[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", payload[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(payload) >= 25:
        bits = int.from_bytes(payload[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None, None
