"""VC-only inference helpers."""

from cosyvoice.vc.voice_package import (
    VoicePackage,
    VoicePackageCompatibilityError,
    VoicePackageError,
    VoicePackageValidationError,
    VoicePromptBranch,
    VoicePromptInputs,
    load_voice_package,
    read_voice_package_metadata,
    save_voice_package,
)

__all__ = [
    "VoicePackage",
    "VoicePackageCompatibilityError",
    "VoicePackageError",
    "VoicePackageValidationError",
    "VoicePromptBranch",
    "VoicePromptInputs",
    "load_voice_package",
    "read_voice_package_metadata",
    "save_voice_package",
]
