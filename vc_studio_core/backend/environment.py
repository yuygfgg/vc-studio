from __future__ import annotations

import os
import warnings


def configure_environment() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HOME", "/tmp/cosyvoice_hf_cache")
    warnings.filterwarnings(
        "ignore",
        message=".*LoRACompatibleLinear.*PEFT backend.*",
        category=FutureWarning,
    )


configure_environment()
