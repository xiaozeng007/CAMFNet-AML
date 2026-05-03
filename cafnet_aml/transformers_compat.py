"""
Runtime compatibility patch for transformers 4.42.x with huggingface-hub>=1.0.

This keeps the local environment untouched and only patches version lookup
inside the current Python process so `transformers` import can proceed.
"""

from __future__ import annotations

import importlib.metadata
from typing import Callable


_PATCHED = False
_ORIG_VERSION_FN: Callable[[str], str] | None = None
_FAKE_HF_HUB_VERSION = "0.99.0"


def _major(version_str: str) -> int:
    try:
        return int(str(version_str).split(".", 1)[0])
    except Exception:
        return 0


def apply_transformers_hub_compat() -> bool:
    """
    Patch `importlib.metadata.version` to report a <1.0 huggingface-hub version
    to older transformers runtime checks.

    Returns:
        bool: True if patch is active (applied now or already applied).
    """
    global _PATCHED, _ORIG_VERSION_FN
    if _PATCHED:
        return True

    orig = importlib.metadata.version

    def patched_version(package: str) -> str:
        ver = orig(package)
        pkg = str(package).lower()
        if pkg in ("huggingface-hub", "huggingface_hub") and _major(ver) >= 1:
            return _FAKE_HF_HUB_VERSION
        return ver

    importlib.metadata.version = patched_version
    _ORIG_VERSION_FN = orig
    _PATCHED = True
    return True

