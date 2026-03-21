"""Load webapp/utils/rclone_utils.py without importing utils/__init__.py (avoids faiss at Flask startup)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

from .paths import get_webapp_root

# Unique name avoids collision with any third-party `rclone_utils` already in sys.modules.
_RCLONE_UTILS_MODULE_NAME = "image_retrieval_webapp_rclone_utils"

_module: ModuleType | None = None


def get_rclone_utils() -> ModuleType:
    global _module
    if _module is None:
        path = get_webapp_root() / "utils" / "rclone_utils.py"
        spec = importlib.util.spec_from_file_location(
            _RCLONE_UTILS_MODULE_NAME, path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load rclone_utils")
        _module = importlib.util.module_from_spec(spec)
        # Required so @dataclass can resolve cls.__module__ during exec_module (Python 3.8+).
        sys.modules[spec.name] = _module
        spec.loader.exec_module(_module)
    return _module
