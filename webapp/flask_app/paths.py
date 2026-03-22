"""Resolve the `webapp/` project root (weights, utils, data, training code)."""
from __future__ import annotations

import os
import pathlib


def get_webapp_root() -> pathlib.Path:
    """Directory containing ``flask_app/``, ``src/``, ``utils/``, etc. Override with ``WEBAPP_ROOT``."""
    override = os.environ.get("WEBAPP_ROOT")
    if override:
        return pathlib.Path(override).resolve()
    pkg = pathlib.Path(__file__).resolve().parent  # flask_app/
    return pkg.parent.resolve()  # webapp/


def default_faiss_index_path() -> str:
    """Default ``data/faiss_index.bin`` under the webapp root."""
    return str(get_webapp_root() / "data" / "faiss_index.bin")


def default_features_paths_json() -> str:
    """Default ``data/features_paths.json`` under the webapp root."""
    return str(get_webapp_root() / "data" / "features_paths.json")
