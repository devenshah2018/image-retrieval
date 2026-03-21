"""Resolve the sibling `webapp/` directory (weights, utils, data, training code)."""
from __future__ import annotations

import os
import pathlib


def get_webapp_root() -> pathlib.Path:
    """Default: repo sibling `webapp/`. Override with env WEBAPP_ROOT."""
    override = os.environ.get("WEBAPP_ROOT")
    if override:
        return pathlib.Path(override).resolve()
    pkg = pathlib.Path(__file__).resolve().parent  # flask_app
    flask_webapp = pkg.parent  # flask-webapp
    repo_root = flask_webapp.parent  # repository root
    return (repo_root / "webapp").resolve()
