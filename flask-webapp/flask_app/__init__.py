"""Flask application factory (image retrieval + rclone)."""
from __future__ import annotations

import os
import sys

from flask import Flask

from .paths import get_webapp_root


def create_app(test_config: dict | None = None) -> Flask:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    flask_webapp_dir = os.path.dirname(pkg_dir)

    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=os.path.join(flask_webapp_dir, "templates"),
        static_folder=os.path.join(flask_webapp_dir, "static"),
    )
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-only-change-in-production"),
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        UPLOAD_ROOT=os.path.join(app.instance_path, "uploads"),
        WEBAPP_ROOT=str(get_webapp_root()),
    )

    if test_config:
        app.config.update(test_config)

    try:
        import faiss  # noqa: F401
        app.config["FAISS_IMPORT_OK"] = True
    except ImportError:
        app.config["FAISS_IMPORT_OK"] = False

    @app.context_processor
    def _inject_runtime():
        return {
            "faiss_import_ok": app.config.get("FAISS_IMPORT_OK", False),
            "flask_python_executable": sys.executable,
        }

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_ROOT"], exist_ok=True)

    from . import routes

    app.register_blueprint(routes.bp)

    return app
