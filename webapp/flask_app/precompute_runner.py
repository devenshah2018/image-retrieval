"""Run ``webapp/precompute.sh`` and stream stdout for SSE (FAISS + metadata)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Iterator, Optional

from .paths import default_faiss_index_path, default_features_paths_json, get_webapp_root
from .runner_lock import JOB_LOCK


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _apply_default_paths_to_session() -> None:
    try:
        from flask import has_request_context, session

        if not has_request_context():
            return
        session["faiss_path"] = default_faiss_index_path()
        session["features_path"] = default_features_paths_json()
        session.modified = True
    except Exception:
        pass


def apply_default_session_paths() -> None:
    """Public hook: set session FAISS paths after a successful index build."""
    _apply_default_paths_to_session()


def stream_precompute(num_per_class: int) -> Iterator[str]:
    """Yield SSE chunks: {line: str}, {done, returncode}, or {error: str}."""
    root = get_webapp_root()
    script = root / "precompute.sh"
    if not script.is_file():
        yield _sse_data({"error": f"precompute.sh not found: {script}"})
        return

    if not JOB_LOCK.acquire(blocking=False):
        yield _sse_data(
            {
                "error": "Another job is running (training or index build). "
                "Wait for it to finish or use one tab only.",
            }
        )
        return

    cmd = ["bash", str(script), "--num-per-class", str(num_per_class)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            yield _sse_data({"line": line.rstrip("\r\n")})
        code = proc.wait()
        if code == 0:
            _apply_default_paths_to_session()
        yield _sse_data({"done": True, "returncode": code})
    except Exception as e:
        yield _sse_data({"error": str(e)})
        if proc is not None and proc.poll() is None:
            proc.kill()
        yield _sse_data({"done": True, "returncode": -1})
    finally:
        JOB_LOCK.release()
