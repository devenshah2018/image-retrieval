"""Run sibling `webapp/train_quick.py` as a subprocess and stream stdout for SSE."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Iterator, Optional

from .paths import get_webapp_root

_lock = threading.Lock()


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_train(
    epochs: int,
    batch_size: int,
    max_train_samples: int,
    *,
    weights_download_url: str = "",
) -> Iterator[str]:
    """Yield SSE chunks: {line: str}, {done, returncode, weights_download?}, or {error: str}."""
    root = get_webapp_root()
    script = root / "train_quick.py"
    if not script.is_file():
        yield _sse_data({"error": f"train_quick.py not found: {script}"})
        return

    if not _lock.acquire(blocking=False):
        yield _sse_data(
            {
                "error": "Training is already running. Wait for it to finish or open one tab only.",
            }
        )
        return

    args = [
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-train-samples",
        str(max_train_samples),
    ]
    cmd = [sys.executable, "-u", str(script), *args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FLASK_TRAIN_STREAM"] = "1"

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
        done: dict = {"done": True, "returncode": code}
        if code == 0 and weights_download_url:
            done["weights_download"] = weights_download_url
        yield _sse_data(done)
    except Exception as e:
        yield _sse_data({"error": str(e)})
        if proc is not None and proc.poll() is None:
            proc.kill()
        yield _sse_data({"done": True, "returncode": -1})
    finally:
        _lock.release()
