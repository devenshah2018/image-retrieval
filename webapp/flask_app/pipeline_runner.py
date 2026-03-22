"""Run ``train_quick.py`` then ``precompute.sh`` in one locked job; stream combined stdout."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Iterator, Optional

from .paths import get_webapp_root
from .precompute_runner import apply_default_session_paths
from .runner_lock import JOB_LOCK


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_train_then_precompute(
    epochs: int,
    batch_size: int,
    max_train_samples: int,
    num_per_class: int,
    *,
    weights_download_url: str = "",
) -> Iterator[str]:
    """Train, then precompute; yield SSE lines then one ``done`` payload."""
    root = get_webapp_root()
    train_script = root / "train_quick.py"
    precompute_script = root / "precompute.sh"
    if not train_script.is_file():
        yield _sse_data({"error": f"train_quick.py not found: {train_script}"})
        return
    if not precompute_script.is_file():
        yield _sse_data({"error": f"precompute.sh not found: {precompute_script}"})
        return

    if not JOB_LOCK.acquire(blocking=False):
        yield _sse_data(
            {
                "error": "Another job is running. Wait for it to finish or use one tab only.",
            }
        )
        return

    proc: Optional[subprocess.Popen] = None
    try:
        train_args = [
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--max-train-samples",
            str(max_train_samples),
        ]
        train_cmd = [sys.executable, "-u", str(train_script), *train_args]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FLASK_TRAIN_STREAM"] = "1"

        proc = subprocess.Popen(
            train_cmd,
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
        train_code = proc.wait()
        proc = None

        if train_code != 0:
            yield _sse_data({"done": True, "returncode": train_code, "phase": "train"})
            return

        yield _sse_data({"line": ""})
        yield _sse_data(
            {
                "line": "========== Index build (precompute) ==========",
            }
        )

        pc_cmd = ["bash", str(precompute_script), "--num-per-class", str(num_per_class)]
        env2 = os.environ.copy()
        env2["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            pc_cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env2,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            yield _sse_data({"line": line.rstrip("\r\n")})
        pc_code = proc.wait()
        proc = None

        if pc_code == 0:
            apply_default_session_paths()

        done: dict = {"done": True, "returncode": pc_code}
        if pc_code != 0:
            done["phase"] = "precompute"
        else:
            done["pipeline_complete"] = True
            if weights_download_url:
                done["weights_download"] = weights_download_url
        yield _sse_data(done)
    except Exception as e:
        yield _sse_data({"error": str(e)})
        if proc is not None and proc.poll() is None:
            proc.kill()
        yield _sse_data({"done": True, "returncode": -1, "phase": "error"})
    finally:
        JOB_LOCK.release()
