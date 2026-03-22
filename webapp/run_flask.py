#!/usr/bin/env python3
"""
Run the dev server using an interpreter that can import faiss.

After `conda activate`, PATH sometimes still resolves `python` to pyenv first.
If CONDA_PREFIX is set (normal when a conda env is active), we use that env's
`bin/python` for the Flask subprocess so Search works.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _python_for_flask() -> str:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = os.path.join(prefix, "bin", "python")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return sys.executable


def _fail_flask_missing(py: str, webapp_root: str) -> None:
    print(
        "Flask is not installed for this interpreter:\n"
        f"  {py}\n\n"
        "Install it into the active env, then retry:\n"
        f"  pip install 'flask>=3.0'\n"
        "  # or:\n"
        f"  pip install -r {os.path.join(webapp_root, 'requirements.txt')}\n\n"
        "If `environment.yaml` was updated after you created the env, sync:\n"
        f"  conda env update -f {os.path.join(webapp_root, 'environment.yaml')}\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))  # webapp/
    os.chdir(root)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["PYTHONPATH"] = root
    os.environ["FLASK_APP"] = "flask_app:create_app"
    py = _python_for_flask()
    check = subprocess.run(
        [py, "-c", "import flask"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        _fail_flask_missing(py, root)
    cmd = [
        py,
        "-m",
        "flask",
        "run",
        "--host=127.0.0.1",
        "--port=5050",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
