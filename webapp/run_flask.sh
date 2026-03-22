#!/usr/bin/env bash
set -euo pipefail

# macOS: PyTorch + FAISS can each link libomp; without this, Search can abort with OMP Error #15.
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"
# Prefer the active conda env's python when CONDA_PREFIX is set (avoids pyenv winning on PATH).
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  exec "$CONDA_PREFIX/bin/python" run_flask.py "$@"
fi
exec python run_flask.py "$@"
