#!/bin/bash

set -euo pipefail

# macOS: PyTorch + FAISS can each link libomp; without this, Search can abort with OMP Error #15.
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"
streamlit run app.py
