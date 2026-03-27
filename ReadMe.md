# Image Retrieval

A comprehensive **content-based image retrieval** system that learns visual embeddings, indexes a gallery with **FAISS**, and serves an interactive **Flask** web application for upload, remote image pick-up, similarity search, training, and dataset exploration. The stack supports efficient CPU similarity search, optional **rclone** integration for cloud storage, and reproducible training pipelines so you can go from raw images to ranked retrieval results with clear overlap between research notebooks and production-style workflows.

## Overview

This project implements a complete embedding-based retrieval workflow that:

- Trains a **ResNet**-backbone model for classification and **L2-normalized embeddings** suited to similarity search  
- Builds and queries a **FAISS** inner-product index over precomputed features and metadata paths  
- Exposes a **Flask** multi-page app: Query, Train, Dataset browser, and curated notebook links  
- Supports **optional rclone** remotes for choosing query images without local copies  
- Separates **Conda** (webapp + ML stack) from a root **venv** (Jupyter notebooks) to avoid conflicting interpreters  

## Features

**Deep embeddings & training:** Transfer learning with `src/model.py`, `train_quick.py`, and device-aware execution (`device.py`) on CUDA, Apple MPS, or CPU.  

**Vector search:** FAISS CPU index for cosine-style similarity when embeddings are normalized; metadata in JSON alongside the index.  

**Web application:** Session-based paths, live training logs, epoch progress, and checkpoint download; static UI under `webapp/templates/` and `webapp/static/`.  

**Data & exploration:** Local **Caltech-101**-style tree under `webapp/caltech101/` with folder search; the Train page runs **training then precompute** in one pipeline (streamed log).  

**Notebooks hub:** Links maintained in `webapp/notebooks.json` to repository notebooks for extraction, retrieval, and evaluation.  

## Documentation

Hands-on tutorials, experiments, and visualizations live under **`notebooks/`**, including:

- **Feature extraction** — embedding pipelines and representation learning  
- **Retrieval using classification** — tying classifiers to retrieval behavior  
- **ResNet retrieval** — transfer learning for image retrieval workflows  
- **Model evaluation** — metrics and comparisons for retrieval quality  

Open any `.ipynb` with the **root `venv`** active (see [Installation](#installation)); use **`pip install -r requirements.txt`** for notebook dependencies—**do not** activate the Conda `w7-image-retrieval` env when running Jupyter, to keep notebook and web stacks isolated.

For training, precompute, FAISS artifacts, and production **gunicorn** deployment, see **[`webapp/readme.md`](webapp/readme.md)**.

---

## System requirements

| Requirement | Notes |
|-------------|--------|
| **OS** | macOS or Linux (Windows with equivalent tooling). |
| **Python** | **3.10** (pinned in `webapp/environment.yaml` for the web stack). |
| **Conda or Mamba** | For the web application: PyTorch, FAISS, Flask. [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). |
| **Homebrew** (macOS) | Optional CLI helpers below. |

### Install system tools (macOS)

```bash
brew install rclone python@3.10
```

- **rclone** — Optional; remote image sources in the web app.  
- **python@3.10** — Optional if you use only Conda; useful for the root **`venv`** used with notebooks.

---

## Installation

Use **one** environment per task:

| Use case | Environment | Purpose |
|----------|-------------|---------|
| **Web application** (`webapp/`) | Conda **`w7-image-retrieval`** | PyTorch, FAISS, Flask, training, search. |
| **Notebooks** | Root **`.venv`** | Jupyter and analysis **without** Conda active. |

### 1. General environment (notebooks)

```bash
cd /path/to/image-retrieval
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Before Jupyter: **`conda deactivate`** so only `.venv` is active.

### 2. Web application environment (Conda)

```bash
cd webapp
conda env create -f environment.yaml
# or: mamba env create -f environment.yaml
```

Refresh if needed (from repo root, env active):

```bash
conda activate w7-image-retrieval
conda env update -f webapp/environment.yaml
```

---

## Usage

### Web application

```bash
cd webapp
conda activate w7-image-retrieval
./run_app.sh
```

Open **http://127.0.0.1:5050**. `./run_app.sh` prefers **`$CONDA_PREFIX/bin/python`** when Conda is active. Alternative: `python run_flask.py` from `webapp/`.

### Jupyter notebooks

```bash
cd /path/to/image-retrieval
conda deactivate
source .venv/bin/activate
jupyter lab
```

Work in **`notebooks/`** with the venv only.

---

## Acknowledgements

This project was developed as part of **The Build Fellowship** program by **OpenAvenue Ventures**, a transformative experience that provided the foundation for building real-world engineering solutions. Special thanks to **Kamalesh Kalirathinam**, **Paul Kefer**, and the **AI-Powered Image Retrieval** team for providing mentorship and support throughout the development process.
