# Workshop 7: Image retrieval (Flask UI + training)

The **web app** is a **Flask** UI (`flask_app/`, `templates/`, `static/`) in this folder. Training, weights, FAISS artifacts, and Python packages **`src/`** and **`utils/`** all live here.

## Minimal setup

Install rclone (optional, for remote images): `brew install rclone`

```bash
mamba env create -f environment.yaml
mamba activate w7-image-retrieval
```

**Device (PyTorch):** Training and retrieval use **`device.py`** — CUDA if available, else MPS on Apple Silicon, else CPU.

Train a quick checkpoint:

```bash
python train_quick.py --epochs 1 --batch-size 64 --max-train-samples 4000
```

Build FAISS index and metadata:

```bash
bash precompute.sh --num-per-class 20
```

### Run the Flask app

From **`webapp/`**:

```bash
./run_app.sh
# or: python run_flask.py
```

Open **http://127.0.0.1:5050**. **`run_flask.py`** sets `PYTHONPATH` to this directory and prefers **`$CONDA_PREFIX/bin/python`** when conda is active (avoids pyenv vs conda mismatches).

Confirm **faiss** imports:

```bash
python -c "import faiss; print('ok')"
```

**Pages:** **Query** (`/`) upload / preview / search; **Train** (`/train`) runs **`train_quick.py`** then **`precompute.sh`** in one streamed pipeline (epochs / batch / max samples / num-per-class); **Dataset** (`/dataset`) browses **`caltech101/`** (Caltech-101 layout); **Notebooks** (`/notebooks`) reads **`notebooks.json`**. **`/config`** redirects to **`/train`**. Weights: **`GET /download/weights`**.

### Rclone (optional)

1. Install rclone (`brew install rclone`).
2. Run the app, open **Query** → **Rclone**: **Start config server**, open the web GUI, add remotes.
3. **Refresh remotes**, pick a remote, **Load image**, then **Search**.

### Production WSGI

From **`webapp/`**:

```bash
export PYTHONPATH="$(pwd)"
export FLASK_SECRET_KEY="your-secret"
gunicorn -w 1 -b 127.0.0.1:5050 "wsgi:app"
```

Use **one worker** so the in-process model cache is shared.

### Pip only

If Flask is missing in an older env:

```bash
pip install -r requirements.txt
# or: conda env update -f environment.yaml
```

## Baseline & assignment

Regular training (not the smoke test):

```bash
python train_quick.py --epochs 5 --batch-size 64 --max-train-samples 0
```

Beat your baseline validation accuracy; document experiments; demo the web app with interesting queries.

## Submit

1. `weights/model.pth`
2. `data/faiss_index.bin`, `data/features_paths.json`
3. Short report: baseline, experiments, best result, lessons
4. Demo: queries and retrieval quality

## Presentation

1. Baseline vs final metric  
2. Top 2–3 changes that helped  
3. ≥3 query examples in the app  
4. One failure case and next steps  
