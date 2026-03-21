# Flask image retrieval UI

HTML/CSS UI for the same retrieval + rclone flow as the Streamlit app in [`../webapp/`](../webapp/). Code and assets live here; **models, FAISS index, dataset paths, and Python `src` / `utils`** stay in **`webapp/`**.

## Layout

```
image-retrieval/
  webapp/           # Streamlit app, training, weights, data/, utils/, src/
  flask-webapp/     # This Flask app (templates, static, flask_app/)
```

`flask_app/paths.py` resolves the sibling `webapp/` directory (override with `WEBAPP_ROOT`).

## Run (localhost)

Activate the same conda env as `webapp`, then start the server with **`run_flask.py`** (recommended). It re-invokes Flask with **`sys.executable`**, so the process that serves HTTP is always the same interpreter you used to launch the script (avoids a mismatched `flask` or `python` on `PATH`).

```bash
conda activate w7-image-retrieval
cd flask-webapp
python run_flask.py
```

Or: `./run_flask.sh` (calls `python run_flask.py`).

Confirm faiss is visible to that interpreter:

```bash
python -c "import faiss; print('ok')"
```

If that fails, install into **this** env (`mamba install -c conda-forge faiss-cpu` or `pip install faiss-cpu`). If you use the IDE “Run” button, **select the conda env’s Python** in the interpreter picker—otherwise Search may still fail while Streamlit works.

**Pyenv vs conda on `PATH`:** If `which python` still points at pyenv after `conda activate`, the banner may show a pyenv path. `run_flask.py` and `run_flask.sh` use **`$CONDA_PREFIX/bin/python`** when `CONDA_PREFIX` is set (active conda env), so the Flask process matches your env even when PATH order is wrong.

This sets `PYTHONPATH` to `../webapp` so `import src` / `import utils` work.

Open **http://127.0.0.1:5050**. The **Query** page (`/`) is for upload, preview, and search; **Train** (`/train`) sets index paths, runs `webapp/train_quick.py` (same interpreter as Flask) with UI-editable epochs / batch size / max train samples, streams logs + epoch progress, and offers **Download model.pth** after a successful run; **Dataset** (`/dataset`) browses **`webapp/caltech101/`** (e.g. `101_ObjectCategories/…`) with folder search and image thumbnails; **Notebooks** (`/notebooks`) lists links from editable **`flask-webapp/notebooks.json`**. **`/config`** redirects to **`/train`**. Weights are served at **`GET /download/weights`**.

## Production WSGI

From `flask-webapp/`:

```bash
export PYTHONPATH="$(cd .. && pwd)/webapp"
export FLASK_SECRET_KEY="your-secret"
gunicorn -w 1 -b 127.0.0.1:5050 "wsgi:app"
```

Use **one worker** so the in-process model cache is shared.

## Dependencies

Use the **same** environment as `webapp/` so **PyTorch**, **faiss-cpu**, and **Flask** are available:

```bash
mamba activate w7-image-retrieval   # after mamba env create -f webapp/environment.yaml
```

### `No module named flask`

`webapp/environment.yaml` lists Flask, but an older env may have been created before that line existed. Install into the active env:

```bash
conda activate w7-image-retrieval
pip install -r flask-webapp/requirements.txt
```

Or sync the whole env from the file (from repo root):

```bash
conda env update -f webapp/environment.yaml
```

If you run with a bare **pyenv** or system Python, install the full stack (e.g. `pip install faiss-cpu torch torchvision flask`); the project’s `webapp/environment.yaml` is the source of truth.
