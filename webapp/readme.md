# Workshop 7: Minimal Run + Final Assignment

This guide gets the **minimum working version** running first, then defines your final assignment.

## 1. Minimal Setup (Get It Running)

Install rclone: `brew install rclone`

Run from `webapp/`:

```bash
mamba env create -f environment.yaml
mamba activate w7-image-retrieval
```

**Device (PyTorch):** Training and retrieval use **`device.py`** — CUDA (NVIDIA) if available, else **MPS** on Apple Silicon (Metal), else CPU. Older code that only checked CUDA always showed CPU on Mac; that is fixed here. (FAISS search is still CPU unless you use a CUDA GPU build.)

Train a quick baseline checkpoint:

Run a quick setup training pass (for smoke test):

```bash
python train_quick.py --epochs 1 --batch-size 64 --max-train-samples 4000
```

Build FAISS index and metadata:

```bash
bash precompute.sh --num-per-class 20
```

Launch the **Streamlit** app:

```bash
cd ../flask-webapp
./run_flask.sh
```

### Flask UI (sibling folder)

The **Flask** HTML/CSS app lives next to `webapp/` in [`../flask-webapp/`](../flask-webapp/). It uses the same `weights/`, `data/`, `src/`, and `utils/` from `webapp/` via `PYTHONPATH`. The **Dataset** tab browses `caltech101/` under `webapp/` (Caltech-101 layout).

```bash
cd ../flask-webapp
./run_flask.sh
```

Details: [`flask-webapp/README.md`](../flask-webapp/README.md).

At this point, the Streamlit app should work with default paths:
- `weights/model.pth`
- `data/faiss_index.bin`
- `data/features_paths.json`

### Rclone (optional, localhost)

The sidebar can **start the official rclone web GUI** (`rclone rcd --rc-web-gui`) so you can add remotes (e.g. Google Drive) **in the browser** without using the terminal `rclone config` wizard as the primary path.

1. Install rclone on your machine (e.g. macOS: `brew install rclone`).
2. Run the app (`./run_app.sh` from `webapp/`, or `./run_flask.sh` from `flask-webapp/`), open the sidebar **Rclone** section.
3. Click **Start config server** (first launch may download the GUI bundle; wait for the success message).
4. Open **Open rclone web GUI ↗**, sign in with the **User / Pass** shown in the sidebar (localhost-only).
5. In the rclone UI, use **Configs** to add a remote (OAuth happens in that UI for Drive-type backends).
6. Back in the app, click **Refresh remotes**, pick a **Remote**, navigate with **Subfolder** / **Open folder**, choose an **Image file**, then **Load image**.
7. Use **Preview** and **Search** as with a local upload.

**Troubleshooting**

- **Port in use**: the app picks a free port starting at 5572; the link always uses the port shown after a successful start.
- **Config server exits immediately**: another process may be bound to the port, or rclone failed—try **Stop config server** and start again, or run `rclone rcd --rc-web-gui` manually in a terminal to see errors.
- **Security**: the rclone RC GUI is bound to `127.0.0.1` only—do not expose it on a public network without TLS and auth.

Quick CLI check (optional):

```bash
python scripts/smoke_rclone.py
```

### E2E checklist (manual)

1. App loads with theme and header; **Clear** resets state.
2. **Start config server** succeeds; **Open rclone web GUI** loads; login works with displayed credentials.
3. Add or confirm a remote in **Configs**; **Refresh remotes** lists it.
4. Browse folders, **Load image** sets **Preview**; **Search** returns similar images (requires model + FAISS artifacts from precompute).
5. **Stop config server** terminates the background `rclone` process.

## 2. Baseline Definition

Use a regular training run as the baseline (not the quick smoke test).

From `webapp/`:

```bash
python train_quick.py --epochs 5 --batch-size 64 --max-train-samples 0
```

Baseline metric to beat:
- Validation accuracy from this regular run (record your own exact value from stdout).

## 3. Final Assignment

Your goal is to produce **better-than-baseline performance** and present how you achieved it.

### Requirements
1. Beat your baseline validation accuracy.
2. Show clear experiments that explain what changed and why.
3. Demo the web app using interesting images and discuss retrieval quality.

### Suggested Experiment Ideas
1. Train longer (`--epochs` > 1).
2. Tune learning rate and batch size.
3. Compare augmentation choices.
4. Change embedding size or model architecture in `src/model.py`.
5. Increase/decrease images per class during precompute (`--num-per-class`).

## 4. What To Submit

1. Final checkpoint: `weights/model.pth`
2. Retrieval artifacts: `data/faiss_index.bin`, `data/features_paths.json`
3. Short experiment report with:
   - Baseline result
   - Experiment table (config, metric, outcome)
   - Best model result
   - Key lessons
4. Web app demo with interesting queries and retrieval outputs.

## 5. Presentation Checklist

1. Show baseline vs final metric.
2. Explain the top 2-3 changes that improved results.
3. Show at least 3 query examples in the web app.
4. Discuss one failure case and what you would try next.
