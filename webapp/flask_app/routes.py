"""HTTP routes: parity with Streamlit app."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
    url_for,
)
from PIL import Image
from werkzeug.utils import secure_filename

from . import dataset_explorer as de
from . import rcd_registry
from . import pipeline_runner
from . import precompute_runner
from . import train_runner
from .paths import (
    default_faiss_index_path,
    default_features_paths_json,
    get_webapp_root,
)
from .rclone_import import get_rclone_utils
from .results import build_result_rows, guess_mime, resolve_result_file_path


def _ru():
    return get_rclone_utils()

bp = Blueprint("main", __name__)

_FLASK_WEBAPP_DIR = Path(__file__).resolve().parent.parent


def _load_notebooks() -> list[dict[str, str]]:
    """Read ``title`` / ``url`` pairs from ``notebooks.json`` (array, or object with a ``notebooks`` array)."""
    path = _FLASK_WEBAPP_DIR / "notebooks.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("notebooks"), list):
        items = raw["notebooks"]
    else:
        return []
    out: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        if title and url:
            out.append({"title": title, "url": url})
    return out


def _device_str() -> str:
    try:
        from device import torch_device_str

        return torch_device_str()
    except ImportError:
        return "cpu"


DEFAULT_FAISS_INDEX_PATH = default_faiss_index_path()
DEFAULT_FEATURES_PATHS_FILE = default_features_paths_json()


def _weights_file_path():
    """Default checkpoint from train_quick.py --output weights/model.pth."""
    return get_webapp_root() / "weights" / "model.pth"


def _parse_train_stream_params():
    """Query string ?epochs=&batch_size=&max_train_samples= for /train/stream."""
    try:
        epochs = int(request.args.get("epochs", 1))
        batch_size = int(request.args.get("batch_size", 64))
        max_train_samples = int(request.args.get("max_train_samples", 4000))
    except (TypeError, ValueError):
        return None
    epochs = max(1, min(500, epochs))
    batch_size = max(1, min(512, batch_size))
    max_train_samples = max(0, min(2_000_000, max_train_samples))
    return epochs, batch_size, max_train_samples


def _parse_precompute_stream_params():
    """Query string ?num_per_class= for /train/precompute/stream."""
    try:
        num_per_class = int(request.args.get("num_per_class", 20))
    except (TypeError, ValueError):
        return None
    num_per_class = max(1, min(500, num_per_class))
    return num_per_class


def _parse_pipeline_stream_params():
    """Query string for /train/pipeline/stream: train + precompute args."""
    parsed = _parse_train_stream_params()
    if parsed is None:
        return None
    pc = _parse_precompute_stream_params()
    if pc is None:
        return None
    epochs, batch_size, max_train_samples = parsed
    return epochs, batch_size, max_train_samples, pc


def _sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def _upload_dir() -> str:
    d = os.path.join(current_app.config["UPLOAD_ROOT"], _sid())
    os.makedirs(d, exist_ok=True)
    return d


def _defaults() -> None:
    if "faiss_path" not in session:
        session["faiss_path"] = DEFAULT_FAISS_INDEX_PATH
    if "features_path" not in session:
        session["features_path"] = DEFAULT_FEATURES_PATHS_FILE
    if "num_results" not in session:
        session["num_results"] = 5
    if "show_categories" not in session:
        session["show_categories"] = False
    if "rclone_cwd" not in session:
        session["rclone_cwd"] = ""
    if "rclone_remote" not in session:
        session["rclone_remote"] = ""
    if "last_remote_pick" not in session:
        session["last_remote_pick"] = None
    if "remotes_list" not in session:
        session["remotes_list"] = []
    if "query_path" not in session:
        session["query_path"] = ""
    if "query_label" not in session:
        session["query_label"] = ""
    if "query_source" not in session:
        session["query_source"] = ""
    if "search_results" not in session:
        session["search_results"] = None


def _maybe_refresh_remotes() -> None:
    ru = _ru()
    if not ru.rclone_binary():
        return
    ok, remotes, _ = ru.list_remotes()
    if ok:
        session["remotes_list"] = remotes
        if not session.get("rclone_remote") and remotes:
            session["rclone_remote"] = remotes[0]
            session["last_remote_pick"] = remotes[0]
    session.modified = True


def _load_indexed_paths():
    fp = session.get("features_path", DEFAULT_FEATURES_PATHS_FILE)
    if not fp or not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _categories_for_sidebar():
    fp = session.get("features_path", DEFAULT_FEATURES_PATHS_FILE)
    if not session.get("show_categories") or not fp or not os.path.isfile(fp):
        return []
    try:
        with open(fp, "r") as f:
            data = json.load(f)
        return sorted(
            {item["category"] for item in data if "category" in item}
        )
    except Exception:
        return []


def _remote_path_parts(cwd: str | None) -> list:
    """Segments of the current rclone path (for breadcrumb UI)."""
    if not cwd or not str(cwd).strip():
        return []
    return [p for p in str(cwd).strip("/").split("/") if p]


def _rclone_browse():
    """Context for sidebar: dirs, files, cwd, remote, error."""
    ru = _ru()
    remotes = session.get("remotes_list") or []
    choice = session.get("rclone_remote") or ""
    cwd = session.get("rclone_cwd") or ""
    out = {
        "dirs": [],
        "files": [],
        "cwd": cwd,
        "remote": choice,
        "ls_error": None,
    }
    if not remotes or not choice or choice not in remotes:
        return out
    if not ru.validate_remote_name(choice, remotes):
        return out
    ok, entries, err = ru.lsjson(choice, cwd)
    if not ok:
        out["ls_error"] = err
        return out
    dirs, files = ru.find_listable_entries(entries)
    out["dirs"] = [ru.entry_remote_path(d) for d in dirs]
    out["files"] = [ru.entry_remote_path(f) for f in files]
    return out


def _sync_rcd_and_session():
    _defaults()
    _maybe_refresh_remotes()
    sid = _sid()
    rcd_registry.sync_proc(sid)
    return sid


def _index_count():
    fp = session.get("features_path", "")
    if not fp or not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r") as f:
            return len(json.load(f))
    except Exception:
        return None


def _redirect_after_settings():
    """POST /settings may include redirect_to=index|query; default Train."""
    dest = (request.form.get("redirect_to") or "").strip().lower()
    if dest in ("index", "query"):
        return redirect(url_for("main.index"))
    return redirect(url_for("main.train_page"))


def _redirect_after_rclone():
    """POST rclone actions may include redirect_to=train; default Query."""
    dest = (request.form.get("redirect_to") or "").strip().lower()
    if dest == "train":
        return redirect(url_for("main.train_page"))
    return redirect(url_for("main.index"))


@bp.route("/", methods=["GET"])
def index():
    sid = _sync_rcd_and_session()
    browse = _rclone_browse()
    browse_path_parts = _remote_path_parts(browse.get("cwd"))
    rclone_at_root = len(browse_path_parts) == 0
    rcd_state = rcd_registry.get_state(sid)
    return render_template(
        "index.html",
        page_id="query",
        search_results=session.get("search_results"),
        query_label=session.get("query_label") or "",
        has_query=bool(session.get("query_path")),
        rclone_available=bool(_ru().rclone_binary()),
        rcd_alive=rcd_registry.rcd_alive(sid),
        rcd_url=rcd_registry.gui_url_for_session(sid),
        rcd_user=rcd_state.get("user"),
        rcd_password=rcd_state.get("password"),
        remotes=session.get("remotes_list") or [],
        browse=browse,
        browse_path_parts=browse_path_parts,
        rclone_at_root=rclone_at_root,
    )


@bp.route("/train", methods=["GET"])
def train_page():
    _sync_rcd_and_session()
    categories = _categories_for_sidebar()
    webapp = get_webapp_root()
    train_script = webapp / "train_quick.py"
    precompute_script = webapp / "precompute.sh"
    wpath = _weights_file_path()
    tr = train_script.is_file()
    pc = precompute_script.is_file()
    return render_template(
        "train.html",
        page_id="train",
        train_script_exists=tr,
        precompute_exists=pc,
        pipeline_ready=tr and pc,
        webapp_root=str(webapp),
        weights_exists=wpath.is_file(),
        download_weights_url=url_for("main.download_weights"),
        pipeline_stream_url=url_for("main.train_pipeline_stream"),
        device=_device_str(),
        index_count=_index_count(),
        categories=categories,
    )


@bp.route("/dataset", methods=["GET"])
def dataset_page():
    """Browse webapp/caltech101/ (Caltech-101 layout); optional ?q= folder name search."""
    _defaults()
    _sid()
    webapp = get_webapp_root()
    root = de.caltech_root(webapp)
    rel = de.safe_rel_path(request.args.get("path", ""))
    q = request.args.get("q", "").strip()

    search_hits: list = []
    if q and root.is_dir():
        search_hits = de.search_folders(root, q)

    dirs: list = []
    files: list = []
    browse_error = None
    if root.is_dir() and not q:
        dirs, files, browse_error = de.list_folder(root, rel)

    crumbs = de.breadcrumb_parts(rel)
    return render_template(
        "dataset.html",
        page_id="dataset",
        dataset_root_exists=root.is_dir(),
        dataset_rel=rel,
        dataset_root_display=str(root),
        dirs=dirs,
        files=files,
        browse_error=browse_error,
        q=q,
        search_hits=search_hits,
        crumbs=crumbs,
    )


@bp.route("/notebooks", methods=["GET"])
def notebooks_page():
    return render_template(
        "notebooks.html",
        page_id="notebooks",
        notebooks=_load_notebooks(),
    )


@bp.route("/dataset/file/<path:rel_path>")
def dataset_file(rel_path):
    """Serve an image file under webapp/caltech101/."""
    root = de.caltech_root(get_webapp_root())
    full = de.resolve_under(root, rel_path)
    if not full or not full.is_file():
        abort(404)
    return send_file(str(full), mimetype=guess_mime(str(full)))


@bp.route("/download/weights")
def download_weights():
    """Serve webapp/weights/model.pth (train_quick default output)."""
    p = _weights_file_path()
    if not p.is_file():
        abort(404)
    return send_file(
        str(p),
        as_attachment=True,
        download_name="model.pth",
        mimetype="application/octet-stream",
    )


@bp.route("/train/stream", methods=["GET"])
def train_stream():
    """Server-Sent Events: lines from webapp/train_quick.py subprocess."""
    _defaults()
    _sid()
    parsed = _parse_train_stream_params()
    if parsed is None:

        def bad():
            yield f"data: {json.dumps({'error': 'Invalid epochs, batch_size, or max_train_samples'})}\n\n"

        return Response(
            stream_with_context(bad()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    epochs, batch_size, max_train_samples = parsed
    weights_url = url_for("main.download_weights")

    def generate():
        for chunk in train_runner.stream_train(
            epochs,
            batch_size,
            max_train_samples,
            weights_download_url=weights_url,
        ):
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/train/precompute/stream", methods=["GET"])
def precompute_stream():
    """Server-Sent Events: lines from webapp/precompute.sh subprocess."""
    _defaults()
    _sid()
    parsed = _parse_precompute_stream_params()
    if parsed is None:

        def bad():
            yield f"data: {json.dumps({'error': 'Invalid num_per_class'})}\n\n"

        return Response(
            stream_with_context(bad()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    num_per_class = parsed

    def generate():
        for chunk in precompute_runner.stream_precompute(num_per_class):
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/train/pipeline/stream", methods=["GET"])
def train_pipeline_stream():
    """SSE: train_quick.py then precompute.sh in one job."""
    _defaults()
    _sid()
    parsed = _parse_pipeline_stream_params()
    if parsed is None:

        def bad():
            yield f"data: {json.dumps({'error': 'Invalid pipeline parameters'})}\n\n"

        return Response(
            stream_with_context(bad()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    epochs, batch_size, max_train_samples, num_per_class = parsed
    weights_url = url_for("main.download_weights")

    def generate():
        for chunk in pipeline_runner.stream_train_then_precompute(
            epochs,
            batch_size,
            max_train_samples,
            num_per_class,
            weights_download_url=weights_url,
        ):
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/config", methods=["GET"])
def config_page():
    """Removed from nav; keep redirect for old bookmarks."""
    return redirect(url_for("main.train_page"))


@bp.route("/settings", methods=["POST"])
def settings():
    _defaults()
    session["faiss_path"] = request.form.get("faiss_path", DEFAULT_FAISS_INDEX_PATH).strip()
    session["features_path"] = request.form.get(
        "features_path", DEFAULT_FEATURES_PATHS_FILE
    ).strip()
    try:
        session["num_results"] = max(
            1, min(10, int(request.form.get("num_results", 5)))
        )
    except ValueError:
        session["num_results"] = 5
    session["show_categories"] = request.form.get("show_categories") == "on"
    session.modified = True
    flash("Settings updated.", "success")
    return _redirect_after_settings()


@bp.route("/upload", methods=["POST"])
def upload():
    _defaults()
    f = request.files.get("file")
    if not f or not f.filename:
        flash("No file selected.", "error")
        return redirect(url_for("main.index"))
    name = secure_filename(f.filename)
    if not name:
        flash("Invalid filename.", "error")
        return redirect(url_for("main.index"))
    ext = os.path.splitext(name)[1].lower()
    if ext not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    ):
        flash("Unsupported image type.", "error")
        return redirect(url_for("main.index"))
    dest = os.path.join(_upload_dir(), "query" + ext)
    f.save(dest)
    session["query_path"] = dest
    session["query_label"] = name
    session["query_source"] = "upload"
    session.modified = True
    flash("Image uploaded.", "success")
    return redirect(url_for("main.index"))


@bp.route("/search", methods=["POST"])
def search():
    try:
        from . import retrieval
    except ImportError as e:
        flash(
            "Search needs PyTorch and faiss in the same Python that runs Flask. "
            "If conda has them but you still see this, you may be using the wrong "
            "interpreter: run ./run_flask.sh (uses python -m flask) or "
            "`python -m flask run` instead of the bare `flask` command. "
            f"({e})",
            "error",
        )
        return redirect(url_for("main.index"))

    _defaults()
    qp = session.get("query_path")
    if not qp or not os.path.isfile(qp):
        flash("Choose a query image (upload or rclone) before searching.", "error")
        return redirect(url_for("main.index"))
    try:
        img = Image.open(qp).convert("RGB")
    except Exception as e:
        flash(f"Could not read query image: {e}", "error")
        return redirect(url_for("main.index"))

    faiss_path = session.get("faiss_path", DEFAULT_FAISS_INDEX_PATH)
    num_results = int(session.get("num_results", 5))
    sims, idx, err = retrieval.run_search(img, faiss_path, num_results)
    if err:
        flash(err, "error")
        return redirect(url_for("main.index"))
    indexed_paths = _load_indexed_paths()
    if not indexed_paths:
        flash("Features paths file missing or invalid.", "error")
        return redirect(url_for("main.index"))

    rows = build_result_rows(sims, idx, indexed_paths)
    session["search_results"] = rows
    session.modified = True
    flash("Search complete.", "success")
    return redirect(url_for("main.index"))


@bp.route("/new-query", methods=["POST"])
def new_query():
    """Clear search results so the upload-focused layout returns; keeps query image."""
    _defaults()
    session["search_results"] = None
    session.modified = True
    return redirect(url_for("main.index"))


@bp.route("/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid:
        rcd_registry.stop_for_session(sid)
        p = os.path.join(current_app.config["UPLOAD_ROOT"], sid)
        shutil.rmtree(p, ignore_errors=True)
    session.clear()
    session.modified = True
    flash("Session cleared.", "success")
    return redirect(url_for("main.index"))


@bp.route("/query-image")
def query_image():
    _defaults()
    qp = session.get("query_path")
    if not qp or not os.path.isfile(qp):
        abort(404)
    return send_file(qp, mimetype=guess_mime(qp))


@bp.route("/result-image/<int:faiss_idx>")
def result_image(faiss_idx: int):
    _defaults()
    data = _load_indexed_paths()
    if not data or faiss_idx < 0 or faiss_idx >= len(data):
        abort(404)
    path = resolve_result_file_path(faiss_idx, data)
    if not path:
        abort(404)
    return send_file(path, mimetype=guess_mime(path))


@bp.route("/rclone/refresh", methods=["POST"])
def rclone_refresh():
    _defaults()
    ru = _ru()
    if not ru.rclone_binary():
        flash("rclone not on PATH.", "error")
        return _redirect_after_rclone()
    ok, remotes, err = ru.list_remotes()
    if ok:
        session["remotes_list"] = remotes
        session.modified = True
        flash("Remotes refreshed.", "success")
    else:
        flash(err or "Could not list remotes.", "error")
    return _redirect_after_rclone()


@bp.route("/rclone/rcd/start", methods=["POST"])
def rclone_rcd_start():
    _defaults()
    sid = _sid()
    ok, msg, port = rcd_registry.start_for_session(sid)
    if ok:
        flash(f"Config server running on port {port}.", "success")
    else:
        flash(msg, "error")
    return _redirect_after_rclone()


@bp.route("/rclone/rcd/stop", methods=["POST"])
def rclone_rcd_stop():
    _defaults()
    rcd_registry.stop_for_session(_sid())
    flash("Config server stopped.", "success")
    return _redirect_after_rclone()


@bp.route("/rclone/set-remote", methods=["POST"])
def rclone_set_remote():
    _defaults()
    new = request.form.get("rclone_remote", "").strip()
    if new != session.get("rclone_remote"):
        session["rclone_cwd"] = ""
    session["rclone_remote"] = new
    session["last_remote_pick"] = new
    session.modified = True
    return _redirect_after_rclone()


@bp.route("/rclone/parent", methods=["POST"])
def rclone_parent():
    _defaults()
    cwd = session.get("rclone_cwd") or ""
    session["rclone_cwd"] = _ru().parent_remote_path(cwd)
    session.modified = True
    return _redirect_after_rclone()


@bp.route("/rclone/open-folder", methods=["POST"])
def rclone_open_folder():
    _defaults()
    sub = request.form.get("subfolder", "").strip()
    if sub and sub != "—":
        session["rclone_cwd"] = sub
    session.modified = True
    return _redirect_after_rclone()


@bp.route("/rclone/load-image", methods=["POST"])
def rclone_load_image():
    _defaults()
    ru = _ru()
    remotes = session.get("remotes_list") or []
    choice = session.get("rclone_remote", "").strip()
    pick = request.form.get("image_file", "").strip()
    if not pick or pick == "—":
        flash("Select an image file.", "error")
        return _redirect_after_rclone()
    if not ru.validate_remote_name(choice, remotes):
        flash("Invalid remote.", "error")
        return _redirect_after_rclone()
    ok, tmp, err = ru.copyto_temp(
        choice, pick, list_cwd=session.get("rclone_cwd") or ""
    )
    if not ok or not tmp:
        flash(err or "Download failed.", "error")
        return _redirect_after_rclone()
    ext = os.path.splitext(pick)[1] or ".jpg"
    dest = os.path.join(_upload_dir(), "query" + ext)
    shutil.copy2(tmp, dest)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    try:
        Image.open(dest).convert("RGB")
    except Exception as e:
        try:
            os.unlink(dest)
        except OSError:
            pass
        flash(f"Could not open image: {e}", "error")
        return _redirect_after_rclone()
    session["query_path"] = dest
    session["query_label"] = f"{choice}:{pick}"
    session["query_source"] = "rclone"
    session.modified = True
    flash("Image loaded from remote.", "success")
    return redirect(url_for("main.index"))
