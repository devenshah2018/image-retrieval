import os

# macOS: set before torch/faiss/numpy load OpenMP (avoids OMP #15 + Abort trap on Search).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import streamlit as st
import torch
from PIL import Image

from device import torch_device
from src.model import ResNetTransferModel
from utils import (
    preprocess_image,
    extract_features,
    load_faiss_index,
    search_similar_images,
    display_results,
)
from utils.rclone_utils import (
    copyto_temp,
    entry_remote_path,
    find_listable_entries,
    generate_rc_credentials,
    list_remotes,
    lsjson,
    parent_remote_path,
    pick_free_port,
    rcd_exited_early,
    rcd_gui_url,
    rclone_binary,
    start_rcd_webgui,
    stop_rcd_process,
    validate_remote_name,
)

# File paths (default)
MODEL_DIR = "weights"
DATA_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DEFAULT_FEATURES_PATHS_FILE = os.path.join(DATA_DIR, "features_paths.json")

st.set_page_config(
    page_title="Image Retrieval",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_styles() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1120px; }
    h1.app-title { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
    p.app-sub { color: #64748b; margin-top: 0; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "rcd_proc": None,
        "rcd_port": 5572,
        "rcd_user": None,
        "rcd_pass": None,
        "rclone_remote": "",
        "rclone_cwd": "",
        "query_image": None,
        "query_label": "",
        "query_source": "",
        "remotes_list": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_rcd_process() -> None:
    proc = st.session_state.get("rcd_proc")
    if proc is not None and proc.poll() is not None:
        st.session_state.rcd_proc = None


@st.cache_resource
def load_model(model_path: str, num_classes: int = 101):
    """Load the trained model (cached once per process)."""
    device = torch_device()
    model = ResNetTransferModel(
        num_classes=num_classes, embedding_size=128, pretrained=False
    ).to(device)
    if not os.path.exists(model_path):
        return None, device
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception:
        return None, device
    model.eval()
    return model, device


def render_header() -> None:
    st.markdown(
        "<h1 class='app-title'>Image retrieval</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='app-sub'>Similar-image search with embeddings and FAISS. "
        "Upload a file or pick an image from an rclone remote.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_app_styles()
    init_session_state()
    sync_rcd_process()

    render_header()

    st.sidebar.title("Settings")
    faiss_index_path = st.sidebar.text_input(
        "FAISS index path", value=DEFAULT_FAISS_INDEX_PATH
    )
    features_paths_file = st.sidebar.text_input(
        "Features paths JSON", value=DEFAULT_FEATURES_PATHS_FILE
    )
    num_results = st.sidebar.slider("Number of results", 1, 10, 5)
    show_all_categories = st.sidebar.checkbox("Show all categories", value=False)

    st.sidebar.divider()
    st.sidebar.subheader("Rclone")
    if not rclone_binary():
        st.sidebar.warning(
            "rclone is not on PATH. Install it (e.g. `brew install rclone`) and restart."
        )
    else:
        if st.sidebar.button("Refresh remotes"):
            ok, remotes, err = list_remotes()
            if ok:
                st.session_state.remotes_list = remotes
            else:
                st.sidebar.error(err or "Could not list remotes")

        if not st.session_state.remotes_list:
            ok, remotes, _ = list_remotes()
            if ok:
                st.session_state.remotes_list = remotes

        c_start, c_stop = st.sidebar.columns(2)
        with c_start:
            if st.button("Start config server", help="Runs rclone rcd with web GUI"):
                stop_rcd_process(st.session_state.rcd_proc)
                st.session_state.rcd_proc = None
                port = pick_free_port(5572)
                st.session_state.rcd_port = port
                u, pw = generate_rc_credentials()
                st.session_state.rcd_user = u
                st.session_state.rcd_pass = pw
                with st.spinner("Starting rclone (first run may download the GUI)…"):
                    proc, err = start_rcd_webgui(port, u, pw)
                    if not proc:
                        st.sidebar.error(err or "Failed to start rclone rcd")
                    else:
                        st.session_state.rcd_proc = proc
                        died, reason = rcd_exited_early(proc, wait_s=2.0)
                        if died:
                            st.session_state.rcd_proc = None
                            stop_rcd_process(proc)
                            st.sidebar.error(reason)
                        else:
                            st.sidebar.success(f"Config server on port {port}")

        with c_stop:
            if st.button("Stop config server"):
                stop_rcd_process(st.session_state.rcd_proc)
                st.session_state.rcd_proc = None
                st.sidebar.info("Config server stopped.")

        rcd_alive = (
            st.session_state.rcd_proc is not None
            and st.session_state.rcd_proc.poll() is None
        )
        if rcd_alive:
            port = st.session_state.rcd_port
            url = rcd_gui_url(port)
            st.sidebar.caption(
                "Open **Configs** in the rclone window to add Google Drive or other remotes."
            )
            st.sidebar.markdown(
                f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
                f"Open rclone web GUI ↗</a>",
                unsafe_allow_html=True,
            )
            if st.session_state.rcd_user and st.session_state.rcd_pass:
                st.sidebar.code(
                    f"User: {st.session_state.rcd_user}\nPass: {st.session_state.rcd_pass}",
                    language=None,
                )

        remotes = st.session_state.remotes_list
        if remotes:
            idx = 0
            if st.session_state.rclone_remote in remotes:
                idx = remotes.index(st.session_state.rclone_remote)
            choice = st.sidebar.selectbox("Remote", remotes, index=idx)
            if "_last_remote_pick" not in st.session_state:
                st.session_state._last_remote_pick = None
            if choice != st.session_state._last_remote_pick:
                st.session_state.rclone_cwd = ""
                st.session_state._last_remote_pick = choice
            st.session_state.rclone_remote = choice

            cwd = st.session_state.rclone_cwd
            if st.sidebar.button("↑ Parent folder", key="rclone_parent"):
                st.session_state.rclone_cwd = parent_remote_path(cwd)

            if validate_remote_name(choice, remotes):
                ok, entries, err = lsjson(choice, cwd)
                if not ok:
                    st.sidebar.warning(err)
                else:
                    dirs, files = find_listable_entries(entries)
                    st.sidebar.caption(f"Path: {choice}:{cwd or '(root)'}")
                    dir_paths = [entry_remote_path(d) for d in dirs]
                    sub = st.sidebar.selectbox(
                        "Subfolder",
                        ["—"] + dir_paths,
                        format_func=lambda p: "—"
                        if p == "—"
                        else (p.split("/")[-1] or p),
                    )
                    if st.sidebar.button("Open folder", key="rclone_open_folder"):
                        if sub != "—":
                            st.session_state.rclone_cwd = sub

                    file_paths = [entry_remote_path(f) for f in files]
                    labels = [p.split("/")[-1] or p for p in file_paths]
                    pick = st.sidebar.selectbox(
                        "Image file",
                        ["—"] + file_paths,
                        format_func=lambda p: "—"
                        if p == "—"
                        else (p.split("/")[-1] or p),
                    )
                    if st.sidebar.button("Load image", key="rclone_load_image"):
                        if pick != "—":
                            ok2, tmp, err2 = copyto_temp(choice, pick, list_cwd=cwd)
                            if ok2 and tmp:
                                try:
                                    img = Image.open(tmp).convert("RGB")
                                    st.session_state.query_image = img
                                    st.session_state.query_label = f"{choice}:{pick}"
                                    st.session_state.query_source = "rclone"
                                except Exception as ex:
                                    st.sidebar.error(f"Could not open image: {ex}")
                                try:
                                    os.unlink(tmp)
                                except OSError:
                                    pass
                            else:
                                st.sidebar.error(err2 or "Download failed")
        else:
            st.sidebar.info(
                "No remotes yet. Start the config server and add one under **Configs**, "
                "then click **Refresh remotes**."
            )

    if show_all_categories and os.path.exists(features_paths_file):
        try:
            with open(features_paths_file, "r") as f:
                indexed_paths = json.load(f)
            categories = sorted(
                {item["category"] for item in indexed_paths if "category" in item}
            )
            st.sidebar.subheader("Categories")
            for i in range(0, len(categories), 2):
                row = categories[i : i + 2]
                cols = st.sidebar.columns(2)
                for j, cat in enumerate(row):
                    cols[j].markdown(f"• {cat}")
        except Exception as e:
            st.sidebar.warning(f"Could not load categories: {e}")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("Query image")
        uploaded_file = st.file_uploader(
            "Upload",
            type=["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"],
            help="Local file upload; overrides rclone selection when set.",
        )
        if uploaded_file is not None:
            st.session_state.query_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.query_label = uploaded_file.name
            st.session_state.query_source = "upload"
        else:
            if st.session_state.get("query_source") == "upload":
                st.session_state.query_image = None
                st.session_state.query_label = ""
                st.session_state.query_source = ""

        b1, b2 = st.columns(2)
        with b1:
            search_button = st.button("Search", type="primary", use_container_width=True)
        with b2:
            clear_button = st.button("Clear", use_container_width=True)

    with right:
        st.subheader("Preview")
        if st.session_state.query_image is not None:
            st.image(
                st.session_state.query_image,
                caption=st.session_state.query_label or "Query",
                use_container_width=True,
            )
        else:
            st.caption("Upload an image or choose one from a remote in the sidebar.")

    if clear_button:
        stop_rcd_process(st.session_state.get("rcd_proc"))
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if not search_button:
        st.markdown("---")
        return

    if st.session_state.query_image is None:
        st.warning("Choose a query image (upload or rclone) before searching.")
        st.markdown("---")
        return

    model, device = load_model(MODEL_PATH)
    if not model:
        st.error(f"Model not found at {MODEL_PATH}. Run precompute.sh / training first.")
        st.stop()

    st.sidebar.caption(f"Device: {device}")

    faiss_index = load_faiss_index(faiss_index_path)
    if faiss_index is None:
        st.error(
            f"FAISS index not found at {faiss_index_path}. Run precompute.sh first."
        )
        st.stop()

    indexed_paths = None
    if os.path.exists(features_paths_file):
        try:
            with open(features_paths_file, "r") as f:
                indexed_paths = json.load(f)
            st.sidebar.info(f"Index entries: {len(indexed_paths)}")
        except Exception as e:
            st.warning(f"Could not load features paths: {e}")
    else:
        st.warning(f"Missing {features_paths_file}. Results may be wrong.")

    if not indexed_paths:
        st.error("Required features paths file missing or empty.")
        st.stop()

    with st.spinner("Searching…"):
        q = st.session_state.query_image
        query_tensor = preprocess_image(q, device)
        query_feature = extract_features(model, query_tensor, device)
        similarities, indices = search_similar_images(
            query_feature, faiss_index, k=num_results
        )
        display_results(similarities, indices, indexed_paths)

    st.markdown("---")


if __name__ == "__main__":
    main()
