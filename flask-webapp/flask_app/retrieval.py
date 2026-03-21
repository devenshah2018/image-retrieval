"""Cached model + search for Flask (single process)."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image

from device import torch_device
from src.model import ResNetTransferModel
from utils import (
    preprocess_image,
    extract_features,
    load_faiss_index,
    search_similar_images,
)

from .paths import get_webapp_root

MODEL_PATH = os.path.join(str(get_webapp_root()), "weights", "model.pth")

_model = None
_device = None


def get_model():
    """Load model once per process."""
    global _model, _device
    if _model is not None:
        return _model, _device
    _device = torch_device()
    model = ResNetTransferModel(
        num_classes=101, embedding_size=128, pretrained=False
    ).to(_device)
    if not os.path.exists(MODEL_PATH):
        _model = None
        return None, _device
    try:
        ckpt = torch.load(MODEL_PATH, map_location=_device)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        _model = None
        return None, _device
    model.eval()
    _model = model
    return _model, _device


def run_search(
    query_image: Image.Image,
    faiss_index_path: str,
    num_results: int,
):
    """Return (similarities, indices) numpy arrays or (None, None) on failure."""
    model, device = get_model()
    if model is None:
        return None, None, "Model not found. Run training / precompute first."
    faiss_index = load_faiss_index(faiss_index_path)
    if faiss_index is None:
        return None, None, f"FAISS index not found at {faiss_index_path}"
    q = query_image.convert("RGB")
    tensor = preprocess_image(q, device)
    feat = extract_features(model, tensor, device)
    sims, idx = search_similar_images(feat, faiss_index, k=num_results)
    return sims, idx, None
