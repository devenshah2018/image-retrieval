"""Build search result rows for HTML (parity with display_utils)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .paths import get_webapp_root


def resolve_local_image_path(img_path: str) -> Optional[str]:
    """
    Resolve a gallery path from features JSON to an absolute file path under ``webapp/``.

    JSON paths are often relative (e.g. ``./caltech101/101_ObjectCategories/...``) and only
    work with ``os.path.exists`` when the process cwd is ``webapp/``. Flask may run with cwd
    elsewhere, so we always anchor against ``get_webapp_root()``.
    """
    root = get_webapp_root().resolve()
    raw = (img_path or "").strip()
    if not raw:
        return None

    trials: List[Path] = []
    p = Path(raw)
    if p.is_absolute():
        trials.append(p)
    else:
        stripped = raw.lstrip("./")
        trials.extend(
            [
                root / raw,
                root / stripped,
                root / raw.replace("./", "", 1),
            ]
        )
        if "caltech101" in raw:
            rel_after = raw.split("caltech101/", 1)[-1].lstrip("/")
            trials.append(root / "caltech101" / rel_after)

    seen: set[str] = set()
    for t in trials:
        try:
            resolved = t.resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return str(resolved)
    return None


def build_result_rows(
    similarities: np.ndarray,
    indices: np.ndarray,
    indexed_paths: Optional[List[dict]],
) -> List[dict[str, Any]]:
    """Each row: faiss_idx, score, caption, has_local, picsum_id."""
    rows: List[dict[str, Any]] = []
    if indices is None or len(indices[0]) == 0:
        return rows
    for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
        idx = int(idx)
        picsum_id = (237 + i * 10) % 1000
        caption = f"Result {i + 1} (Score: {float(score):.2f})"
        has_local = False
        if indexed_paths and idx < len(indexed_paths):
            item = indexed_paths[idx]
            img_path = item["path"]
            category = item.get("category", "Unknown")
            caption = f"{category} (Score: {float(score):.2f})"
            has_local = resolve_local_image_path(img_path) is not None
        rows.append(
            {
                "faiss_idx": idx,
                "score": float(score),
                "caption": caption,
                "has_local": has_local,
                "picsum_id": picsum_id,
            }
        )
    return rows


def resolve_result_file_path(faiss_idx: int, indexed_paths: List[dict]) -> Optional[str]:
    if faiss_idx < 0 or faiss_idx >= len(indexed_paths):
        return None
    img_path = indexed_paths[faiss_idx]["path"]
    return resolve_local_image_path(img_path)


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"
