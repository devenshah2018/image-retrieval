"""Build search result rows for HTML (parity with display_utils)."""
from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np


def paths_to_try_for_item(img_path: str) -> List[str]:
    return [
        img_path,
        img_path.replace("caltech101/", ""),
        os.path.join("caltech101", img_path),
    ]


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
            for p in paths_to_try_for_item(img_path):
                if os.path.exists(p):
                    has_local = True
                    break
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
    for p in paths_to_try_for_item(img_path):
        if os.path.exists(p):
            return p
    return None


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
