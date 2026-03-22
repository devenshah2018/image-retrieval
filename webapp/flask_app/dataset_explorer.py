"""Browse Caltech-101 under webapp/caltech101/ (101_ObjectCategories/...)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXT = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)


def caltech_root(webapp_root: Path) -> Path:
    return webapp_root / "caltech101"


def safe_rel_path(rel: str) -> str:
    rel = rel.replace("\\", "/").strip()
    rel = rel.lstrip("/")
    parts = [p for p in rel.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        return ""
    return "/".join(parts)


def resolve_under(root: Path, rel: str) -> Optional[Path]:
    """Return resolved path if it exists and stays under root; else None."""
    if not root.is_dir():
        return None
    rel = safe_rel_path(rel)
    root_r = root.resolve()
    if not rel:
        return root_r
    cand = (root / rel).resolve()
    try:
        cand.relative_to(root_r)
    except ValueError:
        return None
    return cand if cand.exists() else None


def list_folder(root: Path, rel: str) -> Tuple[List[str], List[str], Optional[str]]:
    """Subfolder names and image file names under rel. Error message if invalid."""
    d = resolve_under(root, rel)
    if d is None:
        return [], [], "Path not found"
    if not d.is_dir():
        return [], [], "Not a folder"
    dirs: List[str] = []
    files: List[str] = []
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir():
            dirs.append(p.name)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXT:
            files.append(p.name)
    return dirs, files, None


def search_folders(root: Path, query: str, max_results: int = 100) -> List[str]:
    """Relative paths (posix) of directories whose name contains query (case-insensitive)."""
    if not query.strip() or not root.is_dir():
        return []
    q = query.lower().strip()
    out: List[str] = []
    root_r = root.resolve()
    for dirpath, dirnames, _ in os.walk(root):
        for d in sorted(dirnames, key=str.lower):
            if q not in d.lower():
                continue
            full = Path(dirpath) / d
            try:
                rel = full.relative_to(root_r)
                out.append(rel.as_posix())
            except ValueError:
                continue
    out.sort()
    return out[:max_results]


def breadcrumb_parts(rel: str) -> List[Tuple[str, str]]:
    """[(label, cumulative_rel), ...] for rel like a/b/c."""
    rel = safe_rel_path(rel)
    if not rel:
        return []
    parts = rel.split("/")
    out: List[Tuple[str, str]] = []
    acc: List[str] = []
    for p in parts:
        acc.append(p)
        out.append((p, "/".join(acc)))
    return out
