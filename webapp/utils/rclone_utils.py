"""
Rclone helpers for localhost: list remotes, browse with lsjson, copy files, manage rcd web GUI.
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import time
import shutil
import string
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)


def rclone_binary() -> Optional[str]:
    return shutil.which("rclone")


def _run_rclone(
    args: List[str],
    *,
    timeout: float = 180,
) -> Tuple[int, str, str]:
    bin_path = rclone_binary()
    if not bin_path:
        return -1, "", "rclone not found in PATH"
    try:
        p = subprocess.run(
            [bin_path, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "rclone command timed out"
    except OSError as e:
        return -1, "", str(e)


def list_remotes() -> Tuple[bool, List[str], str]:
    code, out, err = _run_rclone(["listremotes"])
    if code != 0:
        return False, [], err or out
    names: List[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            line = line[:-1]
        names.append(line)
    return True, sorted(set(names)), ""


def pick_free_port(start: int = 5572, attempts: int = 12) -> int:
    """Return a likely-free localhost TCP port for rcd (best-effort)."""
    for port in range(start, start + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start


def normalize_remote_path(path: str) -> str:
    p = path.strip().replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    p = p.strip("/")
    parts = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise ValueError("path must not contain '..'")
        parts.append(seg)
    return "/".join(parts)


def parent_remote_path(remote_path: str) -> str:
    try:
        p = normalize_remote_path(remote_path)
    except ValueError:
        return ""
    if not p:
        return ""
    parts = p.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def validate_remote_name(name: str, allowed: List[str]) -> bool:
    n = name.strip().rstrip(":")
    return n in allowed


def lsjson(remote: str, remote_path: str) -> Tuple[bool, List[dict], str]:
    """List a directory on the remote. remote_path is relative (no remote: prefix)."""
    rp = normalize_remote_path(remote_path) if remote_path else ""
    remote = remote.strip().rstrip(":")
    target = f"{remote}:{rp}" if rp else f"{remote}:"
    code, out, err = _run_rclone(
        ["lsjson", target],
        timeout=120,
    )
    if code != 0:
        return False, [], (err or out).strip() or "lsjson failed"
    try:
        data = json.loads(out)
        if not isinstance(data, list):
            return False, [], "unexpected lsjson output"
        return True, data, ""
    except json.JSONDecodeError as e:
        return False, [], f"invalid JSON from lsjson: {e}"


def remote_file_path_for_copy(list_cwd: str, path_from_entry: str) -> str:
    """
    Build the full remote path for copyto.

    lsjson Path is relative to the directory being listed (not always remote root).
    copyto remote:basename alone points at the wrong place for Drive.
    """
    p = normalize_remote_path(path_from_entry) if path_from_entry else ""
    c = normalize_remote_path(list_cwd) if list_cwd else ""
    if not c or not p:
        return p
    if p == c or p.startswith(c + "/"):
        return p
    return normalize_remote_path(f"{c}/{p}")


def _candidate_paths_for_copy(list_cwd: str, path_from_entry: str) -> List[str]:
    """Try these in order; Drive/lsjson path semantics vary by version."""
    out: List[str] = []
    seen: set = set()

    def add(s: str) -> None:
        if not s:
            return
        try:
            n = normalize_remote_path(s)
        except ValueError:
            return
        if n and n not in seen:
            seen.add(n)
            out.append(n)

    p = path_from_entry or ""
    c = list_cwd or ""

    add(remote_file_path_for_copy(c, p))
    try:
        add(normalize_remote_path(p))
    except ValueError:
        pass
    if c and p:
        base = os.path.basename(p.replace("\\", "/"))
        if base:
            try:
                add(normalize_remote_path(f"{c}/{base}"))
            except ValueError:
                pass
    return out


def _copyto_one(remote: str, rp: str, dest_file: str) -> Tuple[int, str]:
    bin_path = rclone_binary()
    if not bin_path:
        return -1, "rclone not found in PATH"
    target = f"{remote}:{rp}"
    try:
        p = subprocess.run(
            [bin_path, "copyto", target, dest_file],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return p.returncode, (p.stderr or p.stdout or "").strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, str(e)


def _cat_to_temp(remote: str, rp: str, dest_file: str) -> Tuple[int, str]:
    """Fetch file bytes via rclone cat (works when copyto mis-resolves path)."""
    bin_path = rclone_binary()
    if not bin_path:
        return -1, "rclone not found in PATH"
    target = f"{remote}:{rp}"
    try:
        p = subprocess.run(
            [bin_path, "cat", target],
            capture_output=True,
            timeout=600,
            check=False,
        )
        if p.returncode != 0:
            return p.returncode, (p.stderr or b"").decode("utf-8", errors="replace").strip()
        with open(dest_file, "wb") as f:
            f.write(p.stdout or b"")
        return 0, ""
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, str(e)


def copyto_temp(
    remote: str,
    remote_path: str,
    suffix: str = "",
    *,
    list_cwd: str = "",
) -> Tuple[bool, Optional[str], str]:
    """Copy remote file to a temp path. remote_path is from lsjson; pass list_cwd when listing a subfolder."""
    remote = remote.strip().rstrip(":")
    paths = _candidate_paths_for_copy(list_cwd, remote_path)
    if not paths:
        return False, None, "empty remote path"
    ext = os.path.splitext(paths[0])[1] or suffix or ".bin"
    last_err = ""

    for rp in paths:
        fd, tmp = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        code, err = _copyto_one(remote, rp, tmp)
        if code == 0:
            return True, tmp, ""
        try:
            os.unlink(tmp)
        except OSError:
            pass
        last_err = err or last_err

        fd, tmp = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        code, err = _cat_to_temp(remote, rp, tmp)
        if code == 0:
            return True, tmp, ""
        try:
            os.unlink(tmp)
        except OSError:
            pass
        last_err = err or last_err

    return False, None, last_err or "copyto failed"


def is_image_name(name: str) -> bool:
    _, ext = os.path.splitext(name.lower())
    return ext in IMAGE_EXTENSIONS


@dataclass
class RcdConfig:
    port: int
    user: str
    password: str


def generate_rc_credentials() -> Tuple[str, str]:
    alphabet = string.ascii_letters + string.digits
    user = "app"
    pw = "".join(secrets.choice(alphabet) for _ in range(24))
    return user, pw


def start_rcd_webgui(
    port: int,
    rc_user: str,
    rc_pass: str,
) -> Tuple[Optional[subprocess.Popen], str]:
    """Start rclone rcd with web GUI on 127.0.0.1:port. Caller stores Popen in session_state."""
    if not rclone_binary():
        return None, "rclone not found in PATH"
    addr = f"127.0.0.1:{port}"
    cmd = [
        "rclone",
        "rcd",
        "--rc-web-gui",
        "--rc-web-gui-no-open-browser",
        f"--rc-addr={addr}",
        f"--rc-user={rc_user}",
        f"--rc-pass={rc_pass}",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except OSError as e:
        return None, str(e)
    return proc, ""


def rcd_exited_early(proc: subprocess.Popen, wait_s: float = 1.5) -> Tuple[bool, str]:
    """Return (True, reason) if the process died before wait_s seconds."""
    time.sleep(wait_s)
    code = proc.poll()
    if code is None:
        return False, ""
    return True, f"rclone rcd exited with code {code} (port may be in use or rclone error)"


def stop_rcd_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def rcd_gui_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def entry_remote_path(entry: dict) -> str:
    """Path relative to remote root from lsjson."""
    p = entry.get("Path") or entry.get("Name") or ""
    return str(p).replace("\\", "/").strip()


def find_listable_entries(entries: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split lsjson entries into dirs and image files."""
    dirs: List[dict] = []
    files: List[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = entry_remote_path(e)
        if e.get("IsDir"):
            dirs.append(e)
        elif is_image_name(name):
            files.append(e)
    dirs.sort(key=lambda x: entry_remote_path(x).lower())
    files.sort(key=lambda x: entry_remote_path(x).lower())
    return dirs, files
