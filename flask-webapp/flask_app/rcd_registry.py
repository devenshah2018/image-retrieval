"""Per-session rclone rcd subprocess registry (localhost)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .rclone_import import get_rclone_utils


def _ru():
    return get_rclone_utils()

_registry: Dict[str, Dict[str, Any]] = {}


def get_state(sid: str) -> Dict[str, Any]:
    if sid not in _registry:
        _registry[sid] = {
            "proc": None,
            "port": 5572,
            "user": None,
            "password": None,
        }
    return _registry[sid]


def sync_proc(sid: str) -> None:
    st = get_state(sid)
    proc = st.get("proc")
    if proc is not None and proc.poll() is not None:
        st["proc"] = None


def start_for_session(sid: str) -> tuple[bool, str, Optional[int]]:
    ru = _ru()
    sync_proc(sid)
    st = get_state(sid)
    ru.stop_rcd_process(st.get("proc"))
    st["proc"] = None
    port = ru.pick_free_port(5572)
    st["port"] = port
    u, pw = ru.generate_rc_credentials()
    st["user"] = u
    st["password"] = pw
    proc, err = ru.start_rcd_webgui(port, u, pw)
    if not proc:
        return False, err or "Failed to start rclone rcd", None
    st["proc"] = proc
    died, reason = ru.rcd_exited_early(proc, wait_s=2.0)
    if died:
        st["proc"] = None
        ru.stop_rcd_process(proc)
        return False, reason, None
    return True, "", port


def stop_for_session(sid: str) -> None:
    ru = _ru()
    st = get_state(sid)
    ru.stop_rcd_process(st.get("proc"))
    st["proc"] = None


def gui_url_for_session(sid: str) -> str:
    st = get_state(sid)
    return _ru().rcd_gui_url(st["port"])


def rcd_alive(sid: str) -> bool:
    sync_proc(sid)
    st = get_state(sid)
    p = st.get("proc")
    return p is not None and p.poll() is None
