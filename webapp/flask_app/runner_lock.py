"""Single lock so training and precompute jobs do not run concurrently."""
from __future__ import annotations

import threading

JOB_LOCK = threading.Lock()
