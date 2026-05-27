"""In-process job store for long-running async inference tasks (e.g. FLUX with a cold model cache).

Jobs are kept in a process-local dict — no persistence, no inter-process sharing. That is
intentional: this is a single-worker demo server. The ThreadPoolExecutor is capped at one
worker because the GPU can only run one inference at a time; a second request queues in the
executor rather than OOMing.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | downloading_model | running | done | error
    message: str = ""
    created_at: float = field(default_factory=time.monotonic)
    phase_started_at: float = field(default_factory=time.monotonic)
    result_b64: str | None = None
    result_bbox: list | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_status(self, status: str, message: str = "") -> None:
        with self._lock:
            self.status = status
            self.message = message
            self.phase_started_at = time.monotonic()

    def to_response(self) -> dict:
        now = time.monotonic()
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "elapsed_total_s": now - self.created_at,
            "elapsed_phase_s": now - self.phase_started_at,
            "result_b64": self.result_b64,
            "result_bbox": self.result_bbox,
            "error": self.error,
        }


# TODO: _jobs grows unbounded — completed jobs (and their result_b64 images) are never
#       evicted. Add a TTL sweep or cap (e.g. keep last N) before running this in production.
_jobs: dict[str, Job] = {}
_store_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    with _store_lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def submit(fn, *args, **kwargs):
    return _executor.submit(fn, *args, **kwargs)
