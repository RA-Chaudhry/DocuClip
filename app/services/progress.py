import json
import threading
import time

import redis
from app.core.config import settings


class JobCancelled(Exception):
    """Raised inside workers when a user cancellation was requested."""


def _redis():
    return redis.Redis.from_url(settings.REDIS_URL)


def progress_key(job_id: int) -> str:
    return f"job:{job_id}:progress"


def task_key(job_id: int) -> str:
    return f"job:{job_id}:tasks"


def cancel_key(job_id: int) -> str:
    return f"job:{job_id}:cancel"


def set_progress(job_id: int, stage: str, percent: int, eta_seconds: int = 0, detail: str = ""):
    payload = {
        "stage": stage,
        "percent": max(0, min(100, int(percent))),
        "eta_seconds": max(0, int(eta_seconds or 0)),
        "detail": detail,
        "updated_at": int(time.time()),
    }
    client = _redis()
    client.set(progress_key(job_id), json.dumps(payload), ex=7 * 24 * 3600)
    return payload


def get_progress(job_id: int):
    raw = _redis().get(progress_key(job_id))
    if not raw:
        return {"stage": "Queued", "percent": 0, "eta_seconds": 0, "detail": ""}
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)


def register_task(job_id: int, task_id: str):
    if task_id:
        client = _redis()
        client.sadd(task_key(job_id), task_id)
        client.expire(task_key(job_id), 7 * 24 * 3600)


def request_cancel(job_id: int):
    client = _redis()
    client.set(cancel_key(job_id), "1", ex=24 * 3600)


def is_cancel_requested(job_id: int) -> bool:
    return bool(_redis().exists(cancel_key(job_id)))


def raise_if_cancelled(job_id: int):
    if is_cancel_requested(job_id):
        raise JobCancelled(f"Job {job_id} was cancelled by the user")


class ProgressHeartbeat:
    """Continuously update a job while a slow synchronous stage is running."""

    def __init__(self, job_id: int, stage: str, start_percent: int, end_percent: int,
                 estimated_seconds: int, detail: str = ""):
        self.job_id = job_id
        self.stage = stage
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.estimated_seconds = max(1, estimated_seconds)
        self.detail = detail
        self.started_at = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = None

    def _tick(self):
        while not self.stop_event.wait(1.0):
            elapsed = time.monotonic() - self.started_at
            ratio = min(0.98, elapsed / self.estimated_seconds)
            percent = self.start_percent + int((self.end_percent - self.start_percent) * ratio)
            eta = max(1, int(self.estimated_seconds - elapsed))
            try:
                set_progress(self.job_id, self.stage, percent, eta, self.detail)
            except Exception:
                pass

    def __enter__(self):
        set_progress(self.job_id, self.stage, self.start_percent, self.estimated_seconds, self.detail)
        self.thread = threading.Thread(target=self._tick, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        return False
