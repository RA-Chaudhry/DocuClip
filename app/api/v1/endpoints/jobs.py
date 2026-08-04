from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from app.db.models import VideoJob
from app.db.session import SessionLocal
from app.services.progress import (
    get_progress,
    request_cancel,
    register_task,
    set_progress,
    task_key,
)
from app.worker.celery_app import celery_app
import redis
from app.core.config import settings

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{job_id}/progress")
def job_progress(job_id: int, db: Session = Depends(get_db)):
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    progress = get_progress(job_id)
    progress["job_id"] = job_id
    progress["status"] = job.status
    return progress


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if job.status.upper() in {"COMPLETED", "FAILED", "CANCELLED"}:
        return {"job_id": job_id, "status": job.status, "cancelled": job.status.upper() == "CANCELLED"}

    request_cancel(job_id)
    client = redis.Redis.from_url(settings.REDIS_URL)
    task_ids = set(client.smembers(task_key(job_id)))
    if job.task_id:
        task_ids.add(job.task_id.encode())

    revoked = []
    for raw_task_id in task_ids:
        task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else str(raw_task_id)
        AsyncResult(task_id, app=celery_app).revoke(terminate=True, signal="SIGTERM")
        register_task(job_id, task_id)
        revoked.append(task_id)

    job.status = "CANCELLED"
    db.commit()
    set_progress(job_id, "Cancelled", 100, 0, "Stopped by user")
    return {"job_id": job_id, "status": "CANCELLED", "cancelled": True, "revoked_tasks": revoked}
