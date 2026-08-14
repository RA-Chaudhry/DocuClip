import os
import re
import shutil
import time
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.config import settings

# Aapke DocuClip DB imports
from app.db.session import SessionLocal, engine, Base, database_status, ensure_schema
from app.db.models import VideoJob, ClipResult
from app.worker.tasks.io_tasks import extract_audio_and_proxy
from app.api.v1.endpoints.jobs import router as jobs_router
from app.services.progress import register_task, set_progress

# Initialize database tables if they do not exist
try:
    ensure_schema()
except Exception as exc:
    print(f"Database initialization warning: {exc}")

# Ensure clips and uploads directories exist and mount static files route
os.makedirs("clips", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app = FastAPI(title="DocuClip API", version="0.1.0")
app.include_router(jobs_router)

# Mount /clips directory to serve generated video clips directly to frontend
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join("frontend", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="""
            <html>
                <head>
                    <title>DocuClip Frontend Not Found</title>
                    <style>
                        body { font-family: sans-serif; background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                        h1 { color: #f43f5e; font-size: 2.5rem; margin-bottom: 0.5rem; }
                        p { color: #94a3b8; font-size: 1.1rem; }
                    </style>
                </head>
                <body>
                    <h1>frontend/index.html Not Found</h1>
                    <p>Please ensure that index.html is created in the frontend directory at the project root.</p>
                </body>
            </html>
            """,
            status_code=404
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Database session helper function
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _safe_db_error_response(message: str):
    return {
        "message": message,
        "status": "degraded",
        "warning": "Database service is not available; job was not persisted."
    }

@app.get("/health")
def health_check():
    db = database_status()
    return {
        "status": "healthy" if db["connected"] else "degraded",
        "database": db,
    }

class VideoRequest(BaseModel):
    url: str
    user_style: str = "fast_paced"
    clip_duration: str = "medium"
    video_quality: int = 480
    min_duration_seconds: float = 25.0
    max_duration_seconds: float = 75.0
    target_duration_seconds: float = None


DURATION_PRESETS = {
    "short": (15.0, 30.0),
    "medium": (30.0, 60.0),
    "long": (60.0, 90.0),
}

@app.post("/api/v1/process-video")
def process_video(request: VideoRequest, db: Session = Depends(get_db)):
    min_duration, max_duration = DURATION_PRESETS.get(
        request.clip_duration.lower(),
        (request.min_duration_seconds, request.max_duration_seconds),
    )
    quality = min(settings.MAX_QUALITY, max(360, int(request.video_quality)))
    try:
        # 1. Database mein naya record banayein (Status by default PENDING hoga)
        new_video = VideoJob(
            url=request.url,
            status="PENDING",
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            video_quality=quality,
        )
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        video_id = new_video.id
    except Exception as exc:
        print(f"DB unavailable during video submission: {exc}")
        video_id = None

    try:
        # 2. Celery ko parameters pass karein!
        task = extract_audio_and_proxy.delay(
            url=request.url,
            video_id=video_id if video_id is not None else 0,
            user_style=request.user_style,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            target_duration_seconds=request.target_duration_seconds,
            video_quality=quality,
        )
        if video_id is not None:
            new_video.task_id = str(task.id)
            db.commit()
            register_task(video_id, str(task.id))
            set_progress(video_id, "Queued", 0, 0, "Waiting for worker")
    except Exception as exc:
        print(f"Celery queue unavailable during video submission: {exc}")
        return {
            **_safe_db_error_response("Video submission queued partially; background worker is not available."),
            "task_id": None,
            "video_id": video_id,
            "video_url": request.url,
            "user_style": request.user_style,
            "min_duration_seconds": min_duration,
            "max_duration_seconds": max_duration,
            "clip_duration": request.clip_duration,
            "video_quality": quality,
            "target_duration_seconds": request.target_duration_seconds
        }

    return {
        "message": "Video successfully submitted to the IO Queue!",
        "task_id": str(task.id),
        "video_id": video_id,
        "video_url": request.url,
        "user_style": request.user_style,
        "min_duration_seconds": min_duration,
        "max_duration_seconds": max_duration,
        "clip_duration": request.clip_duration,
        "video_quality": quality,
        "target_duration_seconds": request.target_duration_seconds
    }

@app.post("/api/v1/upload-video")
async def upload_and_process_video(
    file: UploadFile = File(...),
    user_style: str = Form("fast_paced"),
    clip_duration: str = Form("medium"),
    min_duration_seconds: float = Form(25.0),
    max_duration_seconds: float = Form(75.0),
    target_duration_seconds: Optional[float] = Form(None),
    video_quality: int = Form(480),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded video file, saves it to the local uploads directory,
    and initiates the DocuClip processing pipeline directly.
    """
    os.makedirs("uploads", exist_ok=True)
    
    clean_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename or "video.mp4")
    file_id = f"upload_{int(time.time())}_{clean_filename}"
    saved_path = os.path.abspath(os.path.join("uploads", file_id))
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    file_url = f"file://{saved_path}"
    
    min_duration, max_duration = DURATION_PRESETS.get(
        clip_duration.lower(),
        (min_duration_seconds, max_duration_seconds),
    )
    quality = min(settings.MAX_QUALITY, max(360, int(video_quality)))
    
    try:
        new_video = VideoJob(
            url=file_url,
            status="PENDING",
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            video_quality=quality,
        )
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        video_id = new_video.id
    except Exception as exc:
        print(f"DB unavailable during video upload: {exc}")
        video_id = None

    try:
        task = extract_audio_and_proxy.delay(
            url=file_url,
            video_id=video_id if video_id is not None else 0,
            user_style=user_style,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            target_duration_seconds=target_duration_seconds,
            video_quality=quality,
        )
        if video_id is not None:
            new_video.task_id = str(task.id)
            db.commit()
            register_task(video_id, str(task.id))
            set_progress(video_id, "Queued", 0, 0, f"Uploaded video '{file.filename}' queued for processing")
    except Exception as exc:
        print(f"Celery queue unavailable during video upload: {exc}")
        return {
            **_safe_db_error_response("Video upload queued partially; background worker is not available."),
            "task_id": None,
            "video_id": video_id,
            "video_url": file_url,
            "filename": file.filename,
            "user_style": user_style,
            "min_duration_seconds": min_duration,
            "max_duration_seconds": max_duration,
            "clip_duration": clip_duration,
            "video_quality": quality,
            "target_duration_seconds": target_duration_seconds
        }

    return {
        "message": "Local video successfully uploaded and submitted to the processing queue!",
        "task_id": str(task.id),
        "video_id": video_id,
        "video_url": file_url,
        "filename": file.filename,
        "user_style": user_style,
        "min_duration_seconds": min_duration,
        "max_duration_seconds": max_duration,
        "clip_duration": clip_duration,
        "video_quality": quality,
        "target_duration_seconds": target_duration_seconds
    }

@app.get("/api/v1/jobs")
def list_jobs(db: Session = Depends(get_db)):
    video_jobs = db.query(VideoJob).order_by(VideoJob.created_at.desc()).all()
    return [
        {
            "video_id": job.id,
            "url": job.url,
            # A completed job without persisted clips is not a successful job.
            "status": "FAILED" if job.status.upper() == "COMPLETED" and not job.clips else job.status,
            "created_at": job.created_at,
            "clip_count": len(job.clips),
            "task_id": job.task_id,
            "clip_duration": {
                "min": job.min_duration_seconds,
                "max": job.max_duration_seconds,
            },
            "video_quality": job.video_quality,
        }
        for job in video_jobs
    ]

def ensure_clip_files_exist(video_job: VideoJob):
    """Automatically regenerate missing physical clip files on demand if clips exist in DB but not on disk."""
    import os
    if not video_job.clips:
        return
    missing_clips = [c for c in video_job.clips if not os.path.exists(c.clip_path)]
    if not missing_clips:
        return
        
    try:
        from app.services.downloader import download_media
        import subprocess
        
        audio_path, video_path = download_media(video_job.url)
        if video_path and os.path.exists(video_path):
            os.makedirs("clips", exist_ok=True)
            for c in missing_clips:
                duration = max(1.0, c.end_time - c.start_time)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(c.start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264", "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                    c.clip_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0:
                    fb_cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(c.start_time),
                        "-i", video_path,
                        "-t", str(duration),
                        "-c:v", "libx264",
                        "-avoid_negative_ts", "make_zero",
                        c.clip_path
                    ]
                    subprocess.run(fb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Only remove if downloaded temporary file, not local uploaded file
            if not video_job.url.startswith("file://"):
                if audio_path and os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except Exception: pass
                if video_path and os.path.exists(video_path):
                    try: os.remove(video_path)
                    except Exception: pass
    except Exception as e:
        print(f"Error auto-repairing missing clip files for Video ID {video_job.id}: {str(e)}")


@app.get("/api/v1/jobs/{video_id}")
def get_job_status(video_id: int, db: Session = Depends(get_db)):
    import json
    from app.services.downloader import generate_hash
    video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
    if not video_job:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    # Auto-repair missing clip files on disk if any
    ensure_clip_files_exist(video_job)

    # Backfill transcript metadata for clips created before this field was
    # added, when the job transcript is still available in Redis cache.
    transcript_segments = []
    try:
        import redis
        cached = redis.Redis.from_url(settings.REDIS_URL).get(
            f"transcript:{generate_hash(video_job.url)}"
        )
        if cached:
            cached_json = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            transcript_segments = json.loads(cached_json).get("segments", [])
    except Exception as exc:
        print(f"Transcript cache unavailable for Video ID {video_id}: {exc}")

    metadata_updated = False
    if transcript_segments:
        for clip in video_job.clips:
            if not clip.transcript_text:
                clip.transcript_text = " ".join(
                    str(segment.get("text", "")).strip()
                    for segment in transcript_segments
                    if float(segment.get("end", 0.0)) >= clip.start_time
                    and float(segment.get("start", 0.0)) <= clip.end_time
                    and str(segment.get("text", "")).strip()
                ).strip()
                metadata_updated = True
        if metadata_updated:
            db.commit()

    clips_data = []
    for clip in video_job.clips:
        raw_types = getattr(clip, "clip_type", '["educational"]')
        try:
            parsed_types = json.loads(raw_types) if raw_types else ["educational"]
            if not isinstance(parsed_types, list):
                parsed_types = [str(parsed_types)]
        except Exception:
            parsed_types = [str(raw_types)]
            
        clips_data.append({
            "id": clip.id,
            "clip_path": clip.clip_path,
            "hybrid_score": clip.hybrid_score,
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "subject_hint": clip.subject_hint,
            "clip_types": parsed_types,
            "reason": clip.reason,
            "transcript": clip.transcript_text or ""
        })

    return {
        "video_id": video_job.id,
        "url": video_job.url,
        "status": "FAILED" if video_job.status.upper() == "COMPLETED" and not clips_data else video_job.status,
        "created_at": video_job.created_at,
        "clips": clips_data
    }
