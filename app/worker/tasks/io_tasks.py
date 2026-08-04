from celery.utils.log import get_task_logger
import redis
from app.worker.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import VideoJob
from app.services.downloader import generate_hash, download_media
from app.services.progress import (
    JobCancelled,
    ProgressHeartbeat,
    is_cancel_requested,
    raise_if_cancelled,
    register_task,
    set_progress,
)
from app.worker.tasks.gpu_tasks import process_gpu_pipeline

logger = get_task_logger(__name__)

# Celery task declaration jo io_queue par run karegi binding parameters and retries control parameters ke sath
@celery_app.task(queue="io_queue", bind=True, acks_late=True, max_retries=3)
def extract_audio_and_proxy(
    self, 
    url: str, 
    video_id: int, 
    user_style: str = "fast_paced",
    min_duration_seconds: float = 25.0,
    max_duration_seconds: float = 75.0,
    target_duration_seconds: float = None,
    video_quality: int = 480,
):
    """
    RAG & Caching Logic implementation with custom duration support.
    """
    r = redis.Redis.from_url(settings.REDIS_URL)
    video_hash = generate_hash(url)
    cache_key = f"transcript:{video_hash}"
    register_task(video_id, self.request.id)
    cached_data = r.get(cache_key)
    
    db = SessionLocal()
    try:
        raise_if_cancelled(video_id)
        if cached_data is not None:
            cached_str = cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data
            video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
            if video_job:
                video_job.status = "Cache Hit - Processing Transcript"
                db.commit()
            set_progress(video_id, "Downloading", 15, 30, "Transcript cached; preparing media")
                
            logger.info(f"Cache Hit for URL: {url} (ID: {video_id}). Enqueuing GPU pipeline with cached transcript.")
            
            child_task = process_gpu_pipeline.apply_async(
                kwargs={
                    "audio_path": None,
                    "proxy_video_path": None,
                    "video_id": video_id,
                    "url": url,
                    "user_style": user_style,
                    "min_duration_seconds": min_duration_seconds,
                    "max_duration_seconds": max_duration_seconds,
                    "target_duration_seconds": target_duration_seconds,
                    "video_quality": video_quality,
                },
                queue="gpu_queue"
            )
            register_task(video_id, str(child_task.id))
            if video_job:
                video_job.task_id = str(child_task.id)
                db.commit()
            return None, None, cached_str
            
        else:
            video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
            if video_job:
                video_job.status = "Downloading Audio & Proxy"
                db.commit()
            set_progress(video_id, "Downloading", 5, 180, "Downloading merged video")
                
            logger.info(f"Cache Miss for URL: {url} (ID: {video_id}). Starting media download.")
            with ProgressHeartbeat(video_id, "Downloading", 5, 24, 180, "Downloading merged video"):
                audio_path, proxy_video_path = download_media(url, quality=video_quality, job_id=video_id)
            
            if video_job:
                video_job.status = "Audio & Proxy Downloaded"
                db.commit()
            set_progress(video_id, "Downloading", 25, 120, "Media download complete")
                
            logger.info(f"Successfully downloaded media for URL: {url} (ID: {video_id}).")
            
            child_task = process_gpu_pipeline.apply_async(
                kwargs={
                    "audio_path": audio_path,
                    "proxy_video_path": proxy_video_path,
                    "video_id": video_id,
                    "url": url,
                    "user_style": user_style,
                    "min_duration_seconds": min_duration_seconds,
                    "max_duration_seconds": max_duration_seconds,
                    "target_duration_seconds": target_duration_seconds,
                    "video_quality": video_quality,
                },
                queue="gpu_queue"
            )
            register_task(video_id, str(child_task.id))
            if video_job:
                video_job.task_id = str(child_task.id)
                db.commit()
            return audio_path, proxy_video_path, None
    except JobCancelled:
        db.rollback()
        video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
        if video_job:
            video_job.status = "CANCELLED"
            db.commit()
        set_progress(video_id, "Cancelled", 100, 0, "Stopped by user")
        return None, None, None
    except Exception as exc:
        db.rollback()
        logger.error(f"Error handling task extract_audio_and_proxy for ID {video_id}: {str(exc)}")
        
        # --- NEW ADDITION: Mark job as FAILED in database ---
        video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
        if video_job:
            video_job.status = "FAILED"
            db.commit()
        # ----------------------------------------------------
        
        # Automatic retry behavior with 10 seconds delay backoff timing
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()
