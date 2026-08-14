import os
from typing import Optional

from celery.utils.log import get_task_logger
import redis
import json
from app.worker.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import VideoJob, ClipResult
from app.services.downloader import generate_hash
from app.services import transcriber, rag_engine, llm_engine, visual_engine
from app.services.clip_generator import generate_all_clips_parallel
from app.services.progress import (
    JobCancelled,
    ProgressHeartbeat,
    raise_if_cancelled,
    register_task,
    set_progress,
)

logger = get_task_logger(__name__)


def _transcript_for_range(segments: list[dict], start: float, end: float) -> str:
    """Return the exact Whisper text selected by a clip boundary."""
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if float(segment.get("end", 0.0)) >= start
        and float(segment.get("start", 0.0)) <= end
        and str(segment.get("text", "")).strip()
    ).strip()

# GPU pipeline task running on gpu_queue with basic binding and custom exceptions retry parameters
@celery_app.task(queue="gpu_queue", bind=True, acks_late=True, max_retries=1)
def process_gpu_pipeline(
    self, 
    audio_path: str, 
    proxy_video_path: str, 
    video_id: int, 
    url: str, 
    user_style: str = "fast_paced",
    min_duration_seconds: float = 25.0,
    max_duration_seconds: float = 75.0,
    target_duration_seconds: Optional[float] = None,
    video_quality: int = 480,
):
    """
    GPU task flow:
    1. transcribe or fetch cached transcript from Redis.
    2. semantic search/RAG run area selection with 30s overlap.
    3. LLM call to extract viral clips adhering to retention and custom user duration bounds.
    4. Real audio energy & retention curve analysis.
    5. Save generated clips in ClipResult database and cut physical MP4 files.
    """
    logger.info(f"Initiating GPU workflow for Video ID: {video_id}, User Style: {user_style}, Duration bounds: [{min_duration_seconds}s - {max_duration_seconds}s]")
    register_task(video_id, self.request.id)
    set_progress(video_id, "Transcribing", 28, 150, "Preparing transcript")
    
    # Adjust duration limits if user requested specific target duration
    if target_duration_seconds and target_duration_seconds > 0:
        min_duration_seconds = max(10.0, target_duration_seconds * 0.8)
        max_duration_seconds = target_duration_seconds * 1.25
        
    r = redis.Redis.from_url(settings.REDIS_URL)
    video_hash = generate_hash(url)
    cache_key = f"transcript:{video_hash}"
    
    try:
        raise_if_cancelled(video_id)
        # 1. Get transcription (from cache if Cache Hit, else run Whisper)
        if audio_path is None:
            logger.info(f"Cache Hit processing path: Reading transcription from Redis key {cache_key}")
            cached_data = r.get(cache_key)
            if cached_data is None:
                raise ValueError(f"Expected cached transcript under key {cache_key} but got None.")
            
            transcript_dict = json.loads(cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data)
        else:
            logger.info(f"Transcribing audio target: {audio_path}")
            with ProgressHeartbeat(video_id, "Transcribing", 28, 55, 150, "Transcribing audio"):
                transcript_dict = transcriber.transcribe_audio(audio_path)
            r.setex(cache_key, 30 * 24 * 3600, json.dumps(transcript_dict))
            logger.info(f"Successfully cached transcription outputs under Redis key: {cache_key}")
        set_progress(video_id, "Transcribing", 55, 90, "Transcript ready")
        raise_if_cancelled(video_id)
            
        signal_data = {"audio": [], "scenes": []}
        
        # 3. Chunking transcript data (3-minute window blocks with 30s overlap)
        chunks = rag_engine.chunk_transcript(transcript_dict, window_seconds=180, overlap_seconds=30)
        
        if user_style == "fast_paced":
            target_concept = "high energy exciting moment dramatic shock hook emotional peak action viral payoff"
        elif user_style == "educational":
            target_concept = "educational insight explanation core lesson definition information summary knowledge"
        else:
            target_concept = "important highlight interesting statement key takeaway"
            
        logger.info(f"Extracting top chunks matching target query: {target_concept}")
        with ProgressHeartbeat(video_id, "Analyzing", 55, 60, 45, "Finding the strongest moments"):
            top_chunks = rag_engine.get_top_chunks(chunks, target_concept=target_concept, top_k=8)
        set_progress(video_id, "Analyzing", 60, 60, "Checking transcript moments")
        raise_if_cancelled(video_id)

        # Ensure the actual video is available before AI scoring so visual
        # signals are part of clip ranking, including transcript-cache jobs.
        if not proxy_video_path or not os.path.exists(proxy_video_path):
            logger.info(f"Proxy video not present for Video ID {video_id}; downloading for visual analysis.")
            from app.services.downloader import download_media
            with ProgressHeartbeat(video_id, "Analyzing", 60, 63, 60, "Preparing video for visual analysis"):
                dl_audio, dl_video = download_media(url, quality=video_quality, job_id=video_id)
            if not audio_path or not os.path.exists(audio_path):
                audio_path = dl_audio
            proxy_video_path = dl_video

        with ProgressHeartbeat(video_id, "Analyzing", 63, 66, 60, "Finding visual scene and motion hotspots"):
            visual_hotspots = visual_engine.get_top_visual_hotspots(proxy_video_path, count=10)
            visual_score = (
                sum(item["visual_score"] for item in visual_hotspots) / len(visual_hotspots)
                if visual_hotspots else 7.5
            )
        logger.info(f"Visual analysis score for Video ID {video_id}: {visual_score}")
        raise_if_cancelled(video_id)
        
        # 4. Viral hooks generation via LLM engine with real audio energy and custom duration bounds
        logger.info("Executing LLM generate request for clip boundaries coordinates mapping")
        with ProgressHeartbeat(video_id, "Analyzing", 66, 75, 90, "AI is selecting clip boundaries"):
            clips = llm_engine.generate_viral_hooks(
                top_chunks,
                user_style=user_style,
                min_duration=min_duration_seconds,
                max_duration=max_duration_seconds,
                audio_path=audio_path,
                transcript_segments=transcript_dict.get("segments", []),
                visual_score=visual_score,
                visual_hotspots=visual_hotspots,
            )
        set_progress(video_id, "Analyzing", 75, 45, "Validating clip boundaries")
        raise_if_cancelled(video_id)
        
        # 5. Hybrid scoring calculation
        sorted_clips = []
        for clip in clips:
            clip.final_hybrid_score = float(clip.final_hybrid_score or clip.llm_virality_score)
            sorted_clips.append(clip)
            
        sorted_clips.sort(key=lambda x: x.final_hybrid_score, reverse=True)

        if not sorted_clips:
            raise RuntimeError(
                "AI did not return any valid clips. Check the LLM provider/model configuration and retry."
            )
        
        # 6. Save results to database and update status
        db = SessionLocal()
        try:
            video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
            if not video_job:
                raise ValueError(f"VideoJob with id {video_id} not found in database.")
                
            os.makedirs("clips", exist_ok=True)
            set_progress(video_id, "Cutting", 78, max(10, len(sorted_clips) * 15), "Generating MP4 clips")
            generated_paths = generate_all_clips_parallel(
                sorted_clips,
                proxy_video_path,
                "clips",
                max_workers=4,
                filename_prefix=f"job_{video_id}_clip",
                job_id=video_id,
                progress_callback=lambda completed, total: set_progress(
                    video_id,
                    "Cutting",
                    78 + int(18 * completed / max(1, total)),
                    max(1, int((total - completed) * 8)),
                    f"Generated {completed}/{total} clips",
                ),
            )
            raise_if_cancelled(video_id)
            logger.info(f"Generated {len(generated_paths)}/{len(sorted_clips)} clips in parallel.")
            if len(generated_paths) != len(sorted_clips):
                raise RuntimeError(
                    f"Only {len(generated_paths)} of {len(sorted_clips)} clips could be rendered."
                )
            set_progress(video_id, "Cutting", 98, 3, "Saving results")

            for idx, clip in enumerate(sorted_clips):
                clip_filename = f"job_{video_id}_clip_{idx}.mp4"
                clip_types_json = json.dumps(getattr(clip, "clip_types", ["educational"]))
                db_clip = ClipResult(
                    video_job_id=video_id,
                    clip_path=f"clips/{clip_filename}",
                    hybrid_score=clip.final_hybrid_score,
                    start_time=clip.start_time_seconds,
                    end_time=clip.end_time_seconds,
                    subject_hint=clip.subject_hint,
                    clip_type=clip_types_json,
                    reason=clip.reason,
                    transcript_text=_transcript_for_range(
                        transcript_dict.get("segments", []),
                        clip.start_time_seconds,
                        clip.end_time_seconds,
                    )
                )
                db.add(db_clip)

                
            video_job.status = "COMPLETED"
            db.commit()
            set_progress(video_id, "Completed", 100, 0, f"Generated {len(sorted_clips)} clips")
            logger.info(f"Saved {len(sorted_clips)} clips for Video ID {video_id} in database and marked job as COMPLETED.")
        except Exception as db_exc:
            db.rollback()
            logger.error(f"Failed to save clip results to DB for Video ID {video_id}: {str(db_exc)}")
            raise db_exc
        finally:
            db.close()

        # 7. Cleanup downloaded media files (do not delete local uploaded files)
        if not url.startswith("file://"):
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned up audio file: {audio_path}")
                except Exception as cleanup_exc:
                    logger.warning(f"Could not delete audio file {audio_path}: {str(cleanup_exc)}")
                    
            if proxy_video_path and os.path.exists(proxy_video_path):
                try:
                    os.remove(proxy_video_path)
                    logger.info(f"Cleaned up proxy video file: {proxy_video_path}")
                except Exception as cleanup_exc:
                    logger.warning(f"Could not delete proxy video file {proxy_video_path}: {str(cleanup_exc)}")

        logger.info(f"GPU pipeline execution completed successfully. Obtained {len(sorted_clips)} valid clip boundaries.")
        return [c.model_dump() for c in sorted_clips]
        
    except JobCancelled:
        db = SessionLocal()
        try:
            video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
            if video_job:
                video_job.status = "CANCELLED"
                db.commit()
        finally:
            db.close()
        set_progress(video_id, "Cancelled", 100, 0, "Stopped by user")
        return []
    except Exception as exc:
        logger.error(f"Error encountered inside GPU processing workflow for Video ID {video_id}: {str(exc)}")
        # Update job status to FAILED in database
        db = SessionLocal()
        try:
            video_job = db.query(VideoJob).filter(VideoJob.id == video_id).first()
            if video_job:
                video_job.status = "FAILED"
                db.commit()
        except Exception as db_exc:
            db.rollback()
            logger.error(f"Failed to mark Video ID {video_id} as FAILED: {str(db_exc)}")
        finally:
            db.close()
            
        # Automatic task retry behavior countdown parameter 10s
        raise self.retry(exc=exc, countdown=10)
