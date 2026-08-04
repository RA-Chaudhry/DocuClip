import json
import redis
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import ClipResult

REDIS_WEIGHTS_KEY = "docuclip:virality_weights"

DEFAULT_WEIGHTS = {
    "hook": 0.25,
    "emotion": 0.20,
    "curiosity": 0.15,
    "audio": 0.15,
    "visual": 0.10,
    "retention": 0.10,
    "rewatchability": 0.05
}

def get_virality_weights() -> dict:
    """Loads dynamic virality scoring weights from Redis or returns DEFAULT_WEIGHTS."""
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        data = r.get(REDIS_WEIGHTS_KEY)
        if data:
            weights = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
            return weights
    except Exception:
        pass
    return DEFAULT_WEIGHTS.copy()

def save_virality_weights(weights: dict) -> bool:
    """Saves updated virality weights to Redis."""
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.set(REDIS_WEIGHTS_KEY, json.dumps(weights))
        return True
    except Exception as e:
        print(f"Failed to save virality weights to Redis: {str(e)}")
        return False

def update_clip_metrics(db: Session, clip_id: int, watch_time: float, completed: bool, liked: bool) -> bool:
    """Updates user engagement feedback metrics for a ClipResult entry."""
    clip = db.query(ClipResult).filter(ClipResult.id == clip_id).first()
    if not clip:
        return False
        
    current_views = clip.views or 0
    new_views = current_views + 1
    
    # Calculate updated running average watch time
    current_avg_wt = clip.avg_watch_time or 0.0
    new_avg_wt = ((current_avg_wt * current_views) + watch_time) / new_views
    
    # Calculate completion rate ratio
    current_completions = (clip.completion_rate or 0.0) * current_views
    if completed:
        current_completions += 1.0
    new_completion_rate = current_completions / new_views
    
    # Calculate likes
    new_likes = (clip.likes or 0) + (1 if liked else 0)
    
    clip.views = new_views
    clip.avg_watch_time = new_avg_wt
    clip.completion_rate = new_completion_rate
    clip.likes = new_likes
    
    db.commit()
    return True

def compute_performance_score(clip: ClipResult) -> float:
    """
    Computes performance score (0.0 to 10.0) based on real viewer metrics:
    performance_score = (completion_rate * 0.5) + (avg_watch_time_ratio * 0.3) + (likes_per_view * 0.2)
    """
    views = max(1, clip.views or 1)
    duration = max(1.0, (clip.end_time or 60.0) - (clip.start_time or 0.0))
    
    completion_rate = min(1.0, max(0.0, clip.completion_rate or 0.0))
    watch_time_ratio = min(1.0, max(0.0, (clip.avg_watch_time or 0.0) / duration))
    likes_per_view = min(1.0, max(0.0, (clip.likes or 0) / views))
    
    perf = (completion_rate * 0.5) + (watch_time_ratio * 0.3) + (likes_per_view * 0.2)
    return round(float(perf * 10.0), 2)

def adjust_virality_weights(db: Session) -> dict:
    """
    Periodically adjusts scoring weights using gradient-like updates from top performing clips.
    """
    clips = db.query(ClipResult).filter(ClipResult.views >= 5).all()
    if not clips:
        return get_virality_weights()
        
    weights = get_virality_weights()
    top_clips = [c for c in clips if compute_performance_score(c) >= 7.0]
    
    if len(top_clips) >= 3:
        # Increase weight of hook and emotion if top clips achieve high completion rate
        avg_comp = sum(c.completion_rate or 0.0 for c in top_clips) / len(top_clips)
        if avg_comp > 0.7:
            weights["hook"] = min(0.35, weights["hook"] + 0.01)
            weights["emotion"] = min(0.30, weights["emotion"] + 0.01)
            # Normalize remaining weights so sum equals 1.0
            total = sum(weights.values())
            weights = {k: round(v / total, 4) for k, v in weights.items()}
            save_virality_weights(weights)
            
    return weights
