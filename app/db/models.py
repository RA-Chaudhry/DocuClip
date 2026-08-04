import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base

# VideoJob table jo tracking status metadata store karti hai
class VideoJob(Base):
    __tablename__ = "video_jobs"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    task_id = Column(String, nullable=True)
    min_duration_seconds = Column(Float, default=25.0)
    max_duration_seconds = Column(Float, default=75.0)
    video_quality = Column(Integer, default=480)

    # Video ke generate hue clips tracking ke liye relationship
    clips = relationship("ClipResult", back_populates="video_job", cascade="all, delete-orphan")


# ClipResult table jo generated clips segment file details and metrics save karegi
class ClipResult(Base):
    __tablename__ = "clip_results"

    id = Column(Integer, primary_key=True, index=True)
    video_job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    clip_path = Column(String, nullable=False)
    hybrid_score = Column(Float, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    subject_hint = Column(String, nullable=True)
    clip_type = Column(String, nullable=True, default="['educational']")
    reason = Column(String, nullable=True)
    transcript_text = Column(String, nullable=True)

    # Feedback learning loop metrics
    views = Column(Integer, default=0)
    avg_watch_time = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    likes = Column(Integer, default=0)

    # Back relation to identify matching VideoJob object context
    video_job = relationship("VideoJob", back_populates="clips")
