from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_URL = f"sqlite:///{(PROJECT_ROOT / 'docuclip.db').as_posix()}"


def _build_engine():
    database_url = (settings.DATABASE_URL or settings.POSTGRES_URL or DEFAULT_SQLITE_URL).strip()
    # Accept the URL format commonly supplied by hosting platforms.
    if database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg2://" + database_url[len("postgres://"):]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg2://" + database_url[len("postgresql://"):]

    if database_url.startswith("postgresql+"):
        try:
            engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"connect_timeout": 5},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as exc:
            print(f"PostgreSQL unavailable, falling back to SQLite: {exc}")

    return create_engine(
        DEFAULT_SQLITE_URL,
        connect_args={"check_same_thread": False},
    )


def database_status():
    """Return a small, safe diagnostic used by the health endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True, "backend": engine.dialect.name}
    except Exception as exc:
        return {"connected": False, "backend": engine.dialect.name, "error": str(exc)}


# Engine configure kiya hai database connections coordinate karne ke liye
engine = _build_engine()

# SessionLocal class transactions aur state check handles setup karne ke liye
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class mapping and database models extend karne ke liye declaratively
Base = declarative_base()


def ensure_schema():
    """Create missing tables/columns for databases created by older versions."""
    Base.metadata.create_all(bind=engine)
    required_columns = {
        "video_jobs": {
            "url": "VARCHAR",
            "status": "VARCHAR",
            "created_at": "TIMESTAMP",
            "task_id": "VARCHAR",
            "min_duration_seconds": "FLOAT",
            "max_duration_seconds": "FLOAT",
            "video_quality": "INTEGER",
        },
        "clip_results": {
            "video_job_id": "INTEGER",
            "clip_path": "VARCHAR",
            "hybrid_score": "FLOAT",
            "start_time": "FLOAT",
            "end_time": "FLOAT",
            "subject_hint": "VARCHAR",
            "clip_type": "VARCHAR",
            "reason": "VARCHAR",
            "transcript_text": "TEXT",
            "views": "INTEGER",
            "avg_watch_time": "FLOAT",
            "completion_rate": "FLOAT",
            "likes": "INTEGER",
        },
    }

    with engine.begin() as conn:
        inspector = inspect(conn)
        for table_name, columns in required_columns.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name in existing:
                    continue
                if engine.dialect.name == "postgresql":
                    conn.execute(text(
                        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS '
                        f'"{column_name}" {column_type}'
                    ))
                else:
                    conn.execute(text(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type}'
                    ))

# get_db utility: database session retrieve karne aur secure scope close karne ke liye
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
