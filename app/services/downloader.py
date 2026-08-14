import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

from app.core.config import settings


YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize_youtube_url(url: str) -> str:
    """Return one canonical URL so equivalent links share the same cache."""
    if not url:
        return url

    raw_url = url.strip()
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower().removeprefix("www.")
    video_id = None

    if host in {"youtu.be", "youtube.com", "m.youtube.com", "music.youtube.com"}:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and YOUTUBE_ID_PATTERN.match(query_id):
            video_id = query_id
        else:
            path_parts = [part for part in parsed.path.split("/") if part]
            if host == "youtu.be" and path_parts:
                candidate = path_parts[0]
                if YOUTUBE_ID_PATTERN.match(candidate):
                    video_id = candidate
            elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = path_parts[1]
                if YOUTUBE_ID_PATTERN.match(candidate):
                    video_id = candidate

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return raw_url


def generate_hash(url: str) -> str:
    """Hash the normalized URL or local file, ignoring share/tracking query parameters."""
    if not url:
        return hashlib.sha256(b"").hexdigest()
    local_path = url.removeprefix("file://") if url.startswith("file://") else url
    if os.path.exists(local_path):
        try:
            stat = os.stat(local_path)
            file_key = f"file:{os.path.abspath(local_path)}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.sha256(file_key.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(url.encode("utf-8")).hexdigest()
    normalized = normalize_youtube_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def download_media(url: str, output_dir: str = "/tmp/shared", quality: int = 480, job_id: int | None = None) -> tuple[str, str]:
    """Download audio+video once, merge them to one MP4, and return its path.

    If the url points to an existing local file or file:// URI, returns the local path directly.
    """
    # Check if local video file path
    local_path = url.removeprefix("file://") if url.startswith("file://") else url
    if os.path.exists(local_path):
        return (os.path.abspath(local_path), os.path.abspath(local_path))

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            output_dir = tempfile.gettempdir()

    normalized_url = normalize_youtube_url(url)
    output_template = os.path.join(output_dir, "%(id)s_merged.%(ext)s")
    quality = min(settings.MAX_QUALITY, max(360, int(quality)))
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--extractor-args",
        "youtube:player_client=android_vr",
        "-f",
        f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]",
        "--merge-output-format",
        "mp4",
        "--print",
        "after_move:filepath",
        "-o",
        output_template,
        normalized_url,
    ]

    try:
        # ✅ FIX: Popen does NOT support capture_output. Use stdout and stderr PIPE.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # Fixed line
            stderr=subprocess.PIPE,  # Fixed line
            text=True,
        )
        
        deadline = time.monotonic() + (30 * 60)
        while process.poll() is None:
            if job_id is not None:
                from app.services.progress import is_cancel_requested, JobCancelled
                if is_cancel_requested(job_id):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise JobCancelled(f"Job {job_id} cancelled during download")
            if time.monotonic() >= deadline:
                process.terminate()
                process.wait(timeout=5)
                raise RuntimeError("yt-dlp download timed out after 30 minutes")
            time.sleep(0.5)
            
        stdout, stderr = process.communicate()
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown yt-dlp error").strip()
            raise RuntimeError(f"yt-dlp failed with exit code {result.returncode}: {detail[-2000:]}")

        printed_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for candidate in reversed(printed_paths):
            if os.path.isfile(candidate):
                return candidate, candidate

        # Fallback for yt-dlp versions that do not print after_move:filepath.
        video_files = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.endswith(".mp4") and "_merged" in name
        ]
        if video_files:
            video_files.sort(key=os.path.getmtime, reverse=True)
            return video_files[0], video_files[0]

        raise RuntimeError(
            "yt-dlp completed without producing an MP4 file. "
            f"Output: {result.stdout[-1000:]}"
        )
    except Exception:
        raise