import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _boundary(clip_data, name: str) -> float:
    if isinstance(clip_data, dict):
        return float(clip_data[name])
    return float(getattr(clip_data, name))


def cut_single_clip(clip_index, clip_data, video_path, output_dir="clips", filename_prefix="clip", job_id=None):
    """Cut one clip. FFmpeg uses one thread because four jobs run in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    start = _boundary(clip_data, "start_time_seconds")
    end = _boundary(clip_data, "end_time_seconds")
    duration = max(1.0, end - start)
    output_path = os.path.join(output_dir, f"{filename_prefix}_{clip_index}.mp4")

    command = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-threads", "1",
        "-c:a", "aac",
        "-shortest",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while process.poll() is None:
        if job_id is not None:
            from app.services.progress import is_cancel_requested, JobCancelled
            if is_cancel_requested(job_id):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise JobCancelled(f"Job {job_id} cancelled during clip generation")
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:] or "FFmpeg failed")
    return output_path


def generate_all_clips_parallel(clips_list, video_path, output_dir="clips", max_workers=4, filename_prefix="clip", job_id=None, progress_callback=None):
    """Generate all clips concurrently; one failed clip does not cancel others."""
    if not clips_list or not video_path or not os.path.exists(video_path):
        return {}

    results = {}
    worker_count = min(max_workers, len(clips_list))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(cut_single_clip, index, clip, video_path, output_dir, filename_prefix, job_id): index
            for index, clip in enumerate(clips_list)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                print(f"Clip {index} generation failed: {exc}")
            if progress_callback:
                progress_callback(len(results), len(clips_list))
    return results
