import os
import re
import subprocess
from typing import Optional


def _ffmpeg_visual_score(video_path: str) -> float:
    """CPU-friendly visual analysis using FFmpeg filters (no OpenCV needed)."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        duration = max(1.0, float(probe.stdout.strip() or 1.0))

        scene = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", video_path, "-vf",
             "fps=1,select='gt(scene,0.30)',metadata=print", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=max(60, int(duration * 2)),
        )
        scene_cuts = len(re.findall(r"lavfi\.scene_score", scene.stderr))

        stats = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", video_path, "-vf",
             "fps=1,scale=320:-1,signalstats,metadata=print", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=max(60, int(duration * 2)),
        )
        luminance = [float(value) for value in re.findall(
            r"lavfi\.signalstats\.YAVG[:=]([0-9.]+)", stats.stderr
        )]
        avg_motion = sum(abs(b - a) for a, b in zip(luminance, luminance[1:])) / max(1, len(luminance) - 1)
        cuts_per_minute = (scene_cuts / duration) * 60.0
        scene_score = min(10.0, max(2.0, (cuts_per_minute / 20.0) * 10.0))
        motion_score = min(10.0, max(3.0, (avg_motion / 8.0) * 10.0))
        # Face detection is optional; scene changes and motion remain real
        # visual signals when OpenCV is not installed on a CPU-only machine.
        return round((scene_score * 0.55) + (motion_score * 0.45), 2)
    except Exception as exc:
        print(f"FFmpeg visual analysis fallback: {exc}")
        return 7.5


def get_top_visual_hotspots(video_path: str, count: int = 10) -> list[dict]:
    """Return the strongest visual moments from one full-video FFmpeg scan.

    Frames are sampled at 2 FPS in a single FFmpeg process. A large change in
    luminance/chroma between adjacent samples is used as a lightweight proxy
    for scene cuts and motion intensity. Nearby detections are de-duplicated.
    """
    if not video_path or not os.path.exists(video_path) or count <= 0:
        return []
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        duration = max(1.0, float(probe.stdout.strip() or 1.0))
        command = [
            "ffmpeg", "-hide_banner", "-i", video_path,
            "-vf", "fps=2,scale=320:-1,signalstats,metadata=print:file=-",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=max(90, int(duration * 2.5)),
        )
        output = f"{result.stdout}\n{result.stderr}"
        blocks = re.split(r"(?=frame:\s*\d+)", output)
        samples = []
        for block in blocks:
            time_match = re.search(r"pts_time:([0-9.]+)", block)
            if not time_match:
                continue
            values = {}
            for key in ("YAVG", "UAVG", "VAVG"):
                value_match = re.search(rf"lavfi\.signalstats\.{key}[=:]([0-9.]+)", block)
                values[key] = float(value_match.group(1)) if value_match else 0.0
            samples.append({"time": float(time_match.group(1)), "values": values})

        if len(samples) < 2:
            return []
        detections = []
        previous = samples[0]["values"]
        for sample in samples[1:]:
            current = sample["values"]
            change = (
                abs(current["YAVG"] - previous["YAVG"]) * 0.6
                + abs(current["UAVG"] - previous["UAVG"]) * 0.2
                + abs(current["VAVG"] - previous["VAVG"]) * 0.2
            )
            score = round(min(10.0, max(0.0, change / 8.0 * 10.0)), 2)
            detections.append({
                "start_time": round(max(0.0, sample["time"] - 2.0), 2),
                "end_time": round(min(duration, sample["time"] + 8.0), 2),
                "visual_score": score,
            })
            previous = current

        detections.sort(key=lambda item: item["visual_score"], reverse=True)
        hotspots = []
        for item in detections:
            if all(abs(item["start_time"] - existing["start_time"]) >= 8.0 for existing in hotspots):
                hotspots.append(item)
            if len(hotspots) >= count:
                break
        return hotspots
    except Exception as exc:
        print(f"Visual hotspot analysis fallback: {exc}")
        return []

def detect_scene_cuts(video_path: str) -> float:
    """
    Samples frames every 0.5s, computes histogram difference (cv2.compareHist),
    counts significant changes, and returns normalized scene_change_rate (0-10).
    """
    if not video_path or not os.path.exists(video_path):
        return 7.0
        
    try:
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 7.0
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 1.0
        
        sample_step = int(fps * 0.5)  # Sample frame every 0.5 seconds
        if sample_step <= 0:
            sample_step = 15
            
        prev_hist = None
        scene_cuts = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                
                if prev_hist is not None:
                    # Compare histogram correlation
                    sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if sim < 0.65:  # Histogram drop indicates scene cut
                        scene_cuts += 1
                prev_hist = hist
                
            frame_idx += 1
            
        cap.release()
        
        cuts_per_minute = (scene_cuts / max(1.0, duration_sec)) * 60.0
        normalized_score = min(10.0, max(2.0, (cuts_per_minute / 20.0) * 10.0))
        return round(float(normalized_score), 2)
    except Exception as e:
        print(f"Visual scene cuts analysis fallback: {str(e)}")
        return 7.0


def detect_motion_intensity(video_path: str) -> float:
    """
    Frame differencing (cv2.absdiff) to measure average pixel change and normalize to 0-10 score.
    """
    if not video_path or not os.path.exists(video_path):
        return 7.5
        
    try:
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 7.5
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_step = int(fps * 0.5)
        if sample_step <= 0: sample_step = 15
        
        prev_gray = None
        motion_diffs = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    mean_diff = np.mean(diff)
                    motion_diffs.append(mean_diff)
                prev_gray = gray
                
            frame_idx += 1
            
        cap.release()
        
        if not motion_diffs:
            return 7.5
            
        avg_motion = float(np.mean(motion_diffs))
        motion_score = min(10.0, max(3.0, (avg_motion / 15.0) * 10.0))
        return round(float(motion_score), 2)
    except Exception as e:
        print(f"Visual motion intensity analysis fallback: {str(e)}")
        return 7.5


def detect_face_presence(video_path: str) -> float:
    """
    Detects faces in sampled frames using OpenCV Haar Cascade and returns face presence score (0-10).
    """
    if not video_path or not os.path.exists(video_path):
        return 8.0
        
    try:
        import cv2
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            return 8.0
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 8.0
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_step = int(fps * 1.0)
        if sample_step <= 0: sample_step = 30
        
        total_samples = 0
        face_samples = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_step == 0:
                total_samples += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    face_samples += 1
                    
            frame_idx += 1
            
        cap.release()
        
        if total_samples == 0:
            return 8.0
            
        face_ratio = face_samples / total_samples
        face_score = min(10.0, max(4.0, face_ratio * 10.0 + 2.0))
        return round(float(face_score), 2)
    except Exception as e:
        print(f"Visual face presence analysis fallback: {str(e)}")
        return 8.0


def calculate_visual_score(video_path: Optional[str] = None) -> float:
    """
    Combines scene cuts, motion intensity, and face presence into a composite visual score (0-10).
    Formula: visual_score = scene_change_rate * 0.4 + motion_intensity * 0.3 + face_presence * 0.3
    """
    if not video_path or not os.path.exists(video_path):
        return 7.5
        
    # Analyze all three signals in one sampled pass. The older implementation
    # decoded the entire video three times, which made a 6-minute CPU job
    # unnecessarily slow.
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 7.5
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 1.0
        sample_step = max(1, int(fps * 0.75))
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        prev_hist = None
        prev_gray = None
        scene_cuts = 0
        motion_diffs = []
        face_samples = 0
        total_samples = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (320, 180))
                hist = cv2.calcHist([small_gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.65:
                    scene_cuts += 1
                if prev_gray is not None:
                    motion_diffs.append(float(np.mean(cv2.absdiff(prev_gray, small_gray))))
                prev_hist = hist
                prev_gray = small_gray
                total_samples += 1
                if not cascade.empty() and total_samples % 2 == 0:
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        face_samples += 1
            frame_idx += 1
        cap.release()

        cuts_per_minute = (scene_cuts / max(1.0, duration_sec)) * 60.0
        scene_cuts_score = min(10.0, max(2.0, (cuts_per_minute / 20.0) * 10.0))
        avg_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0
        motion_score = min(10.0, max(3.0, (avg_motion / 15.0) * 10.0))
        face_ratio = face_samples / max(1, total_samples // 2)
        face_score = min(10.0, max(4.0, face_ratio * 10.0 + 2.0))
    except ImportError:
        return _ffmpeg_visual_score(video_path)
    except Exception as exc:
        print(f"Combined visual analysis fallback: {exc}")
        return 7.5
    
    composite_visual = (scene_cuts_score * 0.4) + (motion_score * 0.3) + (face_score * 0.3)
    return round(float(min(10.0, max(0.0, composite_visual))), 2)
