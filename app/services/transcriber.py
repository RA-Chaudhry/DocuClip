import os
import re
import email
from email import policy
from faster_whisper import WhisperModel
from app.core.config import settings

_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # RAM OPTIMIZATION 1: cpu_threads aur num_workers limit karo
        # RAM OPTIMIZATION 2: download_root set karo taake model cache disk par rahe
        _WHISPER_MODEL = WhisperModel(
            settings.WHISPER_MODEL or "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=2,                # CPU threads limit (RAM bachata hai)
            num_workers=1,                # Parallel workers 1 (RAM bachata hai)
            download_root="/tmp/whisper_cache"  # Model RAM ki jagah disk par cache
        )
    return _WHISPER_MODEL


# Parse time string like "00:00:17,750" or "00:01:23,456" into seconds (float)
def parse_time_str(time_str: str) -> float:
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        return float(parts[0])


def parse_mhtml_transcript(mhtml_path: str) -> dict:
    try:
        with open(mhtml_path, 'rb') as f:
            msg = email.message_from_bytes(f.read(), policy=policy.default)
            
        title = msg['Subject'] or "Video Segment"
        # Find the html part
        html_content = ""
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                html_payload = part.get_payload(decode=True)
                if html_payload:
                    html_content = html_payload.decode('utf-8', errors='ignore')
                    break
                    
        # Parse figcaptions to get slide timestamps
        segments_list = []
        pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*[\u2013-]\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")
        
        figcaptions = re.findall(r"<figcaption>(.*?)</figcaption>", html_content)
        for idx, fig in enumerate(figcaptions):
            match = pattern.search(fig)
            if match:
                start_sec = parse_time_str(match.group(1))
                end_sec = parse_time_str(match.group(2))
                segments_list.append({
                    "start": start_sec,
                    "end": end_sec,
                    "text": f"Slide {idx + 1} from {title}. Key visual details and context summary.",
                    "words": []
                })
                
        if not segments_list:
            # Fallback if no slides found
            segments_list.append({
                "start": 0.0,
                "end": 60.0,
                "text": f"Overview segment for {title}.",
                "words": []
            })
            
        duration = segments_list[-1]["end"] if segments_list else 60.0
        
        return {
            "language": "en",
            "language_probability": 1.0,
            "duration": duration,
            "segments": segments_list
        }
    except Exception as e:
        print(f"Error parsing MHTML transcript: {str(e)}")
        return {
            "language": "en",
            "language_probability": 1.0,
            "duration": 60.0,
            "segments": [{
                "start": 0.0,
                "end": 60.0,
                "text": "Fallback segment for video.",
                "words": []
            }]
        }


# transcribe_audio function audio file path read karegi aur transcription segments complete detail ke sath return karegi
def transcribe_audio(audio_path: str) -> dict:
    """
    faster-whisper model utilize karke transcription execute karna.
    Word-level timestamps enable kiye gaye hain RAG boundary validation support karne ke liye.
    Supports parsing storyboard MHTML files if passed as the audio path.
    """
    if audio_path and audio_path.lower().endswith('.mhtml'):
        return parse_mhtml_transcript(audio_path)

    # 1. Initialize WhisperModel base configuration matching testing capability limits
    # Compute type int8 laptop and local CPU environment running optimization ke liye useful hai
    model = _get_whisper_model()
    
    # 2. Audio file transcription start format output parameters ke sath
    # word_timestamps=True details include segment word list sequences details
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    
    segments_list = []
    for segment in segments:
        words_list = []
        if segment.words:
            for w in segment.words:
                words_list.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": w.probability
                })
        
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": words_list
        })
        
    # Return formatted result dictionary context
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments_list
    }