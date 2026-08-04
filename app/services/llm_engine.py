import json
import os
import wave
import numpy as np
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.config import settings

# Hook Trigger Keywords for rule-based hook validation
HOOK_TRIGGERS = [
    "why", "how", "what", "secret", "never", "stop", "imagine", "nobody", "truth",
    "hidden", "crazy", "warning", "mistake", "happen", "discover", "shocking", "game changer"
]


# ClipBoundaries schema mapping model boundaries, multi-label categorization, and multi-factor virality scores
class ClipBoundaries(BaseModel):
    start_time_seconds: float
    end_time_seconds: float
    subject_hint: str
    clip_types: List[str] = Field(default_factory=lambda: ["educational"], description="List of types: controversy, emotional, educational, story, shock")
    hook_score: float = Field(default=8.0, description="Hook strength score 0-10")
    emotional_score: float = Field(default=7.5, description="Emotional impact score 0-10")
    curiosity_score: float = Field(default=8.0, description="Curiosity gap score 0-10")
    llm_virality_score: float = Field(default=8.0, description="Overall LLM virality score 0-10")
    reason: str
    final_hybrid_score: float = 0.0


# LLMResponse schema matching outer JSON array structure requirement
class LLMResponse(BaseModel):
    clips: List[ClipBoundaries]


def _fallback_clips(top_chunks: list[dict], min_duration: float, max_duration: float) -> list[ClipBoundaries]:
    """Create safe transcript windows when an external LLM is unavailable."""
    if not top_chunks:
        return []
    target_duration = min(max_duration, max(min_duration, 45.0))
    fallback = []
    for chunk in top_chunks:
        start = float(chunk.get("start", 0.0))
        chunk_end = float(chunk.get("end", start))
        end = min(chunk_end, start + target_duration)
        if end - start < min_duration * 0.8:
            continue
        fallback.append(ClipBoundaries(
            start_time_seconds=start,
            end_time_seconds=end,
            subject_hint="Transcript highlight",
            clip_types=["educational"],
            hook_score=7.0,
            emotional_score=7.0,
            curiosity_score=7.0,
            llm_virality_score=7.0,
            reason="Fallback transcript window used because the AI provider was unavailable.",
            final_hybrid_score=7.0,
        ))
        if len(fallback) >= 5:
            break
    return fallback


def calculate_real_audio_energy(top_chunks: list[dict], start_time: float, end_time: float, audio_path: Optional[str] = None) -> float:
    """
    Calculates actual audio energy score (0.0 to 10.0) using speech cadence (WPM), 
    silence gap ratio, and waveform RMS amplitude variance if audio track is available.
    """
    duration = max(1.0, end_time - start_time)
    matched_text = get_text_in_range(top_chunks, start_time, end_time)
    words = matched_text.strip().split()
    word_count = len(words)
    
    # Calculate Words Per Minute (WPM)
    wpm = (word_count / duration) * 60.0
    
    # Optimal viral speech rate is 130–200 WPM
    if 130.0 <= wpm <= 200.0:
        cadence_score = 9.0
    elif 100.0 <= wpm < 130.0 or 200.0 < wpm <= 230.0:
        cadence_score = 7.5
    else:
        cadence_score = 5.5
        
    # Audio waveform RMS calculation if audio_path exists
    rms_score = 7.5
    if audio_path and os.path.exists(audio_path):
        try:
            with wave.open(audio_path, 'rb') as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                sampwidth = wf.getsampwidth()
                start_frame = int(start_time * framerate)
                end_frame = min(nframes, int(end_time * framerate))
                
                if start_frame < end_frame:
                    wf.setpos(start_frame)
                    frames = wf.readframes(end_frame - start_frame)
                    dtype = np.int16 if sampwidth == 2 else np.uint8
                    audio_data = np.frombuffer(frames, dtype=dtype)
                    if len(audio_data) > 0:
                        rms = np.sqrt(np.mean(audio_data.astype(float)**2))
                        rms_score = min(10.0, max(4.0, (rms / 32768.0) * 20.0 + 5.0))
        except Exception:
            rms_score = 7.5
            
    final_av_score = (cadence_score * 0.6) + (rms_score * 0.4)
    return round(float(min(10.0, max(0.0, final_av_score))), 2)


def predict_retention_curve(top_chunks: list[dict], start_time: float, end_time: float) -> (bool, float):
    """
    Predicts viewer retention behavior and drop-off risks across clip timeline.
    Returns (retention_passed, retention_score).
    """
    duration = end_time - start_time
    if duration < 10.0:
        return True, 7.0
        
    # Check early segment (0-10s) hook retention
    early_text = get_text_in_range(top_chunks, start_time, start_time + 10.0)
    early_hook_val = detect_curiosity_or_shock(early_text)
    
    # Check middle segment pacing dip risk
    mid_text = get_text_in_range(top_chunks, start_time + 10.0, max(start_time + 11.0, end_time - 10.0))
    mid_words = len(mid_text.split())
    mid_duration = max(1.0, duration - 20.0)
    mid_wpm = (mid_words / mid_duration) * 60.0
    
    retention_score = 8.0
    if early_hook_val < 5.5:
        retention_score -= 2.0
    if mid_wpm < 80.0:
        retention_score -= 2.0
        
    passed = retention_score >= 5.5
    return passed, round(float(retention_score), 2)


def get_text_in_range(top_chunks: list[dict], start_time: float, end_time: float) -> str:
    """Extract transcript text snippet between start_time and end_time across chunks."""
    matched_texts = []
    for chunk in top_chunks:
        if chunk["end"] >= start_time and chunk["start"] <= end_time:
            matched_texts.append(chunk["text"])
    return " ".join(matched_texts).strip()


def detect_curiosity_or_shock(first_few_sec_text: str) -> float:
    """Calculates a numerical hook strength score (0.0 to 10.0) based on text triggers."""
    if not first_few_sec_text:
        return 5.0
    text_lower = first_few_sec_text.lower()
    score = 6.0
    
    for trigger in HOOK_TRIGGERS:
        if trigger in text_lower:
            score += 1.0
            
    if "?" in first_few_sec_text or "!" in first_few_sec_text:
        score += 1.5
        
    return min(10.0, score)


def has_strong_hook(top_chunks: list[dict], start_time: float, min_threshold: float = 5.5) -> (bool, float):
    """
    Validates if the first 4 seconds of text has a strong hook.
    Returns (is_strong, calculated_hook_score).
    """
    first_few_sec_text = get_text_in_range(top_chunks, start_time, start_time + 4.0)
    hook_score = detect_curiosity_or_shock(first_few_sec_text)
    return hook_score >= min_threshold, hook_score


def has_strong_ending(top_chunks: list[dict], end_time: float) -> bool:
    """
    Checks if the last 4 seconds of text ends cleanly without cut-off conjunctions.
    """
    last_few_sec_text = get_text_in_range(top_chunks, end_time - 4.0, end_time)
    if not last_few_sec_text:
        return True
    words = last_few_sec_text.strip().split()
    last_word = words[-1].lower().strip(".,!?") if words else ""
    trailing_conjunctions = ["and", "because", "the", "so", "but", "or", "if", "when", "that"]
    if last_word in trailing_conjunctions:
        return False
    return True


def align_clip_end_to_transcript(
    clip: ClipBoundaries,
    transcript_segments: list[dict],
    maximum_duration: float,
) -> None:
    """Move a clip end to a nearby Whisper segment/sentence boundary."""
    if not transcript_segments:
        return

    original_end = clip.end_time_seconds
    for index, segment in enumerate(transcript_segments):
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if segment_start >= original_end or segment_end <= original_end:
            continue

        candidate_end = segment_end
        text = str(segment.get("text", "")).strip()
        next_index = index + 1
        while text and not text.endswith((".", "!", "?", "。", "！", "？")) and next_index < len(transcript_segments):
            next_segment = transcript_segments[next_index]
            next_end = float(next_segment.get("end", candidate_end))
            if next_end - clip.start_time_seconds > maximum_duration:
                break
            text = f"{text} {str(next_segment.get('text', '')).strip()}".strip()
            candidate_end = next_end
            next_index += 1

        if candidate_end - clip.start_time_seconds <= maximum_duration:
            clip.end_time_seconds = candidate_end
        return


def calculate_virality_formula(clip: ClipBoundaries, detected_hook_score: float, audio_visual_score: float = 8.0) -> float:
    """
    Calculates final weighted virality score based on real virality formula:
    VIRAL SCORE = Hook Strength (35%) + Emotional Spike (30%) + Curiosity Gap (20%) + Audio/Visual Energy (15%)
    """
    h_score = max(clip.hook_score, detected_hook_score)
    e_score = clip.emotional_score
    c_score = clip.curiosity_score
    av_score = audio_visual_score
    
    final_score = (h_score * 0.35) + (e_score * 0.30) + (c_score * 0.20) + (av_score * 0.15)
    return round(float(final_score), 2)


def filter_duplicate_topics(clips: list[ClipBoundaries], similarity_threshold: float = 0.85) -> list[ClipBoundaries]:
    """
    Filter out duplicate/redundant clips covering the exact same topic angle.
    Uses sentence embeddings cosine similarity on subject_hint + reason + clip_types.
    """
    if len(clips) <= 1:
        return clips
        
    try:
        from sentence_transformers import SentenceTransformer
        from app.services.rag_engine import get_embedding_model
        model = get_embedding_model()
        texts = [f"{c.subject_hint} {c.reason} {' '.join(c.clip_types)}" for c in clips]
        embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        
        unique_clips = []
        for i, clip in enumerate(clips):
            is_dup = False
            for j in range(len(unique_clips)):
                prev_clip = unique_clips[j]
                if abs(clip.start_time_seconds - prev_clip.start_time_seconds) > 15.0:
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
                        if sim > similarity_threshold:
                            print(f"Filtering duplicate topic clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s) with similarity {sim:.2f}.")
                            is_dup = True
                            break
            if not is_dup:
                unique_clips.append(clip)
        return unique_clips
    except Exception as e:
        print(f"Topic diversity filtering fallback: {str(e)}")
        return clips


def merge_overlapping_clips(clips: list[ClipBoundaries], max_duration_limit: float = 90.0, overlap_threshold_seconds: float = 5.0) -> list[ClipBoundaries]:
    """
    Merge adjacent or heavily overlapping clips that cover the same core topic/story arc.
    """
    if not clips:
        return []
        
    sorted_by_start = sorted(clips, key=lambda c: c.start_time_seconds)
    merged = []
    
    for clip in sorted_by_start:
        if not merged:
            merged.append(clip)
            continue
            
        last = merged[-1]
        if clip.start_time_seconds <= (last.end_time_seconds + overlap_threshold_seconds):
            new_end = max(last.end_time_seconds, clip.end_time_seconds)
            combined_duration = new_end - last.start_time_seconds
            
            if combined_duration <= max_duration_limit:
                last.end_time_seconds = new_end
                last.llm_virality_score = max(last.llm_virality_score, clip.llm_virality_score)
                combined_types = list(set(last.clip_types + clip.clip_types))
                last.clip_types = combined_types
                last.reason = f"{last.reason} | Merged: {clip.reason}"
                continue
                
        merged.append(clip)
        
    return merged


def generate_viral_hooks(
    top_chunks: list[dict], 
    user_style: str = "fast_paced", 
    min_duration: float = 25.0, 
    max_duration: float = 75.0,
    audio_path: Optional[str] = None,
    transcript_segments: Optional[list[dict]] = None,
    visual_score: Optional[float] = None,
    visual_hotspots: Optional[list[dict]] = None,
) -> list[ClipBoundaries]:
    """
    Generate viral clips using Groq API when configured, with a deterministic
    transcript fallback when the remote provider is unavailable.
    Enforces user custom or retention default duration bounds, retention curve prediction, real audio energy,
    multi-label categorization, hook validation, topic diversity filtering, and multi-factor virality formula.
    """
    if not top_chunks:
        return []
        
    if user_style == "fast_paced":
        style_instruction = "Prioritize high-energy hooks, dramatic spikes, curiosity gaps, and fast-moving points."
    elif user_style == "educational":
        style_instruction = "Prioritize clear insight delivery, logical step-by-step breakdowns, and strong educational takeaways."
    else:
        style_instruction = "Identify highly engaging standalone segments with compelling emotional or informational payoff."
        
    system_prompt = (
        f"You are a master viral video editor AI. Analyze the transcript context and extract "
        f"all high-quality viral clips matching requested style: {user_style}. {style_instruction}\n\n"
        f"STRICT EDITING & CONTEXT INTEGRITY RULES:\n"
        f"1. DURATION: Target clip duration must be between {min_duration:.1f} seconds and {max_duration:.1f} seconds. "
        f"DO NOT output clips under {min_duration * 0.8:.1f} seconds or over {max_duration * 1.2:.1f} seconds.\n"
        f"2. COMPLETE CONTEXT & RHYTHM INTEGRITY: Every clip MUST begin at the natural start of a sentence or thought, "
        f"and end ONLY after the speaker fully completes the statement, explanation, or emotional payoff. "
        f"NEVER cut off mid-sentence, mid-thought, or mid-story.\n"
        f"3. HOOK & VIRALITY: The first 3-5 seconds MUST contain a strong hook, question, or curiosity gap. "
        f"Classify each clip with one or more relevant clip_types from: ['controversy', 'emotional', 'educational', 'story', 'shock'].\n"
        f"4. MULTI-FACTOR SCORES: Provide sub-scores from 0.0 to 10.0 for hook_score, emotional_score, curiosity_score, and llm_virality_score.\n"
        f"5. QUANTITY & QUALITY: Extract ALL high-quality clips naturally present in the text. "
        f"Do NOT artificially cap clip count, but DO NOT output weak, filler, or redundant clips.\n\n"
        f"Output JSON strictly with top-level key 'clips' matching this schema:\n"
        f"clips: list of objects each having keys: start_time_seconds (float), end_time_seconds (float), "
        f"subject_hint (str), clip_types (list of str), hook_score (float), emotional_score (float), curiosity_score (float), "
        f"llm_virality_score (float 0-10), reason (str)."
    )
    if visual_hotspots:
        system_prompt += (
            "\n6. VISUAL ALIGNMENT: Prefer transcript moments that overlap the supplied visual hotspots. "
            "Use hotspot timestamps as evidence of scene changes or high motion; do not invent timestamps."
        )
    
    response_text = None
    compact_chunks = [
        {
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": str(chunk.get("text", ""))[:1400],
        }
        for chunk in top_chunks[:8]
    ]
    visual_hotspots = visual_hotspots or []
    llm_context = {
        "transcript_chunks": compact_chunks,
        "visual_hotspots": visual_hotspots[:10],
    }
    
    # 1. Try Groq API if API key is set in settings or env
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if api_key:
        try:
            print("Invoking Groq API service for LLM generation...")
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL or "llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript and visual context:\n{json.dumps(llm_context, ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"}
            )
            response_text = completion.choices[0].message.content
        except Exception as groq_exc:
            print(f"Groq API call failed: {str(groq_exc)}. Falling back to transcript windows...")
            response_text = None

    # 2. No local LLM is used: this laptop runs CPU-only. Use deterministic
    # transcript windows if Groq is unavailable or not configured.
    if not response_text:
        print("Groq unavailable or not configured; using transcript fallback clips.")
        return _fallback_clips(top_chunks, min_duration, max_duration)

    try:
        data = json.loads(response_text)
        llm_response = LLMResponse.model_validate(data)
        
        min_bound = min(c["start"] for c in top_chunks)
        max_bound = max(c["end"] for c in top_chunks)
        
        validated_clips = []
        for clip in llm_response.clips:
            # Bound timestamps to prevent LLM hallucinations
            if clip.start_time_seconds < min_bound:
                clip.start_time_seconds = min_bound
            if clip.end_time_seconds > max_bound:
                clip.end_time_seconds = max_bound
                
            duration = clip.end_time_seconds - clip.start_time_seconds
            
            # 1. Flexible Duration Rule: Check against custom or default user bounds
            allowed_min = min_duration * 0.8
            allowed_max = max_duration * 1.2
            if duration < allowed_min or duration > allowed_max:
                print(f"Discarding clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s): Duration {duration:.1f}s outside bounds ({allowed_min:.1f}s - {allowed_max:.1f}s).")
                continue
                
            # 2. Hook Validation Check
            is_hook_strong, detected_hook_score = has_strong_hook(top_chunks, clip.start_time_seconds)
            if not is_hook_strong and clip.hook_score < 6.0:
                print(f"Discarding clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s): Weak opening hook.")
                continue
                
            # 3. Retention Drop-off Curve Prediction
            retention_passed, retention_score = predict_retention_curve(top_chunks, clip.start_time_seconds, clip.end_time_seconds)
            if not retention_passed:
                print(f"Discarding clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s): High early retention drop-off risk (Score: {retention_score}).")
                continue
                
            # 4. Ending Strength Check
            if transcript_segments:
                align_clip_end_to_transcript(clip, transcript_segments, maximum_duration=allowed_max)
            elif not has_strong_ending(top_chunks, clip.end_time_seconds):
                print(f"Keeping original ending for clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s): transcript boundary unavailable.")
                
            # 5. Real Audio Energy Scoring
            real_av_score = calculate_real_audio_energy(top_chunks, clip.start_time_seconds, clip.end_time_seconds, audio_path)
            if visual_score is not None:
                real_av_score = (real_av_score * 0.5) + (float(visual_score) * 0.5)
            
            # 6. Multi-Factor Virality Formula calculation
            calculated_score = calculate_virality_formula(clip, detected_hook_score, audio_visual_score=real_av_score)
            clip.final_hybrid_score = calculated_score
            
            if calculated_score < 6.5:
                print(f"Discarding clip ({clip.start_time_seconds}s - {clip.end_time_seconds}s): Final Virality Formula score {calculated_score} below threshold 6.5.")
                continue
                
            validated_clips.append(clip)
            
        # 7. Merge overlapping/adjacent clip suggestions for cohesive story flow
        merged_clips = merge_overlapping_clips(validated_clips, max_duration_limit=allowed_max)
        
        # 8. Apply Topic Diversity Filtering
        final_clips = filter_duplicate_topics(merged_clips)
        return final_clips
        
    except Exception as parse_exc:
        print(f"Exception parsing LLM response: {str(parse_exc)}")
        return _fallback_clips(top_chunks, min_duration, max_duration)
