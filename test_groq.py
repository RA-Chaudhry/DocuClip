import os
import sys
from app.core.config import settings
from app.services.llm_engine import generate_viral_hooks

def main():
    print("=" * 50)
    print("DocuClip - Groq API Setup Checker")
    print("=" * 50)

    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    model = settings.GROQ_MODEL or "llama-3.3-70b-versatile"

    print(f"Configured Model: {model}")
    
    if not api_key:
        print("[STATUS] GROQ_API_KEY is EMPTY in .env file.")
        print("[INFO] The system is currently defaulting to local Ollama fallback.")
        print("[ACTION NEEDED] To use Groq API, add your key in .env:")
        print("         GROQ_API_KEY=\"gsk_your_actual_groq_api_key_here\"")
        return

    print(f"[STATUS] GROQ_API_KEY found: {api_key[:6]}...{api_key[-4:]}")
    print("[TEST] Testing Groq API connection...")

    sample_chunks = [
        {
            "start": 0.0,
            "end": 60.0,
            "text": "Welcome back! Today we are going to explore how AI is completely transforming long-form content editing. Most creators waste hours scrubbing through videos to find viral moments. But what if a single AI algorithm could extract the exact emotional peaks and story arcs automatically? Imagine boosting your viewer watch time overnight without changing your workflow."
        },
        {
            "start": 60.0,
            "end": 120.0,
            "text": "Here is the exact secret: by preserving the natural rhythm and complete sentence structure of a speaker, the AI eliminates jarring mid-sentence cuts. Viewers stay hooked from the first three-second hook all the way to the logical conclusion of the idea, driving massive retention on Shorts, TikTok, and Reels."
        }
    ]

    try:
        clips = generate_viral_hooks(sample_chunks, user_style="fast_paced")
        if clips:
            print(f"[SUCCESS] Groq API returned {len(clips)} clip suggestion(s):")
            for idx, clip in enumerate(clips, 1):
                print(f"  Clip #{idx}: {clip.start_time_seconds}s - {clip.end_time_seconds}s | Score: {clip.llm_virality_score} | Reason: {clip.reason}")
        else:
            print("[WARNING] Groq API executed but returned no valid clips.")
    except Exception as e:
        print(f"[ERROR] Groq API test failed: {e}")

if __name__ == "__main__":
    main()
