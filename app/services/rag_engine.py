import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL

# Transcript data ko 3-minute (180 seconds) chunks me sliding window aur 30s overlap ke sath break karne ki logic
def chunk_transcript(transcript_dict: dict, window_seconds: int = 180, overlap_seconds: int = 30) -> list[dict]:
    """
    Continuous segments ko window_seconds ki continuous intervals me 
    overlap_seconds context window ke sath group kiya jata hai boundary integrity ke liye.
    """
    segments = transcript_dict.get("segments", [])
    if not segments:
        return []
        
    chunks = []
    current_window = []
    
    for seg in segments:
        current_window.append(seg)
        window_duration = seg["end"] - current_window[0]["start"]
        
        if window_duration >= window_seconds:
            chunk_text = " ".join(s.get("text", "").strip() for s in current_window if s.get("text"))
            chunks.append({
                "start": current_window[0]["start"],
                "end": current_window[-1]["end"],
                "text": chunk_text
            })
            
            # Slide window keeping overlap context
            target_start = seg["end"] - overlap_seconds
            while current_window and current_window[0]["end"] <= target_start:
                current_window.pop(0)
                
    if current_window:
        chunk_text = " ".join(s.get("text", "").strip() for s in current_window if s.get("text"))
        start_time = current_window[0]["start"]
        end_time = current_window[-1]["end"]
        if not chunks or (start_time != chunks[-1]["start"] or end_time != chunks[-1]["end"]):
            chunks.append({
                "start": start_time,
                "end": end_time,
                "text": chunk_text
            })
            
    return chunks


# Target search concepts ke mapping vector similarity compute karne ke liye function
def get_top_chunks(chunks: list[dict], target_concept: str, top_k: int = 15) -> list[dict]:
    """
    SentenceTransformer model ('all-MiniLM-L6-v2') load karke input target concept
    aur transcript segments chunks ke beech cosine similarity output calculate karein.
    """
    if not chunks:
        return []
        
    # SentenceTransformer model lightweight loading local machine speed ke liye
    model = get_embedding_model()
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Text vectors encode metrics
    chunk_embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    concept_embedding = model.encode(target_concept, convert_to_numpy=True, show_progress_bar=False)
    
    similarities = []
    concept_norm = np.linalg.norm(concept_embedding)
    
    for emb, chunk in zip(chunk_embeddings, chunks):
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0 or concept_norm == 0:
            sim = 0.0
        else:
            # Cosine similarity formula: dot product divided by magnitude product
            sim = np.dot(emb, concept_embedding) / (emb_norm * concept_norm)
            
        chunk_with_score = chunk.copy()
        chunk_with_score["similarity_score"] = float(sim)
        similarities.append(chunk_with_score)
        
    # similarity_score high settings descending filters check
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return similarities[:top_k]
