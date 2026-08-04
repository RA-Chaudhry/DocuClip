import os
import json
import pickle
from typing import Optional, List, Dict

DEFAULT_TRAINING_DATA = [
    {"text": "Why nobody talks about this secret algorithm", "label": 1},
    {"text": "Stop scrolling if you want to double your view count", "label": 1},
    {"text": "This incredible tool completely changed my workflow", "label": 1},
    {"text": "What happens when you delete your social media", "label": 1},
    {"text": "Here is the shocking truth about viral videos", "label": 1},
    {"text": "Never make this mistake when editing videos", "label": 1},
    {"text": "Imagine boosting viewer watch time overnight", "label": 1},
    {"text": "Did you know that 90% of creators fail because of this", "label": 1},
    {"text": "Today we are going to discuss video editing concepts", "label": 0},
    {"text": "Welcome back to another episode of my podcast", "label": 0},
    {"text": "In this video I will show you some basic settings", "label": 0},
    {"text": "Let us take a look at the code structure here", "label": 0},
    {"text": "Okay so first line is import statement", "label": 0},
    {"text": "Hello everyone welcome to my channel subscribe now", "label": 0},
    {"text": "Thank you for watching see you in the next video", "label": 0}
]

MODEL_PATH = os.path.join("app", "services", "hook_classifier.pkl")
_classifier_instance = None
_embedder_instance = None

def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from sentence_transformers import SentenceTransformer
        _embedder_instance = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder_instance

def train_hook_model(dataset_path: Optional[str] = None) -> bool:
    """
    Trains a LogisticRegression hook classifier on training data and saves model locally.
    """
    global _classifier_instance
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        data = DEFAULT_TRAINING_DATA
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]
        
        embedder = get_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        
        clf = LogisticRegression()
        clf.fit(embeddings, labels)
        
        _classifier_instance = clf
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
            
        print("Successfully trained and saved ML Hook Classifier.")
        return True
    except Exception as e:
        print(f"Failed to train ML Hook Classifier: {str(e)}")
        return False

def load_hook_model():
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance
        
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _classifier_instance = pickle.load(f)
            return _classifier_instance
        except Exception:
            pass
            
    # Auto-train default model if missing
    train_hook_model()
    return _classifier_instance

def predict_hook_score(text: str) -> float:
    """
    Predicts virality hook score (0.0 to 10.0) using trained ML SentenceTransformer classifier.
    """
    if not text or not text.strip():
        return 5.0
        
    try:
        clf = load_hook_model()
        if clf is None:
            return 6.0
            
        embedder = get_embedder()
        emb = embedder.encode([text], convert_to_numpy=True)
        
        # Predict probability of being a high-converting hook
        prob = float(clf.predict_proba(emb)[0][1])
        
        # Map probability (0.0 - 1.0) to (2.0 - 10.0) score range
        score = min(10.0, max(2.0, prob * 8.0 + 2.0))
        return round(float(score), 2)
    except Exception as e:
        print(f"ML Hook score prediction fallback: {str(e)}")
        return 6.5
