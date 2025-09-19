# src/models.py
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def get_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def get_classifier(label, seed=42):
    """Tạo classifier cho từng label MBTI"""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed
    )
