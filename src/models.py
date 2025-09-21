from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def get_model(model_name):
    """Tải model embedding."""
    return SentenceTransformer(model_name)

def get_classifier(config, seed=42):
    """
    Tạo classifier dựa trên file config.
    """
    # Lấy loại classifier từ config, mặc định là 'logreg' nếu không có
    clf_type = config.get("type", "logreg").lower()
    # Lấy các tham số của classifier từ config
    params = config.get("params", {})

    if clf_type == "logreg":
        # Thêm random_state vào params để đảm bảo tính nhất quán
        params['random_state'] = seed
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Classifier type '{clf_type}' không được hỗ trợ.")

