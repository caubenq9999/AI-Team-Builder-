# src/train.py
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.models import get_classifier
import joblib

def encode_dataset(model, df):
    """Encode text posts thành vector embeddings"""
    return model.encode(df["posts"].tolist(), show_progress_bar=True)

def train_and_log(model, df_train, df_val, df_test, label, log_path, seed=42):
    """
    Train + log kết quả cho từng label MBTI.
    Tự động gọi get_classifier(label).
    """
    # --- 1. Encode data ---
    X_train, y_train = encode_dataset(model, df_train), df_train[label].values
    X_val, y_val = encode_dataset(model, df_val), df_val[label].values
    X_test, y_test = encode_dataset(model, df_test), df_test[label].values

    # --- 2. Init classifier ---
    clf = get_classifier(label, seed=seed)  # nhớ sửa models.py cho khớp
    clf.fit(X_train, y_train)

    # --- 3. Evaluate ---
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_test_prob = clf.predict_proba(X_test).tolist()
    else:
        y_test_prob = None

    results = {
        "val_acc": float(accuracy_score(y_val, y_val_pred)),
        "val_f1": float(f1_score(y_val, y_val_pred, average="macro")),
        "test_acc": float(accuracy_score(y_test, y_test_pred)),
        "test_f1": float(f1_score(y_test, y_test_pred, average="macro")),
        "checkpoint": {
            "y_true": y_test.tolist(),
            "y_pred": y_test_pred.tolist(),
            "y_prob": y_test_prob
        }
    }

    # --- 4. Save log ---
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{label}] Done. Val Acc={results['val_acc']:.3f}, Test Acc={results['test_acc']:.3f}")
    model_save_path = f"models/clf_{label}.joblib"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"Đã lưu model cho [{label}] tại {model_save_path}")
    return results
