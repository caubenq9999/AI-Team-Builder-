import os
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score
from src.models import get_classifier

def encode_dataset(model, df):
    """Encode text posts thành vector embeddings"""
    print(f"Encoding {len(df)} samples...")
    return model.encode(df["posts"].tolist(), show_progress_bar=True)

def train_and_log(model, df_train, df_val, df_test, label, classifier_config, log_path, seed=42):
    """
    Train + log kết quả cho từng label MBTI, nhận config cho classifier.
    """
    print(f"\n--- Bắt đầu huấn luyện cho khía cạnh [{label}] ---")
    
    # --- 1. Encode data ---
    X_train, y_train = encode_dataset(model, df_train), df_train[label].values
    X_val, y_val = encode_dataset(model, df_val), df_val[label].values
    X_test, y_test = encode_dataset(model, df_test), df_test[label].values

    # --- 2. Init classifier từ config ---
    clf = get_classifier(classifier_config, seed=seed)
    print(f"Đã khởi tạo classifier: {clf}")
    
    clf.fit(X_train, y_train)

    # --- Lưu model đã huấn luyện ---
    model_save_path = f"models/clf_{label}.joblib"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"Đã lưu model cho [{label}] tại {model_save_path}")

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
    return results

