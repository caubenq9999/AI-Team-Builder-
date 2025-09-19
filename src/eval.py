# src/eval.py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def plot_calibration_curve(y_true, y_prob, title="Calibration Curve"):
    """Vẽ biểu đồ calibration (reliability diagram)."""
    if y_prob is None:
        print(f"Bỏ qua calibration plot cho '{title}' vì không có y_prob.")
        return

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability in each bin")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_and_plot(log_file="reports/run_logs.json"):
    """
    Đọc file log tổng hợp, hiển thị kết quả và vẽ các biểu đồ cần thiết.
    """
    # 1. Load log
    with open(log_file, "r", encoding="utf-8") as f:
        run_logs = json.load(f)

    # 2. Tổng hợp kết quả vào DataFrame để dễ nhìn
    summary_data = []
    for label, metrics in run_logs.items():
        summary_data.append({
            "Dimension": label,
            "Val Accuracy": metrics.get("val_acc"),
            "Val F1-score": metrics.get("val_f1"),
            "Test Accuracy": metrics.get("test_acc"),
            "Test F1-score": metrics.get("test_f1"),
        })
    summary_df = pd.DataFrame(summary_data)
    print("--- Bảng tổng hợp kết quả ---")
    print(summary_df.to_string(index=False))
    print("-" * 30)

    # 3. Phân tích và vẽ biểu đồ cho từng khía cạnh
    for label, metrics in run_logs.items():
        print(f"\n--- Phân tích chi tiết cho [{label}] ---")
        checkpoint = metrics.get("checkpoint", {})
        y_true = checkpoint.get("y_true")
        y_pred = checkpoint.get("y_pred")
        y_prob = checkpoint.get("y_prob")

        if y_true is None or y_pred is None:
            print(f"Không có dữ liệu dự đoán ('y_true', 'y_pred') cho {label}.")
            continue

        # Lấy xác suất của lớp 1 (positive class)
        # y_prob có thể là list của list [[prob_0, prob_1], ...]
        if y_prob and isinstance(y_prob[0], list) and len(y_prob[0]) > 1:
            y_prob_positive = [p[1] for p in y_prob]
        else:
            y_prob_positive = None # Logistic Regression có thể không trả về proba

        # In classification report
        print("Classification Report trên tập Test:")
        print(classification_report(y_true, y_pred))

        # Vẽ Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix cho [{label}]")
        plt.show()

        # Vẽ Calibration Curve
        if y_prob_positive:
            plot_calibration_curve(y_true, y_prob_positive, f"Calibration Curve cho [{label}]")

    return summary_df