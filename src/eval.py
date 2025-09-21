import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# --- THÊM CÁC THƯ VIỆN CẦN THIẾT ---
import joblib
from sentence_transformers import SentenceTransformer
from src.data import load_datasets


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

# --- HÀM PHÂN TÍCH LỖI MỚI (TÍCH HỢP TỪ ERROR_ANALYSIS.PY) ---
def perform_error_analysis(num_errors_to_show=15):
    """
    Tải model, chạy dự đoán trên tập test và lọc ra các mẫu bị dự đoán sai.
    """
    print("\n" + "="*50)
    print("BẮT ĐẦU PHÂN TÍCH LỖI DỰ ĐOÁN END-TO-END")
    print("="*50)

    # --- 1. Tải các model cần thiết ---
    print("Đang tải các model cần thiết cho phân tích lỗi...")
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    classifiers = {}
    dimensions = ["IE", "NS", "TF", "JP"]
    label_map = {
        "IE": {0: "I", 1: "E"}, "NS": {0: "N", 1: "S"},
        "TF": {0: "T", 1: "F"}, "JP": {0: "J", 1: "P"}
    }
    for dim in dimensions:
        model_path = f"models/clf_{dim}.joblib"
        classifiers[dim] = joblib.load(model_path)
    print("Tải model hoàn tất.")

    # --- 2. Tải dữ liệu test ---
    print("\nĐang tải và xử lý dữ liệu test...")
    _, _, test_df = load_datasets("data/mbti.csv")

    # --- 3. Thực hiện dự đoán và tìm lỗi ---
    print("Bắt đầu dự đoán trên tập test để tìm lỗi...")
    error_records = []
    for index, row in test_df.iterrows():
        text = row['posts']
        true_mbti = row['type']
        
        embedding = encoder_model.encode([text])
        predicted_mbti = ""
        for dim in dimensions:
            clf = classifiers[dim]
            pred_label = clf.predict(embedding)[0]
            mbti_char = label_map[dim][pred_label]
            predicted_mbti += mbti_char
            
        if predicted_mbti != true_mbti:
            mismatched_dims = [dimensions[i] for i, dim_char in enumerate(predicted_mbti) if dim_char != true_mbti[i]]
            error_records.append({
                "Original Text": text,
                "True MBTI": true_mbti,
                "Predicted MBTI": predicted_mbti,
                "Mismatched Dimensions": ", ".join(mismatched_dims)
            })
            if len(error_records) >= num_errors_to_show:
                break
    
    # --- 4. Hiển thị kết quả phân tích lỗi ---
    if not error_records:
        print("\nChúc mừng! Không tìm thấy lỗi nào trong các mẫu đã kiểm tra.")
        return

    print(f"\n--- Phân tích {len(error_records)} mẫu dự đoán sai đầu tiên ---")
    error_df = pd.DataFrame(error_records)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 200)
    print(error_df)
    print("="*50)
    print("KẾT THÚC PHÂN TÍCH LỖI")
    print("="*50)


def evaluate_and_plot(log_file="reports/run_logs.json"):
    """
    Đọc file log, hiển thị kết quả, vẽ biểu đồ và chạy phân tích lỗi.
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

        if y_prob and isinstance(y_prob[0], list) and len(y_prob[0]) > 1:
            y_prob_positive = [p[1] for p in y_prob]
        else:
            y_prob_positive = None

        print("Classification Report trên tập Test:")
        print(classification_report(y_true, y_pred))

        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix cho [{label}]")
        plt.show()

        if y_prob_positive:
            plot_calibration_curve(y_true, y_prob_positive, f"Calibration Curve cho [{label}]")
    
    # --- 4. GỌI HÀM PHÂN TÍCH LỖI ---
    perform_error_analysis()

    return summary_df
