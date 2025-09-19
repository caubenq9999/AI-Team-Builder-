import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from itertools import combinations
import re

# --- 1. CONFIG & TẢI CÁC MODEL CẦN THIẾT (CHỈ TẢI MỘT LẦN) ---

# Ánh xạ MBTI sang vai trò gợi ý
MBTI_ROLE_MAP = {
    "ENTJ": "Leader / Coordinator", "ESTJ": "Leader / Coordinator",
    "INTJ": "Planner / Architect", "ISTJ": "Planner / Architect",
    "ENTP": "Ideator / Strategist", "INTP": "Ideator / Strategist",
    "ENFJ": "Coach / Mediator", "INFJ": "Coach / Mediator",
    "ESFJ": "Facilitator / Operations", "ISFJ": "Facilitator / Operations",
    "ESFP": "Energizer / UX & Content", "ISFP": "Energizer / UX & Content",
    "ESTP": "Troubleshooter / Executor", "ISTP": "Troubleshooter / Executor",
    "ENFP": "Storyteller / Community", "INFP": "Storyteller / Community"
}

# Thêm lời giải thích cho từng vai trò
ROLE_EXPLANATIONS = {
    "Leader / Coordinator": "**Nhóm Leader/Coordinator (ENTJ, ESTJ):** Giỏi định hướng, lập kế hoạch và điều phối nguồn lực. Họ là những người thuyền trưởng, dẫn dắt đội nhóm đi đúng hướng.",
    "Planner / Architect": "**Nhóm Planner/Architect (INTJ, ISTJ):** Có tư duy chiến lược, khả năng xây dựng hệ thống và kế hoạch dài hạn. Họ là những kiến trúc sư của dự án.",
    "Ideator / Strategist": "**Nhóm Ideator/Strategist (ENTP, INTP):** Sáng tạo, thích khám phá ý tưởng mới và tìm ra các giải pháp độc đáo cho những vấn đề phức tạp.",
    "Coach / Mediator": "**Nhóm Coach/Mediator (ENFJ, INFJ):** Thấu cảm, có khả năng truyền cảm hứng, phát triển tiềm năng con người và giải quyết xung đột nội bộ.",
    "Facilitator / Operations": "**Nhóm Facilitator/Operations (ESFJ, ISFJ):** Thực tế, chú trọng chi tiết và giỏi hỗ trợ, đảm bảo quy trình vận hành trơn tru và mọi người được kết nối.",
    "Energizer / UX & Content": "**Nhóm Energizer/UX & Content (ESFP, ISFP):** Mang lại năng lượng tích cực, có gu thẩm mỹ tốt, phù hợp với các vai trò liên quan đến trải nghiệm người dùng và sáng tạo nội dung.",
    "Troubleshooter / Executor": "**Nhóm Troubleshooter/Executor (ESTP, ISTP):** Linh hoạt, phản ứng nhanh và giỏi giải quyết các vấn đề phát sinh tức thời. Họ là những người thực thi hiệu quả.",
    "Storyteller / Community": "**Nhóm Storyteller/Community (ENFP, INFP):** Giỏi truyền tải thông điệp, xây dựng câu chuyện và kết nối cộng đồng. Họ là linh hồn của đội nhóm."
}


# Tải mô hình embedding
print("Đang tải Sentence Transformer model...")
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Tải xong Sentence Transformer model.")

# Tải 4 mô hình classifier đã huấn luyện
classifiers = {}
dimensions = ["IE", "NS", "TF", "JP"]
for dim in dimensions:
    try:
        model_path = f"models/clf_{dim}.joblib"
        classifiers[dim] = joblib.load(model_path)
        print(f"Đã tải classifier cho: {dim}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {model_path}. Hãy chắc chắn bạn đã huấn luyện và lưu model.")
        exit()

# Ánh xạ nhãn 0/1 sang chữ cái
LABEL_MAP = {
    "IE": {0: "I", 1: "E"},
    "NS": {0: "N", 1: "S"},
    "TF": {0: "T", 1: "F"},
    "JP": {0: "J", 1: "P"}
}

# --- 2. HÀM XỬ LÝ LOGIC CHÍNH ---

def get_explanation(text, dimension):
    """
    Lấy 2-3 từ khóa ảnh hưởng nhất bằng phương pháp Leave-One-Out (nhanh hơn LIME).
    """
    try:
        clf = classifiers[dimension]
        
        # 1. Lấy xác suất gốc
        original_embedding = encoder_model.encode([text])
        original_proba = clf.predict_proba(original_embedding)[0]
        predicted_class_index = np.argmax(original_proba)
        original_confidence = original_proba[predicted_class_index]

        # 2. Tách văn bản thành các từ duy nhất
        words = list(set(re.findall(r'\b\w+\b', text.lower())))
        if not words: return "Không có từ để phân tích"

        # 3. Tính toán sự thay đổi khi bỏ đi từng từ
        word_impacts = {}
        for word in words:
            # Tạo văn bản mới không có từ hiện tại
            temp_text = ' '.join([w for w in text.lower().split() if w != word])
            if not temp_text: continue
            
            temp_embedding = encoder_model.encode([temp_text])
            new_proba = clf.predict_proba(temp_embedding)[0]
            new_confidence = new_proba[predicted_class_index]
            
            # Tác động là sự sụt giảm của độ tin cậy
            impact = original_confidence - new_confidence
            word_impacts[word] = impact

        # 4. Sắp xếp và lấy 3 từ có tác động lớn nhất
        if not word_impacts: return "Không thể tạo giải thích"
            
        sorted_words = sorted(word_impacts.items(), key=lambda item: item[1], reverse=True)
        top_words = [word for word, impact in sorted_words[:3]]
        
        return ", ".join(f"'{word}'" for word in top_words)
    except Exception as e:
        print(f"Lỗi khi tạo giải thích cho {dimension}: {e}")
        return "Không thể tạo giải thích"


def hamming_distance(mbti1, mbti2):
    """Tính khoảng cách Hamming giữa 2 chuỗi MBTI."""
    return sum(c1 != c2 for c1, c2 in zip(mbti1, mbti2))

def get_team_suggestions(result_df):
    """
    Thuật toán tham lam để gợi ý đội nhóm dựa trên kết quả dự đoán.
    """
    if len(result_df) < 4:
        return "Cần ít nhất 4 thành viên để có thể đưa ra gợi ý đội nhóm tối ưu.", ""

    # --- Triển khai giải pháp tham lam ---
    members = result_df.to_dict('records')
    
    # Tính tổng độ tin cậy cho mỗi thành viên
    for member in members:
        member['total_confidence'] = sum(
            float(member[f'Độ tin cậy ({dim})'].strip('%'))
            for dim in dimensions
        )

    # Sắp xếp thành viên theo độ tin cậy giảm dần
    members.sort(key=lambda x: x['total_confidence'], reverse=True)
    
    # Bắt đầu với người có độ tin cậy cao nhất
    optimal_team = [members.pop(0)]
    
    # Thêm 3 thành viên tiếp theo để tối đa hóa sự đa dạng
    while len(optimal_team) < 4 and members:
        best_candidate = None
        max_diversity_score = -1
        
        for candidate in members:
            current_diversity_score = sum(
                hamming_distance(candidate['MBTI Dự đoán'], team_member['MBTI Dự đoán'])
                for team_member in optimal_team
            )
            if current_diversity_score > max_diversity_score:
                max_diversity_score = current_diversity_score
                best_candidate = candidate
        
        if best_candidate:
            optimal_team.append(best_candidate)
            members.remove(best_candidate)
            
    # --- Định dạng kết quả đầu ra ---
    team_roles = [m['Vai trò gợi ý'].split(' / ')[0] for m in optimal_team] # Lấy vai trò chính
    
    # Kiểm tra ràng buộc vai trò
    required_roles = {"Leader", "Planner", "Executor", "Facilitator"}
    present_roles = set(team_roles)
    missing_roles = required_roles - present_roles
    
    # Tạo chuỗi kết quả
    suggestion_text = "### 🤝 Đội nhóm gợi ý (Tối ưu cho sự đa dạng)\n\n"
    for member in optimal_team:
        suggestion_text += f"- **{member['Thành viên']}:** {member['MBTI Dự đoán']} ({member['Vai trò gợi ý']})\n"
        
    suggestion_text += "\n**Phân tích đội nhóm:**\n"
    if not missing_roles:
        suggestion_text += "✅ Đội nhóm này cân bằng, đã có đủ các vai trò cốt lõi: Leader, Planner, Executor, Facilitator.\n"
    else:
        suggestion_text += f"⚠️ **Lưu ý:** Đội nhóm này đang thiếu các vai trò: **{', '.join(missing_roles)}**. Cân nhắc bổ sung thành viên có các vai trò này.\n"

    # Thêm giải thích vai trò
    explanation_text = "### 📖 Giải thích các vai trò có trong đội\n\n"
    unique_roles_in_team = {m['Vai trò gợi ý'] for m in optimal_team}
    for role in unique_roles_in_team:
        if role in ROLE_EXPLANATIONS:
            explanation_text += ROLE_EXPLANATIONS[role] + "\n\n"
            
    return suggestion_text, explanation_text


def process_team_builder(*args):
    """
    Hàm tổng hợp: nhận (tên, văn bản) -> dự đoán -> giải thích -> gợi ý đội nhóm.
    """
    members_data = []
    
    # args là một list phẳng: [name1, text1, name2, text2, ...]
    names = args[0::2] # Lấy tất cả các tên
    texts = args[1::2] # Lấy tất cả các văn bản

    for i, (name, text) in enumerate(zip(names, texts)):
        if not text.strip(): continue
        
        print(f"Đang xử lý Thành viên {i+1}...")
        
        # Quyết định tên thành viên
        member_name = name.strip() if name.strip() else f"Người {i+1}"

        embedding = encoder_model.encode([text])
        mbti_type = ""
        confidences = {}
        explanations = ""
        
        for dim in dimensions:
            # Dự đoán
            clf = classifiers[dim]
            pred_label = clf.predict(embedding)[0]
            pred_proba = clf.predict_proba(embedding)[0]
            confidence = pred_proba[pred_label]
            mbti_char = LABEL_MAP[dim][pred_label]
            mbti_type += mbti_char
            confidences[dim] = f"{confidence*100:.2f}%"

            # Lấy giải thích
            explanation_words = get_explanation(text, dim)
            explanations += f"**{mbti_char}** vì: {explanation_words}\n"


        role = MBTI_ROLE_MAP.get(mbti_type, "Chưa xác định")
        
        member_info = {
            "Thành viên": member_name,
            "MBTI Dự đoán": mbti_type,
            "Vai trò gợi ý": role,
            "Từ khóa ảnh hưởng": explanations.strip()
        }
        for dim in dimensions:
            member_info[f"Độ tin cậy ({dim})"] = confidences[dim]
        
        members_data.append(member_info)

    if not members_data:
        return pd.DataFrame(), "Vui lòng nhập văn bản của ít nhất một thành viên.", ""

    # Sắp xếp lại thứ tự cột cho đẹp mắt
    column_order = [
        "Thành viên", "MBTI Dự đoán", "Vai trò gợi ý", 
        "Độ tin cậy (IE)", "Độ tin cậy (NS)", "Độ tin cậy (TF)", "Độ tin cậy (JP)",
        "Từ khóa ảnh hưởng"
    ]
    result_df = pd.DataFrame(members_data)[column_order]
    
    print("Đã xử lý xong. Đang tạo gợi ý đội nhóm...")
    suggestion_text, explanation_text = get_team_suggestions(result_df)
    
    return result_df, suggestion_text, explanation_text

# --- 3. TẠO GIAO DIỆN WEB VỚI GRADIO ---

APP_DESCRIPTION = """
## Chào mừng đến với AI Team Builder!
1.  **Nhập thông tin:** Điền tên (tùy chọn) và dán một đoạn văn bản (tiếng Anh) của mỗi thành viên vào các ô bên dưới. Cần ít nhất 4 thành viên để có gợi ý đội nhóm.
2.  **Nhấn "Gợi ý đội nhóm":** AI sẽ dự đoán MBTI, sau đó đề xuất một đội hình tối ưu về sự đa dạng và cân bằng vai trò.
3.  **Xem kết quả:** Xem bảng phân tích chi tiết và các gợi ý đội nhóm bên dưới.

**Lưu ý về đạo đức:** Công cụ này chỉ mang tính chất tham khảo. Kết quả dự đoán không phải là một phán quyết về con người và không nên được sử dụng cho các quyết định quan trọng như tuyển dụng. Chúng tôi không lưu lại bất kỳ văn bản nào bạn nhập vào.
"""

NUM_MEMBERS = 5

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI Team Builder Prototype")
    gr.Markdown(APP_DESCRIPTION)

    inputs = []
    with gr.Row():
        for i in range(NUM_MEMBERS):
            with gr.Column():
                name_input = gr.Textbox(
                    label=f"Tên Thành viên {i+1}", 
                    placeholder=f"Tùy chọn (mặc định: Người {i+1})"
                )
                text_input = gr.Textbox(
                    label=f"Văn bản của Thành viên {i+1}", 
                    lines=6, 
                    placeholder="Dán văn bản vào đây..."
                )
                inputs.append(name_input)
                inputs.append(text_input)
    
    submit_button = gr.Button("Gợi ý đội nhóm", variant="primary")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=3): # Tăng độ rộng cột bảng
            gr.Markdown("## 📊 Bảng phân tích chi tiết")
            output_dataframe = gr.DataFrame(label="Kết quả")
        with gr.Column(scale=1):
            gr.Markdown("## 💡 Gợi ý & Giải thích")
            output_suggestion = gr.Markdown(label="Gợi ý đội nhóm")
            output_explanation = gr.Markdown(label="Giải thích vai trò")


    submit_button.click(
        fn=process_team_builder,
        inputs=inputs,
        outputs=[output_dataframe, output_suggestion, output_explanation],
        show_progress="full" # Thêm thanh tiến trình
    )

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    demo.launch()

