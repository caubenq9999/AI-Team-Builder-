# src/data.pip install -r D:\MBTI\reports\requirements.txt
import re
import emoji
import pandas as pd
from sklearn.model_selection import train_test_split

# ===================== #
#   Chuẩn hoá văn bản   #
# ===================== #
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = emoji.demojize(text, language="en")
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text


# ===================== #
#   Sinh 4 nhãn MBTI    #
# ===================== #
def split_ie(t): return 0 if t[0] == "I" else 1
def split_ns(t): return 0 if t[1] == "N" else 1
def split_tf(t): return 0 if t[2] == "T" else 1
def split_jp(t): return 0 if t[3] == "J" else 1


# ===================== #
#   Hàm load dữ liệu    #
# ===================== #
def load_datasets(train_path=None, valid_path=None, test_path=None,
                  split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Nếu có đủ train/valid/test path → đọc trực tiếp.
    Nếu chỉ có 1 file (train_path) → tự chia theo split_ratio.
    """

    if valid_path and test_path:
        # Đã có sẵn 3 file
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
    else:
        # Chỉ có 1 file → tự động chia
        df = pd.read_csv(train_path)
        train_df, temp_df = train_test_split(df, test_size=(1 - split_ratio[0]),
                                             random_state=seed, stratify=df["type"])
        valid_size = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        valid_df, test_df = train_test_split(temp_df, test_size=(1 - valid_size),
                                             random_state=seed, stratify=temp_df["type"])

    # Chuẩn hoá text
    for df in [train_df, valid_df, test_df]:
        df["posts"] = df["posts"].str.replace(r"\|\|\|", " ", regex=True)
        df["posts"] = df["posts"].apply(normalize_text)

        # Sinh thêm 4 nhãn nhị phân
        df["IE"] = df["type"].apply(split_ie)
        df["NS"] = df["type"].apply(split_ns)
        df["TF"] = df["type"].apply(split_tf)
        df["JP"] = df["type"].apply(split_jp)

    return train_df, valid_df, test_df
