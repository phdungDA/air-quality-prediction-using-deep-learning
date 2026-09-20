"""
Tiền xử lý dữ liệu thô -> sequence cho LSTM.
Đọc từ PostgreSQL, lưu X/y ra data/processed/ để tái sử dụng.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.data.dataloader import load_from_db

FEATURE_COLS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET_COL   = "aqi"
NUM_CLASSES  = 5

PROCESSED_DIR = "data/processed"


def scale_features(df, scaler: MinMaxScaler = None):
    """Chuẩn hoá các cột feature về [0,1]. Trả về mảng đã scale + scaler đã fit."""
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[FEATURE_COLS])
    else:
        scaled = scaler.transform(df[FEATURE_COLS])
    return scaled, scaler


def create_sequences(df, window_size: int = 24, scaler: MinMaxScaler = None):
    """
    Sliding window:
    Input (X): window_size giờ liên tiếp của các thành phần ô nhiễm
    Output (y): nhãn AQI (0..4, đã trừ 1 từ 1..5) tại thời điểm NGAY SAU cửa sổ đó
    """
    scaled_features, scaler = scale_features(df, scaler)
    labels = df[TARGET_COL].values

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(scaled_features[i: i + window_size])
        y.append(labels[i + window_size])

    X = np.array(X)
    y = np.array(y) - 1  # 1..5 -> 0..4 để dùng sparse_categorical_crossentropy

    return X, y, scaler


def save_processed(X: np.ndarray, y: np.ndarray, scaler: MinMaxScaler) -> None:
    """Lưu X, y, scaler ra data/processed/."""
    # Nếu data/processed đang là file thì xóa đi rồi tạo folder
    if os.path.isfile(PROCESSED_DIR):
        os.remove(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(f"{PROCESSED_DIR}/X.npy", X)
    np.save(f"{PROCESSED_DIR}/y.npy", y)
    joblib.dump(scaler, f"{PROCESSED_DIR}/scaler.pkl")
    print(f"✅ Đã lưu X{X.shape}, y{y.shape} và scaler vào {PROCESSED_DIR}/")


def load_processed():
    """Đọc lại X, y, scaler từ data/processed/ (bỏ qua bước preprocessing)."""
    X      = np.load(f"{PROCESSED_DIR}/X.npy")
    y      = np.load(f"{PROCESSED_DIR}/y.npy")
    scaler = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
    print(f"✅ Đã load X{X.shape}, y{y.shape} từ {PROCESSED_DIR}/")
    return X, y, scaler


if __name__ == "__main__":
    df = load_from_db()
    X, y, scaler = create_sequences(df, window_size=24)
    save_processed(X, y, scaler)
