"""
Tiền xử lý dữ liệu thô -> sequence cho LSTM.
Schema tạm thời: dùng thẳng các thành phần ô nhiễm của OpenWeatherMap làm feature,
                 dùng "aqi" (1-5) có sẵn trong response làm nhãn classification.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET_COL = "aqi"  # nhãn 1..5 do OpenWeatherMap cung cấp sẵn
NUM_CLASSES = 5


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
