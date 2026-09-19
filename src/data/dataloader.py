"""
dataloader.py
Chuyển dữ liệu time-series đã chuẩn hóa thành dạng Sliding Window (X, y)
để đưa vào mô hình Deep Learning, và chia tập train/val/test theo đúng
thứ tự thời gian (không shuffle).
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

# insert(0, ...) để thư mục src/ được ưu tiên hơn site-packages,
# tránh xung đột với gói PyPI trùng tên "config" (xem giải thích chi
# tiết trong preprocess.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZON,
    SEQUENCES_DIR,
    TARGET_COLUMN,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    WINDOW_SIZE,
)

logger = logging.getLogger("dataloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    horizon: int = PREDICTION_HORIZON,
    target_col: str = TARGET_COLUMN,
) -> tuple:
    """
    Tạo các cặp (X, y) theo kiểu sliding window (look-back window).

    Với window_size=168 (7 ngày) và horizon=24 (24 giờ tới):
        X[i] = dữ liệu đa biến của 168 giờ liên tiếp (t-168 ... t-1)
        y[i] = giá trị PM2.5 tại 24 giờ tiếp theo (t ... t+23)

    Đây là kỹ thuật chuẩn để chuyển bài toán time-series thành bài toán
    học có giám sát (supervised learning) mà LSTM/GRU có thể xử lý:
    mô hình học cách ánh xạ một "cửa sổ" lịch sử sang một "cửa sổ" tương lai.

    Trả về:
        X: mảng shape (n_samples, window_size, n_features)
        y: mảng shape (n_samples, horizon)
    """
    feature_values = df[FEATURE_COLUMNS].values
    target_col_idx = FEATURE_COLUMNS.index(target_col)

    X, y = [], []
    n_total = len(df)

    # Vòng lặp trượt cửa sổ qua toàn bộ chuỗi thời gian
    for i in range(window_size, n_total - horizon + 1):
        X.append(feature_values[i - window_size : i, :])
        y.append(feature_values[i : i + horizon, target_col_idx])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Đã tạo {X.shape[0]} mẫu sliding window. X shape={X.shape}, y shape={y.shape}")
    return X, y


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Chia tập train/validation/test theo ĐÚNG THỨ TỰ THỜI GIAN (không shuffle).

    Giải thích học thuật (quan trọng để đưa vào báo cáo): với bài toán
    time-series, việc shuffle dữ liệu trước khi chia sẽ gây ra
    "data leakage" — mô hình có thể nhìn thấy thông tin từ tương lai
    trong lúc train, dẫn đến kết quả đánh giá bị đánh giá cao giả tạo
    (overly optimistic) so với khả năng dự đoán thực tế. Do đó tập test
    luôn phải là đoạn thời gian NẰM SAU tập train/validation.
    """
    n_total = len(X)
    n_train = int(n_total * TRAIN_SPLIT_RATIO)
    n_val = int(n_total * VAL_SPLIT_RATIO)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    logger.info(
        f"Chia dữ liệu: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} mẫu"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_sequences(X_train, y_train, X_val, y_val, X_test, y_test, out_dir: str = SEQUENCES_DIR):
    """Lưu toàn bộ sequences ra file .npy để tái sử dụng ở bước huấn luyện."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    logger.info(f"Đã lưu toàn bộ sequences vào {out_dir}")


if __name__ == "__main__":
    from config import PROCESSED_DATA_PATH
    from preprocess import scale_features

    # Đọc lại dữ liệu đã làm sạch (chưa chuẩn hóa) rồi scale lại
    # để đảm bảo dataloader có thể chạy độc lập với preprocess
    df_clean = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    df_scaled, _ = scale_features(df_clean)

    X, y = create_sliding_windows(df_scaled)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    save_sequences(X_train, y_train, X_val, y_val, X_test, y_test)