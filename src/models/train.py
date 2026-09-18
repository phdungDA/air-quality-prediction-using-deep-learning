"""
Script train baseline LSTM.
Chạy: python src/models/train.py
(chạy từ thư mục gốc project để .env được load đúng)
"""

import os
import sys
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cho phép import "data.*" / "models.*"

from data.dataloader import fetch_air_pollution_history
from data.preprocess import create_sequences
from models.architectures import build_lstm_baseline
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

WINDOW_SIZE = 24  # 24 giờ lịch sử để dự đoán giờ tiếp theo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    end = int(time.time())
    start = end - 365 * 24 * 3600  # 365 ngày gần nhất

    print("Đang tải dữ liệu từ OpenWeatherMap...")
    df = fetch_air_pollution_history(start=start, end=end)
    print(f"Tải được {len(df)} bản ghi.")

    X, y, scaler = create_sequences(df, window_size=WINDOW_SIZE)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # không shuffle vì là time-series
    )

    model = build_lstm_baseline(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "lstm_baseline.h5")

    callbacks = [
        # Lưu model tại epoch có val_accuracy cao nhất
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Dừng sớm nếu 5 epoch liên tiếp không cải thiện val_accuracy
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
    )

    print(f"Đã lưu model tốt nhất vào {best_model_path}")
    with open(os.path.join(MODEL_DIR, "minmax_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Đã lưu model + scaler vào {MODEL_DIR}")


if __name__ == "__main__":
    main()
