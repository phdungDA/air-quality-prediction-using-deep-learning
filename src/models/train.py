"""
Script train baseline LSTM.
Chạy: python src/models/train.py
(chạy từ thư mục gốc project)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.preprocess import load_processed
from src.models.architectures import build_lstm_baseline
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

WINDOW_SIZE = 24

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")


def main():
    print("Đang load data từ data/processed/ ...")
    X, y, scaler = load_processed()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # không shuffle vì là time-series
    )

    model = build_lstm_baseline(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "lstm_baseline.h5")

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
    )

    joblib.dump(scaler, os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
    print(f"✅ Đã lưu model + scaler vào {MODEL_DIR}/")


if __name__ == "__main__":
    main()
