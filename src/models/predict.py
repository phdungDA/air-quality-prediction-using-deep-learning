"""
Load model + scaler đã train và sinh y_pred từ X.
"""

import os
import pickle

import numpy as np
from tensorflow import keras


def load_artifacts(model_path: str, scaler_path: str):
    """Trả về (model, scaler). compile=False vì chỉ suy luận, không train tiếp."""
    model = keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(model, X: np.ndarray, batch_size: int = 256):
    """
    X: (n_samples, 24, 8), đã scale.
    Trả về (y_pred, y_proba):
      y_proba: (n_samples, 5) xác suất softmax
      y_pred : (n_samples,) nhãn 0..4
    """
    y_proba = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba
