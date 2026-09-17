"""
Baseline model: Input -> LSTM -> Dense -> Softmax -> AQI class
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_baseline(input_shape: tuple, num_classes: int = 5, lstm_units: int = 64) -> keras.Model:
    """
    input_shape: (window_size, num_features)
    num_classes: số lớp AQI (mặc định 5: 1=Good ... 5=Very Poor, đã map về 0..4)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units),
        layers.Dropout(0.2),  # tắt ngẫu nhiên 20% neuron mỗi bước train để giảm overfit
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
