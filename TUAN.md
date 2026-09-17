# Baseline Model — Nguyễn Minh Tuấn

## Nhiệm vụ

Xây dựng **baseline model** đơn giản để dự đoán chỉ số AQI (chất lượng không khí) từ dữ liệu lịch sử.  
Đảm bảo pipeline chạy được end-to-end với dữ liệu mẫu từ OpenWeatherMap.

---

## Baseline model là gì?

Baseline model là model **đơn giản nhất có thể chạy được**, dùng để làm mốc so sánh.  
Sau này nếu xây model phức tạp hơn (GRU, BiLSTM...), ta so kết quả với baseline để biết có cải thiện không.

---

## Luồng hoạt động

```
Dữ liệu 8 chất ô nhiễm (24 giờ liên tiếp)
            ↓
          LSTM
            ↓
         Dense
            ↓
         Softmax
            ↓
  Dự đoán mức AQI (1 trong 5 mức)
```

---

## Input — Đầu vào

Mỗi lần dự đoán, model nhận vào **24 giờ dữ liệu liên tiếp**, gồm 8 chất ô nhiễm:

| Ký hiệu | Chất |
|---|---|
| co | Carbon monoxide |
| no | Nitric oxide |
| no2 | Nitrogen dioxide |
| o3 | Ozone |
| so2 | Sulfur dioxide |
| pm2_5 | Bụi mịn PM2.5 |
| pm10 | Bụi PM10 |
| nh3 | Ammonia |

→ Shape: `(24, 8)` — 24 giờ × 8 chất

---

## Output — Đầu ra

Model dự đoán **mức AQI của giờ tiếp theo** (giờ thứ 25), gồm 5 mức:

| Mức | Ý nghĩa |
|---|---|
| 1 | Tốt |
| 2 | Khá |
| 3 | Trung bình |
| 4 | Xấu |
| 5 | Rất xấu |

→ Output là 1 trong 5 mức trên.

---

## Các file liên quan

```
src/
├── data/
│   ├── dataloader.py     # Gọi API lấy dữ liệu từ OpenWeatherMap
│   └── preprocess.py     # Chuẩn hoá + tạo sliding window 24h
├── models/
│   ├── architectures.py  # Định nghĩa kiến trúc LSTM
│   └── train.py          # Script train model, lưu kết quả
models/
├── lstm_baseline.h5      # Trọng số model sau khi train (tự sinh ra)
└── minmax_scaler.pkl     # Scaler dùng để chuẩn hoá (tự sinh ra)
```

---

## Cách chạy

```bash
# Chạy từ thư mục gốc project
python src/models/train.py
```

Kết quả sau khi chạy xong:
- Model được lưu vào `models/lstm_baseline.h5`
- Scaler được lưu vào `models/minmax_scaler.pkl`
- Terminal hiển thị `accuracy` và `val_accuracy` sau mỗi epoch

---

## Lưu ý cho Nguyễn Ngọc Tiến (Evaluation)

Sau khi train xong, model và scaler đã sẵn sàng để dùng:

- **Model**: `models/lstm_baseline.h5` — load lên rồi gọi `model.predict(X)` là ra `y_pred`
- **Scaler**: `models/minmax_scaler.pkl` — dùng để chuẩn hoá input trước khi đưa vào model
- **y_true**: cột `aqi` trong DataFrame (giá trị 0–4, đã trừ 1 từ 1–5)
- **y_pred**: output của model, lấy `np.argmax(y_pred, axis=1)` để ra nhãn

```python
import pickle
import numpy as np
from tensorflow import keras

model  = keras.models.load_model("models/lstm_baseline.h5")
scaler = pickle.load(open("models/minmax_scaler.pkl", "rb"))

# X shape: (n_samples, 24, 8) — đã được scale bằng scaler
y_pred_proba = model.predict(X)
y_pred       = np.argmax(y_pred_proba, axis=1)  # 0..4
```
