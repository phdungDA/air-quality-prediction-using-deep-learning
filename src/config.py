"""
config.py
Khai báo tập trung các hằng số dùng xuyên suốt pipeline xử lý dữ liệu:
đường dẫn file, danh sách features, target, và tham số sliding window.
Mọi module khác chỉ import từ đây, không hard-code đường dẫn rải rác.
"""

import os

# ------------------------------------------------------------
# ĐƯỜNG DẪN THƯ MỤC (tương đối theo gốc project)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "aqi-Hanoi-22-25-raw-data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "aqi_hanoi_cleaned.csv")
SEQUENCES_DIR = os.path.join(BASE_DIR, "data", "sequences")
SCALER_PATH = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ------------------------------------------------------------
# CỘT DỮ LIỆU
# ------------------------------------------------------------
# Cột thời gian gốc trong file CSV (đã ở múi giờ địa phương Hà Nội)
DATETIME_COLUMN = "Local Time"

# Các cột không mang thông tin dự đoán (cố định giá trị hoặc trùng lặp
# thông tin thời gian) -> loại bỏ khỏi tập huấn luyện
DROP_COLUMNS = ["UTC Time", "City", "Country Code", "Timezone"]

# Các đặc trưng (features) đưa vào mô hình Multivariate Time-Series
FEATURE_COLUMNS = [
    "AQI", "CO", "NO2", "O3", "PM10", "PM25", "SO2",
    "Clouds", "Precipitation", "Pressure",
    "Relative Humidity", "Temperature", "UV Index", "Wind Speed",
]

# Biến mục tiêu cần dự đoán
TARGET_COLUMN = "PM25"

# Tần suất chuẩn của chuỗi thời gian (dữ liệu theo giờ)
TIME_FREQ = "h"

# ------------------------------------------------------------
# THAM SỐ SLIDING WINDOW
# ------------------------------------------------------------
WINDOW_SIZE = 24 * 7      # Dùng 7 ngày dữ liệu quá khứ (168 giờ)
PREDICTION_HORIZON = 24   # Dự đoán 24 giờ tiếp theo
TRAIN_SPLIT_RATIO = 0.8   # 80% train, phần còn lại chia đôi cho val/test
VAL_SPLIT_RATIO = 0.1     # 10% validation, 10% test