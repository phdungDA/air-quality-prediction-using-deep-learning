"""
preprocess.py
Pipeline tiền xử lý dữ liệu AQI Hà Nội: từ file CSV thô đến dữ liệu sạch,
đã chuẩn hóa, sẵn sàng cho bước tạo sliding window.

Luồng xử lý chính (run_pipeline):
    1. load_raw_data       -> đọc CSV, xử lý format đặc thù của file
    2. clean_datetime_index -> parse thời gian, sắp xếp, loại trùng lặp,
                                lấp đầy các mốc giờ bị thiếu (reindex)
    3. select_features     -> chỉ giữ lại các cột dùng cho mô hình
    4. handle_missing_values -> nội suy theo thời gian + fill biên
    5. scale_features      -> MinMaxScaler, lưu lại scaler để dùng khi inference
"""

import logging
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

import sys

# Dùng insert(0, ...) thay vì append(...): đảm bảo thư mục src/ được ưu
# tiên tìm kiếm TRƯỚC site-packages, tránh xung đột với gói PyPI có sẵn
# tên trùng "config" (một package quản lý cấu hình công khai trên PyPI,
# không liên quan đến project). append() sẽ thêm vào CUỐI sys.path,
# khiến Python tìm thấy gói pip "config" trước và báo lỗi ImportError.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATETIME_COLUMN,
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
    TIME_FREQ,
)

logger = logging.getLogger("preprocess")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Đọc file CSV thô.

    Lưu ý đặc thù của file này: mỗi dòng bị bọc trong một cặp dấu ngoặc kép
    bao quanh TOÀN BỘ dòng (kể cả dấu phẩy phân tách cột), khiến
    pandas.read_csv() mặc định hiểu nhầm cả dòng là MỘT cột duy nhất.
    Do đó cần đọc thô từng dòng, bóc dấu ngoặc kép bao ngoài, rồi mới
    parse lại bằng pandas.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Bóc dấu " " bao quanh mỗi dòng (nếu có) trước khi parse CSV thật sự
    cleaned_lines = [line.strip().strip('"') for line in raw_lines]

    from io import StringIO
    csv_text = "\n".join(cleaned_lines)
    df = pd.read_csv(StringIO(csv_text))

    logger.info(f"Đã đọc {len(df)} dòng từ {path}")
    return df


def clean_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa cột thời gian thành index, xử lý 2 vấn đề thường gặp
    trong dữ liệu time-series thu thập từ API/cảm biến thật:

    1. Timestamp trùng lặp (duplicate) — giữ lại bản ghi đầu tiên.
    2. Timestamp bị thiếu (gap) — ví dụ cảm biến mất kết nối vài giờ.
       Ta "reindex" theo tần suất chuẩn (hourly) để các mốc giờ bị thiếu
       lộ diện thành hàng NaN, thay vì bị bỏ sót âm thầm. Đây là bước
       bắt buộc trước khi nội suy, nếu không sliding window sau này sẽ
       nối 2 mốc thời gian không liền kề lại với nhau mà không hay biết.
    """
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])

    # Sắp xếp theo thời gian tăng dần (dữ liệu thô có thể không đảm bảo thứ tự)
    df = df.sort_values(DATETIME_COLUMN)

    n_duplicates = df[DATETIME_COLUMN].duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Phát hiện {n_duplicates} timestamp trùng lặp -> giữ bản ghi đầu tiên")
        df = df.drop_duplicates(subset=DATETIME_COLUMN, keep="first")

    df = df.set_index(DATETIME_COLUMN)

    # Reindex về lưới thời gian đều đặn theo giờ, làm lộ các gap còn thiếu
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=TIME_FREQ)
    n_missing_slots = len(full_range) - len(df)
    if n_missing_slots > 0:
        logger.warning(f"Phát hiện {n_missing_slots} mốc giờ bị thiếu hoàn toàn -> reindex để lộ diện")
    df = df.reindex(full_range)
    df.index.name = DATETIME_COLUMN

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chỉ giữ lại các cột đặc trưng (FEATURE_COLUMNS) dùng cho mô hình,
    loại bỏ các cột định danh không mang thông tin dự đoán
    (City, Country Code, Timezone... vốn là hằng số trong toàn bộ dataset).
    """
    available_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=available_drop, errors="ignore")
    df = df[FEATURE_COLUMNS]
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values theo chiến lược phù hợp với time-series:

    1. Interpolation theo thời gian (method='time'): phù hợp cho các gap
       NGẮN xen giữa (vài giờ) — nội suy tuyến tính có trọng số theo
       khoảng cách thời gian thực tế, chính xác hơn interpolation
       'linear' thông thường khi khoảng cách giữa các mốc không đều.
    2. Forward-fill + Backward-fill: xử lý phần dư ở 2 đầu chuỗi mà
       interpolation không xử lý được (không có điểm neo phía trước/sau).

    Giải thích học thuật: với dữ liệu ô nhiễm không khí, giá trị giữa
    2 thời điểm liền kề (chênh lệch 1-2 giờ) thường biến thiên chậm và
    có tính liên tục vật lý (PM2.5 không thể nhảy vọt tức thời), nên
    interpolation tuyến tính là lựa chọn hợp lý hơn so với việc điền
    giá trị trung bình toàn cục (sẽ làm mất đặc trưng xu hướng cục bộ).
    """
    n_missing_before = df.isnull().sum().sum()

    df = df.interpolate(method="time", limit_direction="both")

    # Phòng trường hợp còn sót giá trị NaN ở 2 đầu chuỗi sau interpolation
    df = df.ffill().bfill()

    n_missing_after = df.isnull().sum().sum()
    logger.info(
        f"Xử lý missing values: {n_missing_before} -> {n_missing_after} giá trị thiếu"
    )
    return df


def scale_features(df: pd.DataFrame, scaler_path: str = SCALER_PATH) -> tuple:
    """
    Chuẩn hóa toàn bộ features về khoảng [0, 1] bằng MinMaxScaler.

    Lý do chọn MinMaxScaler thay vì StandardScaler: các hàm kích hoạt
    phổ biến trong lớp output/gate của LSTM (sigmoid, tanh) hoạt động
    ổn định nhất khi input nằm trong khoảng giới hạn [0, 1] hoặc [-1, 1],
    giúp gradient không bị bão hòa (saturation) trong quá trình huấn luyện.

    Scaler được lưu lại bằng joblib để dùng "transform" (không "fit" lại)
    khi thực hiện inference trên dữ liệu mới — đảm bảo tính nhất quán
    giữa không gian giá trị lúc train và lúc dự đoán thực tế.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    dump(scaler, scaler_path)
    logger.info(f"Đã lưu scaler tại {scaler_path}")

    return scaled_df, scaler


def run_pipeline(
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    scaler_path: str = SCALER_PATH,
    save_unscaled: bool = True,
) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline tiền xử lý từ đầu đến cuối.
    Trả về DataFrame đã chuẩn hóa (scaled), đồng thời lưu ra file CSV
    phiên bản CHƯA chuẩn hóa (để dễ kiểm tra/trực quan hóa bằng mắt)
    và scaler dùng cho bước inference sau này.
    """
    df = load_raw_data(raw_path)
    df = clean_datetime_index(df)
    df = select_features(df)
    df = handle_missing_values(df)

    if save_unscaled:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path)
        logger.info(f"Đã lưu dữ liệu đã làm sạch (chưa chuẩn hóa) tại {processed_path}")

    scaled_df, _ = scale_features(df, scaler_path)
    return scaled_df


if __name__ == "__main__":
    result_df = run_pipeline()
    print(result_df.head())
    print(f"\nShape sau xử lý: {result_df.shape}")