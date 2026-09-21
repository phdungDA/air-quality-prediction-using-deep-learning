"""
Tải và xử lý dữ liệu từ CSV thô.

Pipeline:
  1. Đọc CSV raw
  2. Parse datetime, sort, drop duplicates
  3. Đưa về lưới 1 giờ liên tục (reindex) → giờ bị API bỏ sót thành NaN
  4. Xử lý missing values:
       - gap ngắn (<= MAX_GAP_HOURS giờ)  → nội suy tuyến tính
       - gap dài  (>  MAX_GAP_HOURS giờ)  → giữ NaN, các window chạm vào sẽ bị loại
       - AQI (nhãn) KHÔNG được nội suy: giờ nào thiếu nhãn thì không làm target
  5. Tạo sliding window sequences (X, y), bỏ window chứa dữ liệu khuyết
  6. Chia tập theo thời gian: 70% train / 15% val / 15% test
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ── Hằng số ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_CSV_PATH  = os.path.join(PROJECT_ROOT, "data", "raw", "air_quality_3years.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

FEATURE_COLS  = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET_COL    = "aqi"
WINDOW_SIZE   = 24          # 24 giờ lịch sử → dự đoán giờ tiếp theo
MAX_GAP_HOURS = 6           # gap dài hơn ngưỡng này thì không nội suy
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15 (phần còn lại)

HOUR = pd.Timedelta(hours=1)


# ── 1. Đọc & làm sạch ────────────────────────────────────────────────────────
def load_raw_csv(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Đọc CSV thô, parse datetime, sort theo thời gian, bỏ bản ghi trùng.
    Trả về DataFrame sạch với index liên tục.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file CSV tại '{path}'.\n"
            "Hãy chạy src/ingestion/api_client.py để fetch dữ liệu trước."
        )

    df = pd.read_csv(path)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Sắp xếp theo thời gian
    df = df.sort_values("datetime").reset_index(drop=True)

    # Bỏ bản ghi trùng timestamp
    df = df.drop_duplicates(subset=["dt"]).reset_index(drop=True)

    print(f"✅ Đọc được {len(df)} bản ghi từ {path}")
    return df


# ── 2. Đưa về lưới 1 giờ liên tục ────────────────────────────────────────────
def regularize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex về lưới 1 giờ liên tục từ bản ghi đầu đến bản ghi cuối.
    Giờ nào API không trả về sẽ thành hàng NaN (thay vì bị "nuốt" mất, khiến
    sliding window tưởng 2 dòng liền nhau là 2 giờ liên tiếp).
    """
    df = df.copy()
    df["datetime"] = df["datetime"].dt.floor("60min")
    df = (df.drop_duplicates(subset="datetime", keep="last")
            .set_index("datetime")
            .sort_index())

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=HOUR, name="datetime")
    n_missing_rows = len(full_idx) - len(df)
    df = df.reindex(full_idx)

    # Thống kê các đoạn khuyết
    gap = df[TARGET_COL].isna().to_numpy()
    longest = cur = n_gaps = 0
    prev = False
    for g in gap:
        cur = cur + 1 if g else 0
        if g and not prev:
            n_gaps += 1
        longest = max(longest, cur)
        prev = g

    print(f"✅ Lưới 1 giờ: {len(full_idx)} giờ | thiếu {n_missing_rows} giờ "
          f"({n_missing_rows / len(full_idx):.2%}) | {n_gaps} đoạn khuyết | dài nhất {longest}h")
    return df.reset_index()


# ── 3. Xử lý missing values ──────────────────────────────────────────────────
def _long_gap_mask(is_na: pd.Series, max_gap: int) -> pd.Series:
    """True cho các ô nằm trong đoạn NaN liên tiếp dài hơn max_gap."""
    grp = (is_na != is_na.shift()).cumsum()
    run = is_na.groupby(grp).transform("size")
    return is_na & (run > max_gap)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Reindex về lưới 1 giờ (giờ bị thiếu → NaN).
    - Feature: nội suy tuyến tính cho gap <= MAX_GAP_HOURS; gap dài hơn giữ NaN hoàn toàn
      (không bịa dữ liệu, không ffill/bfill qua nhiều giờ).
    - AQI (nhãn): không nội suy. Giờ thiếu nhãn sẽ không được dùng làm target.
    """
    df = regularize_hourly(df)

    before = int(df[FEATURE_COLS].isna().sum().sum())
    for col in FEATURE_COLS:
        is_na  = df[col].isna()
        filled = df[col].interpolate(method="linear", limit_area="inside")
        filled[_long_gap_mask(is_na, MAX_GAP_HOURS)] = np.nan
        df[col] = filled
    after = int(df[FEATURE_COLS].isna().sum().sum())

    print(f"✅ Feature NaN: {before} → {after} "
          f"(phần còn lại là gap > {MAX_GAP_HOURS}h, các window chạm vào sẽ bị loại)")
    return df


# ── 4. Scale features ────────────────────────────────────────────────────────
def fit_scaler(df_train: pd.DataFrame) -> MinMaxScaler:
    """Fit MinMaxScaler CHỈ trên tập train để tránh data leakage (NaN được bỏ qua khi fit)."""
    scaler = MinMaxScaler()
    scaler.fit(df_train[FEATURE_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """Transform features của một tập bất kỳ bằng scaler đã fit (NaN được giữ nguyên)."""
    return scaler.transform(df[FEATURE_COLS])


# ── 5. Tạo sliding window sequences ──────────────────────────────────────────
def create_sequences(scaled_features: np.ndarray,
                     labels: np.ndarray,
                     window_size: int = WINDOW_SIZE):
    """
    Tạo cặp (X, y) bằng sliding window:
      X[i] = scaled_features[i : i+window_size]   shape: (window_size, n_features)
      y[i] = labels[i+window_size]                 nhãn AQI 0..4

    Bỏ window nếu: (a) có giờ nào trong cửa sổ còn NaN, hoặc (b) nhãn target bị thiếu.
    Vì dữ liệu đã ở lưới 1 giờ liên tục nên mọi window giữ lại đều đúng 24 giờ liên tiếp.
    """
    n = len(scaled_features)
    bad_row = np.isnan(scaled_features).any(axis=1).astype(int)
    csum = np.concatenate([[0], np.cumsum(bad_row)])     # tổng dồn để đếm hàng lỗi trong cửa sổ

    X, y = [], []
    total = max(n - window_size, 0)
    for i in range(total):
        target = labels[i + window_size]
        if csum[i + window_size] - csum[i] > 0 or np.isnan(target):
            continue
        X.append(scaled_features[i : i + window_size])
        y.append(target)

    skipped = total - len(X)
    if skipped:
        print(f"   ⚠️  Bỏ {skipped}/{total} window do chứa dữ liệu khuyết")
    if not X:
        raise ValueError("Không tạo được window nào (dữ liệu quá ngắn hoặc khuyết quá nhiều).")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── 6. Chia tập theo thời gian (không shuffle) ───────────────────────────────
def split_by_time(df: pd.DataFrame):
    """
    Chia DataFrame theo thứ tự thời gian:
      70% đầu   → train
      15% giữa  → val
      15% cuối  → test

    Trả về (df_train, df_val, df_test).
    """
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val :].reset_index(drop=True)

    print(f"✅ Chia tập  →  train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)} giờ")
    return df_train, df_val, df_test


# ── Pipeline tổng ─────────────────────────────────────────────────────────────
def build_dataset(csv_path: str = RAW_CSV_PATH,
                  window_size: int = WINDOW_SIZE,
                  save: bool = True):
    """
    Chạy toàn bộ pipeline:
      load → reindex 1h + xử lý missing → split → scale (fit trên train) → sequences → (save)

    Trả về:
      X_train, y_train,
      X_val,   y_val,
      X_test,  y_test,
      scaler
    """
    # 1. Load & clean
    df = load_raw_csv(csv_path)
    df = handle_missing(df)

    # Chuyển nhãn AQI 1..5 → 0..4 (giờ thiếu nhãn vẫn là NaN)
    df[TARGET_COL] = df[TARGET_COL] - 1

    # 2. Chia theo thời gian TRƯỚC khi scale
    df_train, df_val, df_test = split_by_time(df)

    # 3. Fit scaler chỉ trên train
    scaler = fit_scaler(df_train)

    # 4. Scale từng tập
    X_tr_scaled = apply_scaler(df_train, scaler)
    X_va_scaled = apply_scaler(df_val,   scaler)
    X_te_scaled = apply_scaler(df_test,  scaler)

    labels_train = df_train[TARGET_COL].values
    labels_val   = df_val[TARGET_COL].values
    labels_test  = df_test[TARGET_COL].values

    # 5. Tạo sequences
    X_train, y_train = create_sequences(X_tr_scaled, labels_train, window_size)
    X_val,   y_val   = create_sequences(X_va_scaled, labels_val,   window_size)
    X_test,  y_test  = create_sequences(X_te_scaled, labels_test,  window_size)

    print(f"\n📐 Kích thước các tập sau sliding window (window={window_size}h):")
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}   y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}  y_test:  {y_test.shape}")

    # 6. Lưu ra data/processed/
    if save:
        _save_processed(X_train, y_train, X_val, y_val, X_test, y_test, scaler)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ── Lưu / load processed ─────────────────────────────────────────────────────
def _save_processed(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Lưu toàn bộ tập processed ra data/processed/."""
    if os.path.isfile(PROCESSED_DIR):
        os.remove(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

    print(f"✅ Đã lưu toàn bộ tập processed + scaler vào '{PROCESSED_DIR}/'")


def load_processed():
    """Đọc lại các tập đã xử lý từ data/processed/ (bỏ qua bước build)."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    scaler  = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))

    print(f"✅ Load processed  →  train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ── Chạy trực tiếp để kiểm tra ───────────────────────────────────────────────
if __name__ == "__main__":
    build_dataset()
