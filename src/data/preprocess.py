"""
Tiền xử lý dữ liệu thô → sequences cho LSTM/GRU.

Module này là entry point để:
  - Chạy toàn bộ pipeline lần đầu (build_dataset)
  - Hoặc load lại tập đã xử lý (load_processed) khi đã có sẵn

Pipeline thực tế nằm trong src/data/dataloader.py:
  load CSV → handle missing → split 70/15/15 → MinMaxScale → sliding window
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataloader import (                                # noqa: F401  (re-export)
    build_dataset, load_processed, PROCESSED_DIR, RAW_CSV_PATH,
)


def run_preprocessing(force: bool = False):
    """
    Kiểm tra xem data/processed/ đã có đủ file chưa.
    - Nếu chưa có (hoặc force=True): chạy build_dataset() từ CSV.
    - Nếu đã có: load_processed() để tiết kiệm thời gian.

    Trả về: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    required_files = [
        "X_train.npy", "y_train.npy",
        "X_val.npy",   "y_val.npy",
        "X_test.npy",  "y_test.npy",
        "scaler.pkl",
    ]
    all_exist = all(
        os.path.exists(os.path.join(PROCESSED_DIR, f)) for f in required_files
    )

    # CSV raw mới hơn processed (vd vừa fetch lại 3 năm) → processed đã cũ, phải build lại
    stale = False
    if all_exist and os.path.exists(RAW_CSV_PATH):
        stale = os.path.getmtime(RAW_CSV_PATH) > os.path.getmtime(
            os.path.join(PROCESSED_DIR, "X_train.npy")
        )
        if stale:
            print("⚠️  CSV raw mới hơn data/processed/ → build lại.")

    if all_exist and not force and not stale:
        print("📂 Tìm thấy data/processed/ — load lại thay vì build lại.")
        return load_processed()
    else:
        print("🔄 Chạy pipeline tiền xử lý từ đầu...")
        return build_dataset(save=True)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()
    print("\n✅ Preprocessing hoàn tất.")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   X_test  : {X_test.shape}")
