"""
Lấy dữ liệu chất lượng không khí lịch sử từ OpenWeatherMap và lưu ra CSV.

Mặc định lấy khoảng thời gian CỐ ĐỊNH 09/09/2023 → 09/09/2026 (UTC) để cả nhóm
có cùng một bộ dữ liệu, kết quả các lần train so sánh được với nhau.
Khoảng dài được chia thành nhiều request (mỗi chunk CHUNK_DAYS ngày) để tránh
response quá lớn / timeout, có retry khi lỗi mạng, 429 hoặc 5xx.

Chạy từ thư mục gốc project:
    python src/ingestion/api_client.py                      # 09/09/2023 → 09/09/2026
    python src/ingestion/api_client.py --start 2024-01-01 --end 2025-01-01   # ghi đè khi cần thử nghiệm
"""

import os
import csv
import time
import argparse
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Đọc các biến từ file .env (API_KEY, latitude, longtitude)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("API_KEY")
LAT     = os.getenv("latitude")  or os.getenv("LAT")
LON     = os.getenv("longtitude") or os.getenv("longitude") or os.getenv("LON")

BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# ── Cấu hình lấy dữ liệu ─────────────────────────────────────────────────────
# Khoảng thời gian cố định (UTC). Đổi ở đây nếu cả nhóm thống nhất mốc mới.
DEFAULT_START   = "2023-09-09"
DEFAULT_END     = "2026-09-09"
CHUNK_DAYS      = 30            # mỗi request lấy tối đa 30 ngày (~720 bản ghi/giờ)
REQUEST_TIMEOUT = 30            # giây
MAX_RETRIES     = 4
RETRY_BACKOFF   = 2             # giây; chờ 2, 4, 8, ... giữa các lần thử lại
SLEEP_BETWEEN   = 1.1           # giây; free tier giới hạn 60 calls/phút

# OpenWeatherMap chỉ có dữ liệu lịch sử air pollution từ 2020-11-27 00:00 UTC
MIN_HISTORY_TS = 1606435200

# Khớp với src/data/dataloader.py (RAW_CSV_PATH)
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "air_quality_3years.csv")


# ── Tiện ích thời gian ───────────────────────────────────────────────────────
def _utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _fmt(ts: int) -> str:
    return _utc(ts).strftime("%Y-%m-%d %H:%M:%S")


def _date_to_ts(text: str) -> int:
    """'YYYY-MM-DD' → Unix timestamp (00:00 UTC)."""
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def resolve_range(start: str | None = None,
                  end: str | None = None) -> tuple[int, int]:
    """
    Xác định (start_ts, end_ts) theo giờ UTC:
      - start : --start nếu có, ngược lại DEFAULT_START (2023-09-09)
      - end   : --end   nếu có, ngược lại DEFAULT_END   (2026-09-09)
    """
    start_ts = _date_to_ts(start or DEFAULT_START)
    end_ts   = _date_to_ts(end or DEFAULT_END)

    if start_ts >= end_ts:
        raise ValueError("start phải nhỏ hơn end.")
    return start_ts, end_ts


# ── Gọi API ──────────────────────────────────────────────────────────────────
def _check_config() -> None:
    missing = [name for name, val in
               (("API_KEY", API_KEY), ("latitude", LAT), ("longtitude", LON)) if not val]
    if missing:
        raise RuntimeError(f"Thiếu biến trong .env: {', '.join(missing)}")


def _request_chunk(session: requests.Session, start_ts: int, end_ts: int) -> list[dict]:
    """Gọi API cho 1 chunk, retry khi lỗi mạng / 429 / 5xx. Lỗi 4xx khác (vd 401) raise ngay."""
    params = {"lat": LAT, "lon": LON, "start": start_ts, "end": end_ts, "appid": API_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json().get("list", [])

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise                       # lỗi không thể retry (401, 400, ...)
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * 2 ** (attempt - 1)
            print(f"   ⚠️  Lỗi ({e}); thử lại lần {attempt}/{MAX_RETRIES - 1} sau {wait}s...")
            time.sleep(wait)

    return []


def _iter_chunks(start_ts: int, end_ts: int, chunk_days: int):
    step = chunk_days * 24 * 3600
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + step, end_ts)
        yield cur, nxt
        cur = nxt


def _to_record(item: dict) -> dict:
    comp = item["components"]
    return {
        "dt"       : item["dt"],
        "datetime" : _fmt(item["dt"]),
        "aqi"      : item["main"]["aqi"],
        "co"       : comp["co"],
        "no"       : comp["no"],
        "no2"      : comp["no2"],
        "o3"       : comp["o3"],
        "so2"      : comp["so2"],
        "pm2_5"    : comp["pm2_5"],
        "pm10"     : comp["pm10"],
        "nh3"      : comp["nh3"],
    }


def fetch_air_quality(start_ts: int, end_ts: int, chunk_days: int = CHUNK_DAYS) -> list[dict]:
    """
    Lấy dữ liệu chất lượng không khí từ OpenWeatherMap cho khoảng [start_ts, end_ts].

    Khoảng thời gian dài được chia thành nhiều chunk, gộp lại, bỏ trùng theo `dt`
    và sắp xếp tăng dần theo thời gian.

    Tham số:
        start_ts   : Unix timestamp bắt đầu
        end_ts     : Unix timestamp kết thúc
        chunk_days : số ngày mỗi request

    Trả về:
        Danh sách bản ghi, mỗi bản ghi là dict gồm:
        {
            "dt"      : int   – Unix timestamp
            "datetime": str   – Dạng đọc được (UTC), vd "2025-09-09 00:00:00"
            "aqi"     : int   – Chỉ số AQI (1=Tốt … 5=Rất xấu)
            "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3" : float (μg/m³)
        }
    """
    _check_config()

    if start_ts < MIN_HISTORY_TS:
        print(f"⚠️  API chỉ có dữ liệu từ {_fmt(MIN_HISTORY_TS)} UTC → tự chỉnh start.")
        start_ts = MIN_HISTORY_TS

    chunks = list(_iter_chunks(start_ts, end_ts, chunk_days))
    by_dt: dict[int, dict] = {}

    with requests.Session() as session:
        for i, (c_start, c_end) in enumerate(chunks, start=1):
            items = _request_chunk(session, c_start, c_end)
            for item in items:
                by_dt[item["dt"]] = _to_record(item)     # trùng dt → ghi đè, không nhân đôi
            print(f"   [{i:>3}/{len(chunks)}] {_fmt(c_start)} → {_fmt(c_end)} : "
                  f"{len(items)} bản ghi (tổng unique: {len(by_dt)})")
            if i < len(chunks):
                time.sleep(SLEEP_BETWEEN)

    return [by_dt[k] for k in sorted(by_dt)]


# ── Lưu CSV ──────────────────────────────────────────────────────────────────
def _prepare_output_dir(path: str) -> None:
    """
    Đảm bảo thư mục chứa file tồn tại.
    Bản cũ lưu thẳng ra `data/raw` (file không đuôi) → đổi tên thành backup để tạo được thư mục.
    """
    out_dir = os.path.dirname(path)
    if os.path.isfile(out_dir):
        backup = out_dir + "_1year_legacy.csv"
        os.replace(out_dir, backup)
        print(f"📦 '{out_dir}' đang là file (bản 1 năm cũ) → đã đổi tên thành '{backup}'")
    os.makedirs(out_dir, exist_ok=True)


def save_to_csv(records: list[dict], path: str = RAW_CSV_PATH) -> None:
    """Ghi đè toàn bộ records ra file CSV (ghi file tạm rồi thay thế để tránh hỏng file)."""
    if not records:
        raise ValueError("Không có bản ghi nào để lưu.")

    _prepare_output_dir(path)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp_path, path)

    print(f"✅ Đã lưu {len(records)} bản ghi vào {path}")


# ── Chạy trực tiếp ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch lịch sử chất lượng không khí từ OpenWeatherMap")
    parser.add_argument("--start", help=f"Ngày bắt đầu YYYY-MM-DD (mặc định {DEFAULT_START})")
    parser.add_argument("--end",   help=f"Ngày kết thúc YYYY-MM-DD (mặc định {DEFAULT_END})")
    parser.add_argument("--out",   default=RAW_CSV_PATH, help="Đường dẫn file CSV đầu ra")
    args = parser.parse_args()

    start_ts, end_ts = resolve_range(args.start, args.end)
    print(f"Đang lấy data từ {_fmt(start_ts)} → {_fmt(end_ts)} UTC")

    data = fetch_air_quality(start_ts, end_ts)
    if not data:
        raise SystemExit("❌ API không trả về bản ghi nào.")

    expected = (end_ts - max(start_ts, MIN_HISTORY_TS)) // 3600
    print(f"\n✅ Lấy được {len(data)} bản ghi (~{len(data) / expected:.1%} so với {expected} giờ kỳ vọng).")
    print("Bản ghi đầu tiên:")
    for k, v in data[0].items():
        print(f"  {k:10} : {v}")
    print(f"Bản ghi cuối cùng: {data[-1]['datetime']}")

    save_to_csv(data, args.out)


if __name__ == "__main__":
    main()
