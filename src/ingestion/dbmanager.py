import csv
import psycopg2
from psycopg2.extras import execute_batch

# ── Kết nối PostgreSQL ────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname"  : "postgres",
    "user"    : "admin",
    "password": "admin123",
    "host"    : "localhost",
    "port"    : 5432,
}

RAW_CSV_PATH = "data/raw"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS air_quality (
    dt        BIGINT PRIMARY KEY,
    datetime  TIMESTAMP,
    aqi       INTEGER,
    co        FLOAT,
    no        FLOAT,
    no2       FLOAT,
    o3        FLOAT,
    so2       FLOAT,
    pm2_5     FLOAT,
    pm10      FLOAT,
    nh3       FLOAT
);
"""

INSERT_SQL = """
INSERT INTO air_quality (dt, datetime, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3)
VALUES (%(dt)s, %(datetime)s, %(aqi)s, %(co)s, %(no)s, %(no2)s, %(o3)s, %(so2)s, %(pm2_5)s, %(pm10)s, %(nh3)s)
ON CONFLICT (dt) DO NOTHING;
"""


def get_connection():
    """Tạo kết nối đến PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Tạo bảng air_quality nếu chưa có."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    print("✅ Bảng air_quality đã sẵn sàng.")


def load_from_csv(path: str = RAW_CSV_PATH) -> list[dict]:
    """Đọc dữ liệu từ file CSV raw."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "dt"      : int(row["dt"]),
                "datetime": row["datetime"],
                "aqi"     : int(row["aqi"]),
                "co"      : float(row["co"]),
                "no"      : float(row["no"]),
                "no2"     : float(row["no2"]),
                "o3"      : float(row["o3"]),
                "so2"     : float(row["so2"]),
                "pm2_5"   : float(row["pm2_5"]),
                "pm10"    : float(row["pm10"]),
                "nh3"     : float(row["nh3"]),
            })
    return records


def insert_records(records: list[dict]) -> None:
    """Đẩy toàn bộ records vào PostgreSQL, bỏ qua nếu đã tồn tại."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, INSERT_SQL, records, page_size=500)
        conn.commit()
    print(f"✅ Đã insert {len(records)} bản ghi vào PostgreSQL.")


# ── Chạy trực tiếp ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    print(f"Đang đọc data từ {RAW_CSV_PATH} ...")
    records = load_from_csv()
    print(f"✅ Đọc được {len(records)} bản ghi.")

    insert_records(records)
