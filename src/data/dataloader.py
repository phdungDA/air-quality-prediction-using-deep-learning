"""
Tải dữ liệu từ PostgreSQL thay vì gọi API trực tiếp.
"""

import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://admin:admin123@localhost:5432/postgres"

QUERY = """
SELECT dt, datetime, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
FROM air_quality
ORDER BY dt ASC;
"""


def load_from_db() -> pd.DataFrame:
    """Đọc toàn bộ dữ liệu từ PostgreSQL, trả về DataFrame."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(QUERY, conn)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"✅ Đọc được {len(df)} bản ghi từ PostgreSQL.")
    return df


if __name__ == "__main__":
    df = load_from_db()
    print(df.head())
    print(df.shape)
