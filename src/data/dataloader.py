"""
Tải dữ liệu thô từ OpenWeatherMap Air Pollution History API.
LƯU Ý: schema dữ liệu chưa chốt, đây là bản tạm dùng thẳng response gốc của API
       (aqi 1-5 + các thành phần ô nhiễm), sẽ đổi khi có schema chính thức.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
LAT = os.getenv("latitude", "21.0285")
LON = os.getenv("longtitude", "105.8542")

BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"


def fetch_air_pollution_history(start: int, end: int, lat: str = LAT, lon: str = LON) -> pd.DataFrame:
    """
    Gọi API và trả về DataFrame thô.
    Mỗi dòng là 1 bản ghi theo giờ gồm: dt, aqi (1-5), và các thành phần ô nhiễm
    (co, no, no2, o3, so2, pm2_5, pm10, nh3).
    """
    params = {"lat": lat, "lon": lon, "start": start, "end": end, "appid": API_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    records = []
    for item in payload.get("list", []):
        row = {
            "dt": item["dt"],
            "aqi": item["main"]["aqi"],
            **item["components"],
        }
        records.append(row)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["dt"], unit="s")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # test nhanh: lấy 1 ngày dữ liệu
    df = fetch_air_pollution_history(start=1757350800, end=1757437200)
    print(df.head())
    print(df.shape)
