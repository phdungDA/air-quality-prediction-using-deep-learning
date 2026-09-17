import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Đọc các biến từ file .env (API_KEY, latitude, longtitude)
load_dotenv()

API_KEY  = os.getenv("API_KEY")
LAT      = os.getenv("latitude")
LON      = os.getenv("longtitude")

BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"


def fetch_air_quality(start_ts: int, end_ts: int) -> list[dict]:
    """
    Lấy dữ liệu chất lượng không khí từ OpenWeatherMap.

    Tham số:
        start_ts : Unix timestamp bắt đầu  (vd: 1788541200)
        end_ts   : Unix timestamp kết thúc (vd: 1788886800)

    Trả về:
        Danh sách các bản ghi, mỗi bản ghi là dict gồm:
        {
            "dt"      : int   – Unix timestamp
            "datetime": str   – Dạng đọc được, vd "2026-09-05 07:00:00"
            "aqi"     : int   – Chỉ số AQI (1=Tốt … 5=Rất xấu)
            "co"      : float – Carbon monoxide  (μg/m³)
            "no"      : float – Nitric oxide
            "no2"     : float – Nitrogen dioxide
            "o3"      : float – Ozone
            "so2"     : float – Sulfur dioxide
            "pm2_5"   : float – Bụi mịn PM2.5
            "pm10"    : float – Bụi PM10
            "nh3"     : float – Ammonia
        }
    """
    params = {
        "lat"   : LAT,
        "lon"   : LON,
        "start" : start_ts,
        "end"   : end_ts,
        "appid" : API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()          # Tự throw lỗi nếu status != 200

    raw_list = response.json().get("list", [])

    records = []
    for item in raw_list:
        records.append({
            "dt"       : item["dt"],
            "datetime" : datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
            "aqi"      : item["main"]["aqi"],
            "co"       : item["components"]["co"],
            "no"       : item["components"]["no"],
            "no2"      : item["components"]["no2"],
            "o3"       : item["components"]["o3"],
            "so2"      : item["components"]["so2"],
            "pm2_5"    : item["components"]["pm2_5"],
            "pm10"     : item["components"]["pm10"],
            "nh3"      : item["components"]["nh3"],
        })

    return records


# ── Chạy thử trực tiếp để kiểm tra ──────────────────────────────────────────
if __name__ == "__main__":
    # Lấy thử 4 tiếng gần đây (tính theo giờ UTC)
    from datetime import timezone, timedelta

    now        = datetime.now(timezone.utc)
    end_ts     = int(now.timestamp())
    start_ts   = int((now - timedelta(hours=4)).timestamp())

    print(f"Đang lấy data từ {datetime.utcfromtimestamp(start_ts)} → {datetime.utcfromtimestamp(end_ts)} UTC")

    data = fetch_air_quality(start_ts, end_ts)

    print(f"\n✅ Lấy được {len(data)} bản ghi. Ví dụ bản ghi đầu tiên:")
    for k, v in data[0].items():
        print(f"  {k:10} : {v}")
