# tensorflow==2.16.1 (đã cập nhật trong requirements.txt) là bản đầu tiên hỗ trợ chính thức Python 3.12
FROM python:3.12-slim

# Thư viện hệ thống cần cho tensorflow / build một số package C-extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài dependencies trước để tận dụng Docker layer cache
# (chỉ rebuild lại bước này khi requirements.txt đổi, không phải mỗi lần đổi code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào image
COPY . .

# Mặc định chạy script train baseline.
# Có thể override khi chạy: docker run <image> streamlit run app/app.py
CMD ["python", "src/models/train.py"]
