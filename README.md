# Air Quality Prediction Using Time-Series Deep Learning 🌬️📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Tổng quan dự án (Project Overview)
Dự án áp dụng các mô hình học sâu chuỗi thời gian (Time-Series Deep Learning) như LSTM, GRU, BiLSTM để dự đoán chỉ số chất lượng không khí (AQI) và nồng độ bụi mịn PM2.5 dựa trên dữ liệu lịch sử và các yếu tố khí tượng học. 

## 📂 Cấu trúc thư mục (Directory Structure)
```text
air-quality-prediction/
├── data/               # Dữ liệu (Raw & Processed) - Đã được gitignore
├── models/             # Trọng số mô hình và Scalers - Đã được gitignore
├── notebooks/          # Jupyter notebooks cho EDA và thử nghiệm
├── src/                # Mã nguồn chính (Pipeline xử lý và huấn luyện)
├── app/                # Mã nguồn Web App Dashboard (Streamlit/FastAPI)
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Tài liệu hướng dẫn
