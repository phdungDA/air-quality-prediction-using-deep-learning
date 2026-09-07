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
```

##📊 Tập dữ liệu (Dataset)
Nguồn: [Tên nguồn/API, ví dụ: EPA, OpenWeatherMap, hoặc trạm quan trắc địa phương]

Đặc trưng (Features): PM2.5, PM10, NO2, Nhiệt độ, Độ ẩm, Tốc độ gió...

Target: Dự đoán PM2.5 hoặc AQI cho 24-48 giờ tới.

##🚀 Hướng dẫn cài đặt (Installation)
1. Clone repository:
```text
git clone [https://github.com/phdungDA/air-quality-prediction-using-deep-learning.git](https://github.com/phdungDA/air-quality-prediction-using-deep-learning.git)
cd air-quality-prediction
```

2. Tạo môi trường ảo và cài đặt thư viện:
```text
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tải dữ liệu và mô hình (Pre-trained):

Tải file dataset và đặt vào thư mục data/raw/.

Tải file model weights và đặt vào thư mục models/.

##⚙️ Hướng dẫn sử dụng (Usage)
1. Huấn luyện lại mô hình (Training):
```text
python src/models/train.py
```

2. Khởi chạy Ứng dụng Web / Dashboard (Inference):
```text
streamlit run app/app.py
```

##📈 Hiệu năng mô hình (Model Performance)

| Mô hình | RMSE | MAE | R2 |
| :--- | :---: | :---: | :---: |
| Baseline | Index | Index | Index |
| LSTM |  Index | Index | Index |
| GRU | Index | Index | Index |

##👥 Nhóm phát triển (Contributors)
[phdungDA] - role: Data Engineer
---bổ sung thêm tên và vị trí vào đây
