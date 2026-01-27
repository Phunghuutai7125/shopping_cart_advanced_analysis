# AIR GUARD: Dự báo chất lượng không khí dựa trên dữ liệu

## Giới thiệu

Dự án AIR GUARD áp dụng các thuật toán học bán giám sát (Semi-Supervised Learning) để dự báo chất lượng không khí (AQI) dựa trên nồng độ PM2.5, nhằm giải quyết bài toán khan hiếm dữ liệu có nhãn trong thực tế.

## Cấu trúc thư mục

```
air_guard/
├── data/                      # Dữ liệu thô và đã xử lý
├── src/                       # Source code
│   ├── preprocessing.py       # Tiền xử lý dữ liệu
│   ├── feature_engineering.py # Tạo đặc trưng
│   ├── models.py             # Các mô hình cơ bản
│   ├── self_training.py      # Thuật toán Self-Training
│   ├── co_training.py        # Thuật toán Co-Training
│   └── evaluation.py         # Đánh giá mô hình
├── notebooks/                # Jupyter notebooks thử nghiệm
├── results/                  # Kết quả thử nghiệm
├── dashboard/               # Streamlit dashboard
└── requirements.txt         # Thư viện cần thiết
```

## Phương pháp

### 1. Self-Training
- Huấn luyện mô hình trên tập dữ liệu nhỏ có nhãn
- Dự đoán nhãn cho dữ liệu chưa có nhãn
- Chọn các dự đoán có độ tin cậy cao (> ngưỡng τ)
- Thêm vào tập huấn luyện và lặp lại

### 2. Co-Training
- Sử dụng 2 mô hình với 2 view đặc trưng khác nhau
- View 1: Đặc trưng thời gian (giờ, ngày, tháng, lag PM2.5)
- View 2: Đặc trưng thời tiết (nhiệt độ, áp suất, độ ẩm, gió)
- Hai mô hình trao đổi nhãn giả cho nhau qua các vòng lặp

## Cài đặt

```bash
# Tạo môi trường ảo
conda create -n air_guard_env python=3.9
conda activate air_guard_env

# Cài đặt thư viện
pip install -r requirements.txt
```

## Sử dụng

### 1. Tiền xử lý dữ liệu
```bash
python src/preprocessing.py
```

### 2. Huấn luyện mô hình
```bash
# Baseline
python src/train_baseline.py

# Self-Training
python src/train_self_training.py

# Co-Training
python src/train_co_training.py
```

### 3. Chạy dashboard
```bash
streamlit run dashboard/app.py
```

## Kết quả

Các kết quả thử nghiệm được lưu trong thư mục `results/`:
- metrics_baseline.json
- metrics_self_training.json
- metrics_co_training.json
- visualization plots

## Tác giả

Mini Project - Data Mining Course
