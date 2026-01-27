# HƯỚNG DẪN SỬ DỤNG AIR GUARD PROJECT

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Cài đặt](#cài-đặt)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Chạy project](#chạy-project)
5. [Dashboard](#dashboard)
6. [Cấu trúc code](#cấu-trúc-code)
7. [Tùy chỉnh](#tùy-chỉnh)

---

## 🎯 Giới thiệu

AIR GUARD là một project học bán giám sát (Semi-Supervised Learning) để dự báo chất lượng không khí (AQI) dựa trên nồng độ PM2.5.

### Các thuật toán triển khai:
- **Baseline**: HistGradientBoostingClassifier
- **Self-Training**: Tự gán nhãn với độ tin cậy cao
- **Co-Training**: 2 models với 2 views đặc trưng khác nhau

---

## 🔧 Cài đặt

### Bước 1: Clone/Download project

```bash
# Download project từ link được cung cấp
# Hoặc copy thư mục air_guard
```

### Bước 2: Tạo môi trường ảo

```bash
# Dùng conda (khuyến nghị)
conda create -n air_guard_env python=3.9
conda activate air_guard_env

# Hoặc dùng venv
python -m venv air_guard_env
source air_guard_env/bin/activate  # Linux/Mac
# air_guard_env\Scripts\activate  # Windows
```

### Bước 3: Cài đặt thư viện

```bash
cd air_guard
pip install -r requirements.txt
```

---

## 📊 Chuẩn bị dữ liệu

### Tải dữ liệu Beijing PM2.5

Dữ liệu có thể tải từ:
- UCI Machine Learning Repository
- Kaggle: Beijing PM2.5 Dataset
- Link: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

### Đặt dữ liệu vào thư mục

```bash
air_guard/
├── data/
│   └── beijing_pm25.csv  # Đặt file dữ liệu ở đây
```

### Format dữ liệu cần có:

Các cột cần thiết:
- `year`, `month`, `day`, `hour` HOẶC `date`
- `PM2.5`: Nồng độ PM2.5
- `TEMP`: Nhiệt độ
- `PRES`: Áp suất
- `DEWP`: Điểm sương
- `RAIN`: Lượng mưa
- `WSPM`: Tốc độ gió

**Lưu ý**: Nếu không có dữ liệu, script sẽ tự động tạo dữ liệu mẫu để demo.

---

## 🚀 Chạy project

### Chạy toàn bộ pipeline

```bash
cd src
python main.py
```

Script sẽ thực hiện:
1. Tiền xử lý dữ liệu
2. Feature engineering
3. Huấn luyện Baseline
4. Huấn luyện Self-Training
5. Huấn luyện Co-Training
6. So sánh kết quả
7. Lưu kết quả vào thư mục `results/`

### Kết quả output

```
results/
├── metrics_baseline.json
├── metrics_self_training.json
├── metrics_co_training.json
├── history_self_training.json
├── history_co_training.json
├── self_training_history.png
├── model_comparison.png
├── per_class_comparison.png
├── cm_baseline.png
├── cm_self_training.png
└── cm_co_training.png
```

---

## 📱 Dashboard

### Chạy Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard sẽ mở tại: http://localhost:8501

### Các trang trong Dashboard:

1. **Tổng quan**: Giới thiệu project và kết quả tổng quan
2. **Baseline Model**: Chi tiết mô hình cơ sở
3. **Self-Training**: Quá trình và kết quả Self-Training
4. **Co-Training**: Quá trình và kết quả Co-Training
5. **So sánh**: So sánh hiệu năng các models

---

## 📁 Cấu trúc code

### 1. preprocessing.py

Module tiền xử lý dữ liệu:
- `DataPreprocessor`: Class chính
  - `load_data()`: Đọc dữ liệu
  - `clean_data()`: Làm sạch dữ liệu
  - `create_aqi_labels()`: Tạo nhãn AQI
  - `train_test_split()`: Chia train/test
  - `create_labeled_unlabeled_split()`: Tạo labeled/unlabeled

**Sử dụng:**

```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(cutoff_date='2017-01-01')
data = preprocessor.preprocess_pipeline(
    'data/beijing_pm25.csv',
    labeled_ratio=0.1
)

labeled_df = data['labeled']
unlabeled_df = data['unlabeled']
test_df = data['test']
```

### 2. feature_engineering.py

Module tạo đặc trưng:
- `FeatureEngineer`: Class chính
  - `create_temporal_features()`: Đặc trưng thời gian
  - `create_lag_features()`: Đặc trưng lag
  - `create_weather_features()`: Đặc trưng thời tiết
  - `get_feature_views()`: Lấy 2 views cho co-training

**Sử dụng:**

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df = engineer.feature_engineering_pipeline(df, create_lags=True)
X, y = engineer.prepare_features_labels(df)

# Lấy 2 views cho co-training
view1, view2 = engineer.get_feature_views()
```

### 3. self_training.py

Module Self-Training:
- `SelfTraining`: Class chính
  - `fit()`: Huấn luyện
  - `predict()`: Dự đoán
  - `get_history()`: Lấy lịch sử
  - `plot_history()`: Vẽ biểu đồ

**Sử dụng:**

```python
from self_training import SelfTraining

model = SelfTraining(
    confidence_threshold=0.9,
    max_iter=10,
    min_new_per_iter=20
)

model.fit(
    X_labeled=X_train,
    y_labeled=y_train,
    X_unlabeled=X_unlabeled,
    X_val=X_val,
    y_val=y_val
)

predictions = model.predict(X_test)
model.plot_history(save_path='history.png')
```

### 4. co_training.py

Module Co-Training:
- `CoTraining`: Class chính
  - `fit()`: Huấn luyện
  - `predict()`: Dự đoán (với ensemble)
  - `get_history()`: Lấy lịch sử

**Sử dụng:**

```python
from co_training import CoTraining

model = CoTraining(
    confidence_threshold=0.9,
    max_iter=10,
    max_new_per_model=100
)

model.fit(
    X_labeled=X_train,
    y_labeled=y_train,
    X_unlabeled=X_unlabeled,
    view1_features=view1,
    view2_features=view2,
    X_val=X_val,
    y_val=y_val
)

predictions = model.predict(X_test, use_ensemble=True)
```

### 5. evaluation.py

Module đánh giá:
- `ModelEvaluator`: Class chính
  - `evaluate()`: Tính metrics
  - `plot_confusion_matrix()`: Vẽ confusion matrix
  - `compare_models()`: So sánh models

**Sử dụng:**

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, "Model Name")
evaluator.print_evaluation(metrics)
evaluator.plot_confusion_matrix(y_true, y_pred)
```

---

## ⚙️ Tùy chỉnh

### Thay đổi tham số trong main.py

```python
# Tỷ lệ dữ liệu có nhãn
LABELED_RATIO = 0.1  # 10%

# Ngưỡng tin cậy
CONFIDENCE_THRESHOLD = 0.9  # 90%

# Số vòng lặp
MAX_ITER = 10

# Số mẫu tối thiểu mỗi vòng
MIN_NEW_PER_ITER = 20
```

### Thử nghiệm với ngưỡng khác

```python
# Trong Self-Training
self_trainer = SelfTraining(
    confidence_threshold=0.85,  # Thử 85% thay vì 90%
    max_iter=15
)

# Trong Co-Training
co_trainer = CoTraining(
    confidence_threshold=0.92,  # Thử 92%
    max_new_per_model=150       # Tăng số mẫu mỗi vòng
)
```

### Thay đổi mô hình cơ sở

```python
from sklearn.ensemble import RandomForestClassifier

# Dùng Random Forest thay vì HistGradientBoosting
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

self_trainer = SelfTraining(base_model=base_model)
```

### Tạo views features khác

```python
# Trong feature_engineering.py
def get_custom_views(self):
    """Tạo views tùy chỉnh"""
    # View 1: Chỉ temporal
    view1 = self.temporal_features
    
    # View 2: Chỉ lag + weather
    view2 = self.lag_features + self.weather_features
    
    return view1, view2
```

---

## 📝 Yêu cầu bài tập

### Phần bắt buộc:

1. ✅ Huấn luyện Self-Training
   - Thử nghiệm ngưỡng τ khác nhau
   - Trình bày biểu đồ diễn biến
   - So sánh với baseline

2. ✅ Huấn luyện Co-Training
   - Mô tả 2 views features
   - Theo dõi diễn biến 2 models
   - So sánh với self-training

3. ✅ So sánh tham số
   - Thử nghiệm ít nhất 1 cấu hình khác
   - Phân tích kết quả

4. ✅ Dashboard Streamlit
   - Trực quan hóa kết quả
   - Dễ sử dụng và hiểu

### Phần nâng cao (khuyến khích):

- Label Propagation/Spreading
- Dynamic Threshold (FlexMatch)
- Focal Loss
- Ensemble methods

---

## 🐛 Xử lý lỗi

### Lỗi: "File not found"
```bash
# Đảm bảo đặt dữ liệu đúng chỗ
ls data/beijing_pm25.csv

# Hoặc để script tạo dữ liệu mẫu tự động
```

### Lỗi: "Module not found"
```bash
# Cài lại requirements
pip install -r requirements.txt

# Kiểm tra môi trường đã activate
conda activate air_guard_env
```

### Lỗi: "No module named 'src'"
```bash
# Chạy từ thư mục src/
cd src
python main.py
```

---

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Python version: 3.8+
2. Tất cả thư viện đã cài đặt
3. Dữ liệu đúng format
4. Chạy từ đúng thư mục

---

## 🎓 Tài liệu tham khảo

- Scikit-learn Documentation
- Self-Training Paper
- Co-Training Paper
- Beijing PM2.5 Dataset

---

**Chúc bạn thành công với project!** 🚀
