# Shopping Cart Analysis

Phân tích dữ liệu bán lẻ nhằm khám phá mối quan hệ giữa các sản phẩm thường được mua cùng nhau bằng các kỹ thuật **Association Rule Mining** như **Apriori** và **FP-Growth**.  
Project triển khai pipeline đầy đủ từ xử lý dữ liệu → khai thác luật → so sánh thuật toán → trực quan hóa kết quả.

---

## Features

- Làm sạch dữ liệu & xử lý giao dịch lỗi
- Xây dựng basket matrix (transaction × product)
- Khai thác tập mục phổ biến (Frequent Itemsets)
- Sinh luật kết hợp (Association Rules)
- Hỗ trợ 2 thuật toán:
  - Apriori
  - FP-Growth
- So sánh Apriori vs FP-Growth
- Các chỉ số đánh giá:
  - Support
  - Confidence
  - Lift
- Trực quan hóa với:
  - Bar chart
  - Scatter plot
  - Network graph
  - Biểu đồ tương tác Plotly
- Tự động hóa pipeline bằng **Papermill**
- Dashboard tương tác bằng **Streamlit**

---

## Project Structure

```text
shopping_cart_advanced_analysis/
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   └── processed/
│       ├── cleaned_uk_data.csv
│       ├── basket_bool.parquet
│       ├── rules_apriori_filtered.csv
│       └── rules_fpgrowth_filtered.csv
│
├── notebooks/
│   ├── preprocessing_and_eda.ipynb
│   ├── basket_preparation.ipynb
│   ├── apriori_modelling.ipynb
│   ├── fp_growth_modelling.ipynb
│   ├── compare_apriori_fpgrowth.ipynb
│   └── runs/
│       ├── preprocessing_and_eda_run.ipynb
│       ├── basket_preparation_run.ipynb
│       ├── apriori_modelling_run.ipynb
│       ├── fp_growth_modelling_run.ipynb
│       └── compare_apriori_fpgrowth_run.ipynb
│
├── src/
│   └── apriori_library.py
│
├── dashboard/
│   ├── app.py
│   └── requirements.txt
│
├── run_papermill.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <your_repo_url>
cd shopping_cart_advanced_analysis
conda create -n shopping_env python=3.11
conda activate shopping_env
pip install -r requirements.txt
```

Data Preparation
Đặt file gốc tại:

```bash
data/raw/online_retail.csv
```
File output sẽ được sinh tự động vào:

```bash
data/processed/
```

Run Pipeline (Recommended)
Chạy toàn bộ phân tích chỉ với 1 lệnh:

```bash
python run_papermill.py
```
Kết quả sinh ra:

```bash
data/processed/
├── cleaned_uk_data.csv
├── basket_bool.parquet
├── rules_apriori_filtered.csv
└── rules_fpgrowth_filtered.csv

notebooks/runs/
├── preprocessing_and_eda_run.ipynb
├── basket_preparation_run.ipynb
├── apriori_modelling_run.ipynb
├── fp_growth_modelling_run.ipynb
└── compare_apriori_fpgrowth_run.ipynb
```

### Changing Parameters
Các tham số có thể chỉnh trong `run_papermill.py` hoặc trong cell `PARAMETERS` của mỗi notebook:

```python
MIN_SUPPORT=0.01
MAX_LEN=3
FILTER_MIN_CONF=0.3
FILTER_MIN_LIFT=1.2
```
Papermill cho phép chạy pipeline với cấu hình khác nhau mà không cần sửa notebook gốc.

### Visualization & Results
Các notebook modelling hiển thị các biểu đồ:

Top luật theo Lift

Top luật theo Confidence

Scatter Support – Confidence – Lift

Network graph giữa các sản phẩm

Biểu đồ Plotly tương tác

Có thể export notebook kết quả sang HTML:

```bash
jupyter nbconvert notebooks/runs/priori_modelling_run.ipynb --to html
```

### Ứng dụng thực tế
Product recommendation

Cross-selling strategy

Combo gợi ý sản phẩm

Phân tích hành vi mua hàng

Sắp xếp sản phẩm tại siêu thị

### Tech Stack

| Công nghệ | Mục đích |
|----------|----------|
| Python | Ngôn ngữ chính |
| Pandas | Xử lý dữ liệu transaction |
| MLxtend | Apriori / FP-Growth association rules |
| Papermill | Chạy pipeline notebook tự động |
| Matplotlib & Seaborn | Visualization biểu đồ tĩnh |
| Plotly | Dashboard / biểu đồ tương tác |
| Jupyter Notebook | Môi trường notebook |

### Roadmap
Streamlit dashboard

Weighted association rules

Correlation-aware rule ranking


### Author
Project được thực hiện bởi:
Trang Le

📄 License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.
