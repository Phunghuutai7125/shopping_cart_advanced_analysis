
# Shopping Cart — Tóm tắt dự án (Clustering từ Association Rules)

Tài liệu này tóm tắt các bước đã thực hiện trong mini-project: từ khai thác luật kết hợp → chuyển luật thành đặc trưng khách hàng → ghép RFM → phân cụm → profiling → dashboard .

## Cấu trúc dự án
```
shopping_cart_advanced_analysis/
├── data/
│   ├── raw/                       # Dữ liệu gốc (online_retail.csv)
│   └── processed/                 # Kết quả pipeline (CSV, PNG, parquet)
├── notebooks/                     # Notebook khám phá và thử nghiệm
├── src/                           # Script và thư viện chính
│   ├── apriori_library.py         # Helpers cho mining và preprocessing
│   ├── cluster_from_rules.py      # Tạo feature từ luật
│   ├── cluster_pipeline.py        # Pipeline clustering
│   ├── compare_configurations.py  # So sánh cấu hình
│   └── streamlit_app.py           # Dashboard (Tiếng Việt)
├── requirements.txt               # Các phụ thuộc Python
└── README.md                      # Tài liệu này
```

## Tóm tắt những gì đã làm
- Tiền xử lý dữ liệu: `data/raw/online_retail.csv` → `data/processed/cleaned_uk_data.csv`.
- Chuẩn bị giỏ hàng và ma trận boolean: `data/processed/basket_bool.parquet`.
- Khai thác luật (Apriori & FP-Growth): `data/processed/rules_apriori_filtered.csv`, `data/processed/rules_fpgrowth_filtered.csv`.
- Chuyển luật thành đặc trưng khách hàng:
	- Nhị phân: `data/processed/customer_rule_features_binary.csv`
	- Weighted: `data/processed/customer_rule_features_weighted.csv`
- Tính RFM và kết hợp: `data/processed/customer_rule_features_rfm.csv`.
- Chọn số cụm (Silhouette / Elbow): `data/processed/k_selection_results.csv`.
- Huấn luyện KMeans và lưu nhãn: `data/processed/customer_clusters.csv`.
- Giảm chiều (PCA) để trực quan: `data/processed/cluster_pca.png`.
- Tạo hồ sơ & tên cụm (EN+VN): `data/processed/cluster_profile.csv`, `data/processed/cluster_profiles_named.csv`.
- So sánh cấu hình clustering: `data/processed/config_comparison.csv`.
- Dashboard Streamlit  với tải CSV: `src/streamlit_app.py`.

## Scripts chính
- `src/apriori_library.py` — helpers cho preprocessing, basket, mining.
- `src/cluster_from_rules.py` — mine rules → tạo binary features.
- `src/cluster_pipeline.py` — tạo weighted features, RFM, chọn K, phân cụm.
- `src/compare_configurations.py` — so sánh cấu hình clustering.
- `src/generate_cluster_profiles.py` — đặt tên và mô tả cụm.
- `src/streamlit_app.py` 

## Cách chạy (Windows)
1. Kích hoạt môi trường và cài thư viện:
```bash
conda activate shopping_env
pip install -r requirements.txt
```
2. Chạy pipeline (tuần tự):
```bash
python src/cluster_from_rules.py
python src/cluster_pipeline.py
python src/generate_cluster_profiles.py
python src/compare_configurations.py  # tùy chọn
```
3. Chạy dashboard Streamlit:
```bash
python -m streamlit run src/streamlit_app.py
# Mở http://localhost:8501
```

## Lưu ý kỹ thuật
- Apriori có thể tiêu thụ nhiều RAM trên ma trận lớn — nếu gặp lỗi, tăng `min_support` hoặc dùng FP-Growth.
- Đã fix các lỗi hay gặp: Windows console Unicode, lỗi mảng rỗng khi so sánh cấu hình.

## Trạng thái hiện tại
- Hầu hết pipeline và artifacts đã sinh ra trong `data/processed/`.
- Dashboard tiếng Việt đã sẵn sàng, có biểu đồ phân bố cụm, PCA tương tác (nếu dữ liệu weighted có sẵn), top rules và chức năng tải CSV.

## Gợi ý bước tiếp
- Tinh chỉnh UI Streamlit (biểu đồ, filter nâng cao).
- Thử thuật toán phân cụm khác (DBSCAN/HDBSCAN). 
- Review manual cho top rules (human-in-loop) để map đến SKU/nhóm sản phẩm.

---

## Lap4


### Mục tiêu
Phần lab này tập trung vào việc áp dụng thuật toán **Apriori** để:
- Phân tích dữ liệu giao dịch bán lẻ
- Khai thác các tập sản phẩm thường xuyên
- Xây dựng và đánh giá các luật kết hợp
- Trực quan hóa kết quả và rút ra nhận xét

### Các bước chính đã thực hiện
- Làm sạch dữ liệu và lọc theo quốc gia United Kingdom
- Phân tích khám phá dữ liệu (EDA) theo thời gian và sản phẩm
- Chuyển dữ liệu sang dạng giỏ hàng (basket / boolean matrix)
- Áp dụng thuật toán Apriori để sinh frequent itemsets
- Tạo và đánh giá Association Rules bằng các chỉ số:
  - Support
  - Confidence
  - Lift
- Trực quan hóa kết quả bằng biểu đồ Scatter và Network Graph

### Nhận xét
- Phần lớn các luật có **support thấp** do số lượng sản phẩm lớn và đa dạng.
- Các luật có **confidence cao** thể hiện khả năng mua kèm rõ rệt.
- **Lift > 1** cho thấy mối quan hệ mua chung mạnh hơn so với ngẫu nhiên.
- Kết quả có thể được ứng dụng trong gợi ý sản phẩm và bán chéo.

### Kết luận
Bài lab giúp hiểu rõ hơn về cách áp dụng thuật toán Apriori trong thực tế,
đồng thời cho thấy vai trò của khai phá dữ liệu trong việc hỗ trợ quyết định kinh doanh.

