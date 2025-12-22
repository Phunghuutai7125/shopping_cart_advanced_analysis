# Phân Tích Giỏ Hàng: Hành Trình Từ Dữ Liệu Đến Insight Kinh Doanh

## Bài Toán: Tại Sao Khách Hàng Mua Những Sản Phẩm Cùng Lúc?

Hãy tưởng tượng bạn đang đứng ở siêu thị, cầm giỏ hàng. Bạn thấy ai đó bỏ cà rốt, rồi thịt bò, rồi khoai tây vào giỏ. Tại sao lại là những món này? Có phải họ đang chuẩn bị nấu món hầm? Hay đơn giản là thói quen?

Đây chính là câu hỏi cốt lõi mà dự án này giải quyết: **"Khách hàng mua gì cùng nhau, và tại sao?"**

Chúng ta có dữ liệu bán lẻ từ một cửa hàng online ở UK, với hàng trăm nghìn giao dịch. Mỗi giao dịch ghi lại: ai mua, mua gì, bao nhiêu, giá bao nhiêu. Từ đó, chúng ta dùng **Association Rule Mining** để tìm ra các "mối quan hệ bí mật" giữa sản phẩm.

---

## Pipeline Thực Hiện: Từ Dữ Liệu Thô Đến Luật Kết Hợp

### Bước 1: Làm Sạch Dữ Liệu (Data Cleaning)
Dữ liệu thô luôn có vấn đề:
- Giao dịch bị hủy (InvoiceNo bắt đầu bằng 'C')
- Sản phẩm không tên hoặc giá âm
- Khách hàng không xác định

Chúng ta lọc ra chỉ dữ liệu hợp lệ từ UK, tạo ra tập dữ liệu sạch với 350k+ giao dịch.

### Bước 2: Xây Dựng Basket Matrix
Mỗi giao dịch thành một "basket" - danh sách sản phẩm được mua cùng lúc. Chúng ta biến đổi thành ma trận nhị phân: hàng là giao dịch, cột là sản phẩm, giá trị 1 nếu có mua.

Kết quả: ma trận 18k giao dịch × 3.8k sản phẩm.

### Bước 3: Khai Thác Luật Kết Hợp
Chúng ta dùng 2 thuật toán:

**Apriori**: "Nếu A thì B" - kiểm tra tất cả combinations có thể
**FP-Growth**: "Nén dữ liệu thành cây, rồi khai thác" - hiệu quả hơn nhiều

Cả hai đều tìm ra các luật như: "Nếu mua HERB MARKER PARSLEY và ROSEMARY thì 95% mua THYME"

### Bước 4: Đánh Giá và So Sánh
Mỗi luật được đánh giá bằng 3 chỉ số:
- **Support**: Tỷ lệ giao dịch chứa luật (độ phổ biến)
- **Confidence**: Xác suất B khi đã có A (độ tin cậy)
- **Lift**: Luật mạnh hơn ngẫu nhiên bao nhiêu (độ thú vị)

---

## Kết Quả So Sánh: Apriori vs FP-Growth

### Về Thời Gian: FP-Growth Nhanh Hơn Đáng Kể
Với min_support = 1%, FP-Growth chạy nhanh hơn Apriori khoảng 2-3 lần. Tại sao?

**Apriori** như người kiểm tra từng combination một: "A và B có đủ phổ biến? A,B,C thì sao?" - số lượng tăng exponential.

**FP-Growth** như người nén thông tin vào cây: "Tôi ghi nhớ pattern rồi khai thác một lần" - hiệu quả với dữ liệu lớn.

### Về Chất Lượng Luật: Hoàn Toàn Giống Nhau
Cả hai thuật toán sinh ra cùng 1.796 luật, với cùng support, confidence, lift. Điều này chứng tỏ implementation đúng.

### Về Độ Nhạy Tham Số: Cùng Pattern
Khi giảm min_support từ 1% xuống 0.5%, cả hai đều sinh ra nhiều luật hơn, nhưng FP-Growth vẫn nhanh hơn.

---

## Insight Nổi Bật: Bí Mật Trong Giỏ Hàng

### 1. **Đế Chế Herb Markers**
Các sản phẩm "HERB MARKER" (PARSLEY, ROSEMARY, THYME, MINT, BASIL, CHIVES) có mối liên kết cực mạnh (lift > 70). Tại sao?

Khách hàng đang mua nguyên liệu nấu ăn gia đình hoặc bán chuyên nghiệp. Họ không mua lẻ từng loại, mà mua cả bộ.

**Hành Động Cho Quản Lý**: Tạo "Herb Garden Starter Kit" - combo 3-4 loại với giá ưu đãi. Sắp xếp kệ herb cạnh nhau để tăng cross-selling.

### 2. **Sản Phẩm "Ngôi Sao" vs "Hub Tần Suất"**
Phân tích cho thấy:
- **Revenue Stars**: Sản phẩm giá cao, bán ít (như decor items) - đóng góp doanh thu lớn nhưng không thường xuyên
- **Frequency Hubs**: Sản phẩm giá rẻ, bán nhiều (như stationery) - xuất hiện khắp nơi nhưng doanh thu nhỏ

**Hành Động Cho Quản Lý**: Tối ưu inventory khác nhau. Revenue stars cần stock ít nhưng đảm bảo có sẵn. Frequency hubs cần dự trữ nhiều.

### 3. **Mùa Vụ và Thời Gian**
Luật kết hợp thay đổi theo thời gian. Mùa lễ hội có pattern khác mùa bình thường.

**Hành Động Cho Quản Lý**: Theo dõi seasonal patterns để điều chỉnh stock và promotion kịp thời.

### 4. **Giá Trị Thực Sự Của Luật**
Không chỉ đếm số lượng, mà tính **giá trị**. Một luật có confidence cao nhưng sản phẩm giá rẻ có thể không đáng quan tâm bằng luật với sản phẩm premium.

Chúng ta phát triển **weighted support**: tỷ lệ doanh thu từ luật đó trên tổng doanh thu.

### 5. **Mạng Liên Kết Sản Phẩm**
Network graph cho thấy HERB MARKER THYME là trung tâm - sản phẩm "hub" kết nối nhiều sản phẩm khác.

**Hành Động Cho Quản Lý**: Sử dụng hub products để recommend. "Khách mua THYME thì gợi ý ROSEMARY và PARSLEY".

---

## Kết Luận và Đề Xuất Hành Động

### Những Gì Chúng Ta Học Được
1. **Dữ liệu kể chuyện**: Mỗi con số là manh mối về hành vi khách hàng
2. **Thuật toán quan trọng**: FP-Growth hiệu quả hơn cho big data
3. **Business value**: Luật kết hợp không chỉ thú vị, mà tạo ra hành động kinh doanh thực tế

### Đề Xuất Tiếp Theo
1. **Triển khai hệ thống recommendation** dựa trên luật kết hợp
2. **A/B testing** các combo promotion
3. **Mở rộng sang weighted rules** - tính toán dựa trên giá trị, không chỉ tần suất
4. **Real-time analysis** khi khách hàng thêm sản phẩm vào giỏ

### Lời Nhắn Cuối
Data mining không phải về thuật toán phức tạp. Nó về việc hiểu con người qua những gì họ mua. Mỗi luật kết hợp là một câu chuyện nhỏ về cuộc sống hàng ngày. Và những câu chuyện đó có thể biến thành doanh thu.

---

## Tech Stack & Cách Chạy

### Công Nghệ Sử Dụng
- **Python** cho xử lý dữ liệu
- **MLxtend** cho Apriori/FP-Growth
- **Pandas** cho data manipulation
- **Matplotlib/Seaborn** cho visualization
- **Papermill** cho automation

### Cách Chạy Dự Án
```bash
# 1. Clone và setup environment
git clone <repo_url>
cd shopping_cart_advanced_analysis_lab2
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline
python run_papermill.py

# 3. Xem kết quả trong notebooks/runs/
```

### Cấu Trúc Dự Án
```
├── data/raw/online_retail.csv          # Dữ liệu gốc
├── data/processed/                      # Kết quả xử lý
├── notebooks/                           # Analysis notebooks
├── src/apriori_library.py              # Core functions
└── run_papermill.py                     # Automation script
```

---

*Tác giả: Data Scientist - Dự án phân tích giỏ hàng nâng cao*
*Ngày: $(date)*
*Phong cách: Feynman - giải thích cho người mới bắt đầu*
