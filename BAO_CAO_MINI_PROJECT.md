# BÁO CÁO MINI PROJECT

## DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ (AQI) DỰA TRÊN PM2.5

### Ứng dụng học bán giám sát (Semi-Supervised Learning)

---

## 1. Giới thiệu

Ô nhiễm không khí, đặc biệt là bụi mịn PM2.5, đang trở thành một trong những vấn đề môi trường nghiêm trọng tại các đô thị lớn. Nồng độ PM2.5 cao có ảnh hưởng trực tiếp đến sức khỏe con người và chất lượng sống. Vì vậy, việc dự báo và đánh giá chất lượng không khí (Air Quality Index – AQI) có ý nghĩa quan trọng trong công tác quản lý môi trường và cảnh báo sớm.

Trong thực tế, dữ liệu quan trắc không khí thường tồn tại dưới dạng dữ liệu lớn nhưng chỉ có một phần nhỏ được gán nhãn đầy đủ. Điều này gây khó khăn cho các phương pháp học có giám sát truyền thống. Do đó, trong mini project này, em tập trung nghiên cứu và triển khai các phương pháp **học bán giám sát (Semi-Supervised Learning)** nhằm khai thác hiệu quả cả dữ liệu có nhãn và chưa có nhãn để dự báo AQI dựa trên PM2.5.

---

## 2. Dữ liệu và tiền xử lý

### 2.1 Dữ liệu

Dữ liệu được sử dụng trong project là bộ dữ liệu **Beijing PM2.5**, bao gồm các thông tin quan trắc môi trường theo thời gian như:

* Nồng độ PM2.5
* Nhiệt độ (TEMP)
* Áp suất (PRES)
* Độ ẩm và điểm sương (DEWP)
* Lượng mưa (RAIN)
* Tốc độ gió (WSPM)
* Thông tin thời gian (năm, tháng, ngày, giờ)

### 2.2 Tiền xử lý dữ liệu

Các bước tiền xử lý được thực hiện bao gồm:

* Loại bỏ hoặc xử lý các giá trị bị thiếu
* Chuẩn hóa định dạng thời gian
* Tạo nhãn AQI từ giá trị PM2.5 theo các ngưỡng tiêu chuẩn
* Chia dữ liệu thành ba tập: tập có nhãn (labeled), tập chưa có nhãn (unlabeled) và tập kiểm tra (test)

Trong đó, chỉ một tỷ lệ nhỏ dữ liệu được giữ lại làm dữ liệu có nhãn để mô phỏng bài toán thiếu nhãn trong thực tế.

---

## 3. Phương pháp

### 3.1 Mô hình cơ sở (Baseline)

Mô hình baseline được xây dựng bằng thuật toán **HistGradientBoostingClassifier**. Mô hình này chỉ được huấn luyện trên tập dữ liệu có nhãn và đóng vai trò làm mốc so sánh cho các phương pháp học bán giám sát.

### 3.2 Self-Training

Self-Training là phương pháp học bán giám sát đơn giản, trong đó mô hình ban đầu được huấn luyện trên dữ liệu có nhãn. Sau đó, mô hình sẽ dự đoán nhãn cho dữ liệu chưa có nhãn và lựa chọn các mẫu có độ tin cậy cao để bổ sung vào tập huấn luyện. Quá trình này được lặp lại cho đến khi đạt số vòng lặp tối đa hoặc không còn đủ mẫu thỏa mãn ngưỡng tin cậy.

### 3.3 Co-Training

Co-Training sử dụng hai mô hình khác nhau, mỗi mô hình được huấn luyện trên một tập đặc trưng (view) riêng biệt:

* View 1: Đặc trưng thời gian và các đặc trưng trễ (lag features)
* View 2: Đặc trưng liên quan đến điều kiện thời tiết

Hai mô hình sẽ luân phiên gán nhãn giả cho dữ liệu chưa có nhãn của nhau, từ đó mở rộng tập huấn luyện một cách hiệu quả hơn so với self-training.

---

## 4. Đánh giá và kết quả

Các mô hình được đánh giá thông qua các chỉ số phổ biến như Accuracy, Precision, Recall và F1-score. Ngoài ra, project còn sử dụng confusion matrix và các biểu đồ so sánh để trực quan hóa hiệu năng của từng phương pháp.

Kết quả thực nghiệm cho thấy:

* Mô hình baseline cho kết quả thấp nhất do chỉ sử dụng dữ liệu có nhãn
* Self-Training giúp cải thiện hiệu năng nhờ khai thác thêm dữ liệu chưa có nhãn
* Co-Training cho kết quả tốt nhất trong hầu hết các chỉ số, đặc biệt khi hai tập đặc trưng có tính bổ sung lẫn nhau

---

## 5. Kết luận

Trong mini project này, em đã triển khai và so sánh các phương pháp học bán giám sát trong bài toán dự báo chất lượng không khí. Kết quả cho thấy các phương pháp Semi-Supervised Learning, đặc biệt là Co-Training, có khả năng cải thiện đáng kể hiệu năng so với mô hình học có giám sát truyền thống khi dữ liệu có nhãn bị hạn chế.

Hạn chế của project là chưa xem xét các yếu tố không gian và chưa thử nghiệm trên nhiều bộ dữ liệu khác nhau. Trong tương lai, có thể mở rộng bằng cách kết hợp thêm các phương pháp như Label Propagation hoặc các mô hình học sâu để nâng cao độ chính xác dự báo.

---

**Kết thúc báo cáo.**
