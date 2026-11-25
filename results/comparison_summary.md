# Báo Cáo Đánh Giá So Sánh Mô Hình (UIT-VSFC Test Set - N=3.166)
**Dự án**: Project 1 — Text Classification of Vietnamese Student Feedback  
**Nhóm**: Team 25  

## 1. Bảng So Sánh Hiệu Năng Thực Nghiệm

| Mô hình | Sentiment Accuracy | Sentiment Macro-F1 | Topic Accuracy | Topic Macro-F1 | Tốc độ suy luận |
|---|---|---|---|---|---|
| **Classical Baseline (TF-IDF + LR)** | **87.81%** | **72.76%** | **82.66%** | **73.69%** | 48287.4 câu/s |
| **PhoBERT Multi-Task (Đề xuất)** | **93.97%** | **84.29%** | **88.72%** | **80.30%** | 17.3 câu/s |
| **Mức cải thiện của PhoBERT** | **+6.16%** | **+11.53%** | **+6.06%** | **+6.62%** | — |

## 2. Phân Tích Lỗi Sai Tiêu Biểu (Error Analysis)
### Mẫu 1 (PhoBERT vượt trội Baseline)
- **Câu phản hồi**: *"nói tiếng anh lưu loát ."*
- **Nhãn thực tế**: `positive | lecturer`
- **Baseline dự đoán**: `positive | others`
- **PhoBERT dự đoán**: `positive | lecturer`

### Mẫu 2 (Thử thách chung (câu khó / mơ hồ))
- **Câu phản hồi**: *"tính điểm thi đua các nhóm ."*
- **Nhãn thực tế**: `positive | training_program`
- **Baseline dự đoán**: `neutral | training_program`
- **PhoBERT dự đoán**: `neutral | training_program`

### Mẫu 3 (Thử thách chung (câu khó / mơ hồ))
- **Câu phản hồi**: *"trong trường macbook thầy số hai thì không có máy nào số một ."*
- **Nhãn thực tế**: `positive | others`
- **Baseline dự đoán**: `negative | facility`
- **PhoBERT dự đoán**: `negative | facility`

### Mẫu 4 (PhoBERT vượt trội Baseline)
- **Câu phản hồi**: *"cung cấp kiến thức bổ ích ."*
- **Nhãn thực tế**: `positive | lecturer`
- **Baseline dự đoán**: `positive | training_program`
- **PhoBERT dự đoán**: `positive | lecturer`

### Mẫu 5 (Thử thách chung (câu khó / mơ hồ))
- **Câu phản hồi**: *"cách mà cô tiếp cận với sinh viên ."*
- **Nhãn thực tế**: `neutral | lecturer`
- **Baseline dự đoán**: `positive | lecturer`
- **PhoBERT dự đoán**: `positive | lecturer`

### Mẫu 6 (Thử thách chung (câu khó / mơ hồ))
- **Câu phản hồi**: *"giữa lý thuyết từ vựng với trò chơi để dễ tiếp thu ."*
- **Nhãn thực tế**: `negative | training_program`
- **Baseline dự đoán**: `positive | training_program`
- **PhoBERT dự đoán**: `positive | training_program`

### Mẫu 7 (Thử thách chung (câu khó / mơ hồ))
- **Câu phản hồi**: *"phần lớn chỉ là lý thuyết và bài tập ."*
- **Nhãn thực tế**: `positive | training_program`
- **Baseline dự đoán**: `negative | training_program`
- **PhoBERT dự đoán**: `negative | training_program`

