# Project 1 — Text Classification of Vietnamese Student Feedback

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Tr-Hieu205/DoAn_TriTueNhanTaoNangCao&branch=main&mainModule=app/app.py)
[![GitHub Release Weights](https://img.shields.io/badge/Weights-PhoBERT%20v1.0.0%20(516%20MB)-8B5CF6)](https://github.com/Tr-Hieu205/DoAn_TriTueNhanTaoNangCao/releases/tag/v1.0.0)

**🌐 Web Demo Trực Tuyến (Streamlit Cloud)**: [Mở / Khởi chạy Web App trên Streamlit Cloud](https://share.streamlit.io/deploy?repository=Tr-Hieu205/DoAn_TriTueNhanTaoNangCao&branch=main&mainModule=app/app.py)

**Team 25**:
- Lê Văn Hoài Rin (Student ID: 29211332113)
- Nguyễn Trung Hiếu (Student ID: 29211362521)

Dự án máy học và xử lý ngôn ngữ tự nhiên phân loại phản hồi sinh viên tiếng Việt dựa trên bộ dữ liệu học thuật **UIT-VSFC** (Vietnamese Students' Feedback Corpus).
Nghiên cứu so sánh thực nghiệm hai phương pháp theo đúng yêu cầu đề bài:
1. **Classical Baseline**: TF-IDF n-gram (1-2) + Dual Balanced Logistic Regression.
2. **Main Model (Proposed)**: PhoBERT Multi-Task (`vinai/phobert-base-v2` + Masked Mean Pooling + Dual MLP Heads) phân loại đồng thời cả **Cảm xúc** (3 lớp) và **Chủ đề** (4 lớp).

---

## 1. Cài Đặt (Setup)

Yêu cầu Python 3.10+ và cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

---

## 2. Dữ Liệu (Data)

Bộ dữ liệu **UIT-VSFC** chính thức được tải tự động từ Hugging Face endpoint về `data/uit_vsfc/`:
```bash
python data/download_data.py
```
Quy mô phân chia chuẩn (không rò rỉ dữ liệu giữa các tập):
- **Tập huấn luyện (Train)**: 11.426 câu
- **Tập kiểm tra (Validation)**: 1.583 câu
- **Tập kiểm thử độc lập (Test)**: 3.166 câu

**Hai tác vụ phân loại**:
- **Cảm xúc (Sentiment - 3 nhãn)**: `0: negative` (tiêu cực), `1: neutral` (trung tính), `2: positive` (tích cực).
- **Chủ đề (Topic - 4 nhãn)**: `0: lecturer` (giảng viên), `1: training_program` (chương trình đào tạo), `2: facility` (cơ sở vật chất), `3: others` (khác).

---

## 3. Hướng Dẫn Chạy (How to Run)

### ⚠️ Trọng số mô hình đã huấn luyện sẵn (Pretrained Checkpoints)
- **Classical Baseline**: File `saved_models/baseline.joblib` (1.23 MB) có sẵn trực tiếp trong repo.
- **PhoBERT Multi-Task**: File `saved_models/phobert/model.pt` (516.56 MB) vượt giới hạn 100 MB của GitHub nên được lưu trữ tại [GitHub Releases v1.0.0](https://github.com/Tr-Hieu205/DoAn_TriTueNhanTaoNangCao/releases/tag/v1.0.0):
  - **Tự động**: Khi chạy `streamlit run app/app.py` hoặc `python evaluation/evaluate.py`, code sẽ **tự động tải** file `model.pt` từ GitHub Releases về thư mục `saved_models/phobert/`.
  - **Thủ công**: Tải trực tiếp từ [link tải model.pt](https://github.com/Tr-Hieu205/DoAn_TriTueNhanTaoNangCao/releases/download/v1.0.0/model.pt) rồi đặt vào `saved_models/phobert/model.pt`.

### A. Huấn luyện mô hình Baseline cổ điển (~2 giây):
```bash
python training/train_baseline.py
```
*Trọng số lưu tại `saved_models/baseline.joblib`.*

### B. Huấn luyện mô hình PhoBERT Multi-Task:
```bash
python training/train_phobert.py
```
*Trọng số lưu tại `saved_models/phobert/model.pt`.*

### C. Đánh giá và so sánh thực nghiệm trên tập Test (N=3.166):
```bash
python evaluation/evaluate.py
```

### D. Khởi chạy Web Demo (Feedback Analyzer Dashboard):
```bash
streamlit run app/app.py
```

### E. Chạy kiểm thử tự động:
```bash
python tests/test_core.py
```

---

## 4. Kết Quả Thực Nghiệm (Results)

Đánh giá thực tế trên toàn bộ 3.166 câu của tập Test UIT-VSFC:

| Mô hình | Sentiment Accuracy | Sentiment Macro-F1 | Topic Accuracy | Topic Macro-F1 | Tốc độ suy luận |
|---|---|---|---|---|---|
| **Classical Baseline (TF-IDF + LR)** | **87.81%** | **72.76%** | **82.66%** | **73.69%** | ~48.200 câu/giây |
| **PhoBERT Multi-Task (Đề xuất)** | **93.97%** | **84.29%** | **88.72%** | **80.30%** | ~17.3 câu/giây |
| **Mức cải thiện của PhoBERT** | **+6.16%** | **+11.53%** | **+6.06%** | **+6.62%** | — |

**Nhận xét**: PhoBERT Multi-Task vượt trội mô hình Baseline cổ điển trên cả 2 tác vụ, cải thiện mạnh mẽ nhất ở độ đo `Sentiment Macro-F1` (**+11.53%**) và `Topic Macro-F1` (**+6.62%**), đồng thời vượt qua mốc chuẩn cũ trên GitHub (93.49% / 82.89% và 88.41% / 80.10%).

---

## 5. Phân Tích Lỗi Sai (Short Error Analysis)

Từ kết quả chạy thực nghiệm trực tiếp của `evaluation/evaluate.py`, nhóm đúc kết 3 nhóm lỗi sai điển hình:

1. **Hiểu ngữ cảnh ẩn vs Đếm từ khóa bề mặt (PhoBERT thắng Baseline)**:
   - *Câu phản hồi*: `"nói tiếng anh lưu loát ."` (Nhãn thực tế: `positive | lecturer`).
   - *Baseline*: Dự đoán `positive | others` do câu không chứa các từ khóa hiển ngôn như "thầy", "cô".
   - *PhoBERT*: Nhận biết được hành động "nói tiếng Anh lưu loát" là lời khen ngợi kỹ năng sư phạm của giảng viên, dự đoán chính xác `positive | lecturer`.
   - *Câu mỉa mai/ẩn ý*: `"trong trường macbook thầy số hai thì không có máy nào số một ."` (Nhãn thực tế: `positive | others`). Baseline bị các từ phủ định "không", "máy" đánh lừa thành `negative | facility`, trong khi PhoBERT hiểu được cấu trúc khen hài hước.

2. **Khó khăn ở lớp Trung tính (`neutral`) do mất cân bằng dữ liệu**:
   - *Câu phản hồi*: `"cách mà cô tiếp cận với sinh viên ."` (Nhãn thực tế: `neutral | lecturer`).
   - Cả 2 mô hình đều dự đoán nhầm thành `positive | lecturer`.
   - *Nguyên nhân*: Lớp `neutral` chỉ chiếm **5.27%** (167/3.166 câu test). Ngữ cảnh câu miêu tả trung tính rất mỏng và dễ bị thiên lệch về các từ mang sắc thái khen ngợi chung.

3. **Giao thoa giữa Chương trình đào tạo và Giảng viên**:
   - *Câu phản hồi*: `"cung cấp kiến thức bổ ích ."` (Nhãn thực tế: `positive | lecturer`).
   - Baseline đoán nhầm sang `training_program` vì từ "kiến thức", trong khi PhoBERT xác định chính xác giảng viên là chủ thể truyền đạt kiến thức trong tiết học.

---

## 📁 Cấu Trúc Mã Nguồn

```
.
├── app/
│   ├── app.py                  # Single-Page Bento Grid Dashboard phân tích phản hồi sinh viên
│   └── ui_charts.py            # Giao diện Vibrant AI Gradient CSS & hệ thống biểu đồ Plotly tương tác
├── data/
│   ├── download_data.py        # Tự động tải dataset Parquet từ Hugging Face
│   ├── dataset.py              # Xử lý dữ liệu, tách từ PyVi và DataLoader PyTorch
│   └── uit_vsfc/               # Bộ dữ liệu gốc (train, validation, test.parquet)
├── evaluation/
│   └── evaluate.py             # Đánh giá Accuracy & Macro-F1, trích xuất Error Analysis
├── models/
│   ├── baseline_model.py       # Mô hình Classical Baseline (TF-IDF + Dual LR)
│   └── phobert_model.py        # Mô hình PhoBERT Multi-Task (Mean Pooling + Dual MLP)
├── saved_models/
│   ├── baseline.joblib         # Trọng số Baseline đã huấn luyện
│   └── phobert/                # Trọng số PhoBERT Multi-Task (model.pt trên GitHub Releases v1.0.0 + tokenizer)
├── tests/
│   └── test_core.py            # Kiểm thử tự động (dữ liệu, model, pipeline)
├── training/
│   ├── train_baseline.py       # Huấn luyện mô hình Baseline cổ điển
│   └── train_phobert.py        # Huấn luyện mô hình PhoBERT Multi-Task
├── results/
│   ├── metrics.json            # File JSON chứa toàn bộ chỉ số thực nghiệm
│   └── comparison_summary.md   # Báo cáo tóm tắt so sánh
├── requirements.txt            # Thư viện phụ thuộc
└── (3 file hồ sơ nộp đồ án: 1 slide .pptx + 2 báo cáo 15-20 trang .docx)
```
