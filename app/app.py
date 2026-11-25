"""
app/app.py
Giao diện Web Demo Feedback Analyzer Dashboard cho phản hồi sinh viên UIT-VSFC.
Sử dụng Streamlit thuần túy so sánh song song Classical Baseline vs PhoBERT Multi-Task.
"""

import os
import sys
sys.dont_write_bytecode = True
import time

import streamlit as st
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask

st.set_page_config(
    page_title="Feedback Analyzer Dashboard (UIT-VSFC)",
    page_icon="🎓",
    layout="wide",
)

BASELINE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
PHOBERT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "phobert")

PRESETS = {
    "-- Tự nhập văn bản phản hồi --": "",
    "[Tích cực - Giảng viên] Giảng bài nhiệt tình": "Thầy cô giảng dạy rất nhiệt tình, giải đáp thắc mắc chi tiết và hỗ trợ sinh viên hết mình.",
    "[Tiêu cực - Giảng viên] Dạy lướt nhanh, thiếu tương tác": "Giảng viên vào lớp trễ, giảng bài quá nhanh và không hướng dẫn sinh viên làm bài tập.",
    "[Tích cực - Cơ sở vật chất] Phòng học hiện đại, máy lạnh tốt": "Phòng học khang trang, máy lạnh chạy êm và máy chiếu phục vụ thuyết trình rất sắc nét.",
    "[Tiêu cực - Cơ sở vật chất] Phòng nóng bức, wifi chập chờn": "Phòng học quá nóng bức, máy lạnh bị hỏng và kết nối wifi liên tục bị rớt mạng.",
    "[Tích cực - Chương trình đào tạo] Nội dung thực tế, hữu ích": "Nội dung môn học rất thực tế, bài tập lớn giúp nâng cao kỹ năng nghề nghiệp rõ rệt.",
    "[Tiêu cực - Chương trình đào tạo] Nặng lý thuyết, thi dày đặc": "Lịch thi bố trí quá dày đặc và chương trình học quá nặng về lý thuyết hàn lâm.",
    "[Trung tính - Thông báo trường] Lịch đăng ký tín chỉ": "Nhà trường đã thông báo thời hạn đóng học phí và mở cổng đăng ký môn học học kỳ mới.",
    "[Ngữ cảnh đặc biệt - Mỉa mai] Phòng xông hơi giữa trưa": "Cảm ơn nhà trường đã tạo cơ hội cho sinh viên trải nghiệm phòng học xông hơi giữa trưa nắng.",
}


@st.cache_resource(show_spinner=False)
def load_models():
    """Tải và lưu trữ cả 2 mô hình vào bộ nhớ đệm."""
    b_model = VietnameseFeedbackBaseline.load(BASELINE_PATH) if os.path.isfile(BASELINE_PATH) else None
    p_model = None
    has_weights = os.path.isfile(os.path.join(PHOBERT_DIR, "model.pt")) or os.path.isfile(os.path.join(PHOBERT_DIR, "pytorch_model.bin"))
    has_meta = os.path.isfile(os.path.join(PHOBERT_DIR, "model_meta.json"))
    if os.path.isdir(PHOBERT_DIR) and (has_weights or has_meta):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not has_weights:
                with st.spinner("Đang tải trọng số PhoBERT (516 MB) từ GitHub Releases lần đầu..."):
                    p_model = PhoBERTMultiTask.from_pretrained(PHOBERT_DIR, device=device)
            else:
                p_model = PhoBERTMultiTask.from_pretrained(PHOBERT_DIR, device=device)
        except Exception:
            p_model = None
    return b_model, p_model


def render_prediction_column(title: str, result: dict, latency: float, col):
    """Hiển thị kết quả dự đoán của một mô hình trong một cột Streamlit."""
    with col:
        st.subheader(title)
        st.caption(f"Thời gian suy luận: **{latency*1000:.1f} ms**")

        # 1. Cảm xúc (Sentiment)
        sent = result["sentiment"]
        st.metric(
            label="Cảm xúc (Sentiment)",
            value=sent["label"].upper(),
            delta=f"{sent['confidence']*100:.1f}% độ tin cậy",
        )
        for lbl, prob in sent["distribution"].items():
            st.write(f"- `{lbl}`: {prob*100:.1f}%")
            st.progress(float(min(max(prob, 0.0), 1.0)))

        st.markdown("---")

        # 2. Chủ đề (Topic)
        topic = result["topic"]
        clean_topic = topic["label"].replace("_", " ").upper()
        st.metric(
            label="Chủ đề (Topic)",
            value=clean_topic,
            delta=f"{topic['confidence']*100:.1f}% độ tin cậy",
        )
        for lbl, prob in topic["distribution"].items():
            display_lbl = lbl.replace("_", " ")
            st.write(f"- `{display_lbl}`: {prob*100:.1f}%")
            st.progress(float(min(max(prob, 0.0), 1.0)))


def main():
    st.title("🎓 Student Feedback Analyzer Dashboard")
    st.write(
        "Hệ thống phân loại tự động Cảm xúc (3 lớp) và Chủ đề (4 lớp) trên tập dữ liệu **UIT-VSFC**. "
        "So sánh trực tiếp **Classical Baseline (TF-IDF + LR)** đối đầu **PhoBERT Multi-Task**."
    )

    baseline_model, phobert_model = load_models()

    with st.sidebar:
        st.header("⚙️ Chế độ phân tích")
        mode = st.radio(
            "Chọn chế độ hiển thị:",
            options=[
                "So sánh song song (Baseline vs PhoBERT)",
                "PhoBERT Multi-Task (Mô hình chính)",
                "Classical Baseline (TF-IDF + LR)",
            ],
            index=0,
        )

        st.markdown("---")
        st.subheader("Trạng thái mô hình")
        st.write(f"• **Baseline**: {'✅ Sẵn sàng' if baseline_model else '⚠️ Chưa huấn luyện'}")
        st.write(f"• **PhoBERT**: {'✅ Sẵn sàng' if phobert_model else '⚠️ Chưa huấn luyện'}")
        st.markdown("---")
        st.caption("Đồ án Trí tuệ Nhân tạo Nâng cao | Team 25")

    # Chọn câu mẫu và nhập văn bản
    selected_preset = st.selectbox("Chọn câu phản hồi mẫu để kiểm thử:", options=list(PRESETS.keys()), index=1)
    text_input = st.text_area(
        label="Nội dung phản hồi sinh viên tiếng Việt:",
        value=PRESETS[selected_preset],
        height=110,
        placeholder="Nhập câu phản hồi sinh viên tiếng Việt tại đây...",
    )

    if st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True):
        cleaned = text_input.strip()
        if not cleaned:
            st.warning("Vui lòng nhập văn bản hoặc chọn câu mẫu trước khi phân tích.")
            return

        if "song song" in mode:
            if not baseline_model or not phobert_model:
                st.error("Chưa nạp đủ 2 mô hình. Vui lòng chạy các lệnh huấn luyện trước.")
                return

            t0 = time.time()
            res_b = baseline_model.predict(cleaned, word_segment=True)
            dt_b = time.time() - t0

            t1 = time.time()
            res_p = phobert_model.predict(cleaned, word_segment=True)
            dt_p = time.time() - t1

            c1, c2 = st.columns(2)
            render_prediction_column("📊 Classical Baseline", res_b, dt_b, c1)
            render_prediction_column("🚀 PhoBERT Multi-Task", res_p, dt_p, c2)

        elif "PhoBERT" in mode:
            if not phobert_model:
                st.error("Mô hình PhoBERT chưa sẵn sàng. Vui lòng chạy: python training/train_phobert.py")
                return
            t0 = time.time()
            res_p = phobert_model.predict(cleaned, word_segment=True)
            dt_p = time.time() - t0
            c = st.container()
            render_prediction_column("🚀 PhoBERT Multi-Task (Mô hình chính)", res_p, dt_p, c)

        else:
            if not baseline_model:
                st.error("Mô hình Baseline chưa sẵn sàng. Vui lòng chạy: python training/train_baseline.py")
                return
            t0 = time.time()
            res_b = baseline_model.predict(cleaned, word_segment=True)
            dt_b = time.time() - t0
            c = st.container()
            render_prediction_column("📊 Classical Baseline (TF-IDF + LR)", res_b, dt_b, c)


if __name__ == "__main__":
    main()
