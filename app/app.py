"""
app/app.py
Bảng điều khiển Bento 2 Tab phân tích phản hồi sinh viên UIT-VSFC (Song ngữ 100% Tiếng Việt / Tiếng Anh).
Đối sánh trực tiếp PhoBERT Đa nhiệm và Mô hình Cơ sở (TF-IDF + Hồi quy Logistic).
"""

import os
import sys
sys.dont_write_bytecode = True
import html
import json
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import torch

APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from app.ui_charts import (
        compute_prediction_features,
        extract_token_feature_importance,
        format_class_label,
        inject_vibrant_theme,
        make_confidence_gauge,
        make_confusion_matrix_heatmap,
        make_headline_comparison_chart,
        make_label_distribution_chart,
        make_multimetric_radar_chart,
        make_per_class_f1_chart,
        make_probability_bar_chart,
        make_sentence_multitask_radar_chart,
        make_split_donut_chart,
        make_token_importance_chart,
        render_kpi_grid,
        render_section_header,
        render_segmented_tokens,
        render_top_title_bar,
    )
except (ImportError, ModuleNotFoundError):
    from ui_charts import (
        compute_prediction_features,
        extract_token_feature_importance,
        format_class_label,
        inject_vibrant_theme,
        make_confidence_gauge,
        make_confusion_matrix_heatmap,
        make_headline_comparison_chart,
        make_label_distribution_chart,
        make_multimetric_radar_chart,
        make_per_class_f1_chart,
        make_probability_bar_chart,
        make_sentence_multitask_radar_chart,
        make_split_donut_chart,
        make_token_importance_chart,
        render_kpi_grid,
        render_section_header,
        render_segmented_tokens,
        render_top_title_bar,
    )

from data.dataset import SENTIMENT_LABELS, TOPIC_LABELS, segment_text
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask

st.set_page_config(
    page_title="UIT-VSFC AI Feedback Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASELINE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
PHOBERT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "phobert")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics.json")

SCENARIOS: List[Dict[str, str]] = [
    {
        "id": "01",
        "title_vi": "01 · Giảng viên - Tích cực: Giảng dạy nhiệt tình, giải đáp chi tiết",
        "title_en": "01 · Lecturer - Positive: Enthusiastic teaching & detailed guidance",
        "text": "Thầy cô giảng dạy rất nhiệt tình, giải đáp thắc mắc chi tiết và hỗ trợ sinh viên hết mình.",
        "en_translation": "The lecturer teaches very enthusiastically, answers questions in detail, and supports students wholeheartedly.",
        "expected_sent": "positive",
        "expected_topic": "lecturer",
        "difficulty_vi": "Rõ ràng · Từ khóa cảm xúc trực tiếp",
        "difficulty_en": "Explicit · Direct sentiment lexical cues",
    },
    {
        "id": "02",
        "title_vi": "02 · Giảng viên - Tiêu cực: Vào lớp trễ, giảng bài quá nhanh",
        "title_en": "02 · Lecturer - Negative: Late arrival & rushed lecture pace",
        "text": "Giảng viên vào lớp trễ, giảng bài quá nhanh và không hướng dẫn sinh viên làm bài tập.",
        "en_translation": "The lecturer arrives late to class, lectures too fast, and does not guide students on exercises.",
        "expected_sent": "negative",
        "expected_topic": "lecturer",
        "difficulty_vi": "Rõ ràng · Phê bình phương pháp giảng dạy",
        "difficulty_en": "Explicit · Direct pedagogical criticism",
    },
    {
        "id": "03",
        "title_vi": "03 · Cơ sở vật chất - Tích cực: Phòng học khang trang, thiết bị sắc nét",
        "title_en": "03 · Facility - Positive: Modern classroom & sharp projector",
        "text": "Phòng học khang trang, máy lạnh chạy êm và máy chiếu phục vụ thuyết trình rất sắc nét.",
        "en_translation": "The classroom is spacious, the air conditioner runs smoothly, and the presentation projector is very sharp.",
        "expected_sent": "positive",
        "expected_topic": "facility",
        "difficulty_vi": "Rõ ràng · Đa thực thể trang thiết bị",
        "difficulty_en": "Explicit · Multi-entity classroom equipment",
    },
    {
        "id": "04",
        "title_vi": "04 · Cơ sở vật chất - Tiêu cực: Phòng nóng bức, máy lạnh hỏng, rớt mạng",
        "title_en": "04 · Facility - Negative: Hot room, broken AC & unstable network",
        "text": "Phòng học quá nóng bức, máy lạnh bị hỏng và kết nối wifi liên tục bị rớt mạng.",
        "en_translation": "The classroom is too hot, the air conditioner is broken, and the wireless connection keeps dropping.",
        "expected_sent": "negative",
        "expected_topic": "facility",
        "difficulty_vi": "Rõ ràng · Phàn nàn hạ tầng phòng học",
        "difficulty_en": "Explicit · Infrastructure complaint",
    },
    {
        "id": "05",
        "title_vi": "05 · Chương trình đào tạo - Tích cực: Nội dung thực tế, nâng cao kỹ năng",
        "title_en": "05 · Training Program - Positive: Practical course & career skills",
        "text": "Nội dung môn học rất thực tế, bài tập lớn giúp nâng cao kỹ năng nghề nghiệp rõ rệt.",
        "en_translation": "The course content is very practical, and the major project significantly improves professional skills.",
        "expected_sent": "positive",
        "expected_topic": "training_program",
        "difficulty_vi": "Trung bình · Ngữ cảnh học thuật & kỹ năng",
        "difficulty_en": "Medium · Curriculum & skill development",
    },
    {
        "id": "06",
        "title_vi": "06 · Chương trình đào tạo - Tiêu cực: Lịch thi dày đặc, nặng lý thuyết",
        "title_en": "06 · Training Program - Negative: Dense exam schedule & heavy theory",
        "text": "Lịch thi bố trí quá dày đặc và chương trình học quá nặng về lý thuyết hàn lâm.",
        "en_translation": "The exam schedule is arranged too densely and the curriculum is overly heavy on academic theory.",
        "expected_sent": "negative",
        "expected_topic": "training_program",
        "difficulty_vi": "Trung bình · Đánh giá cấu trúc học phần",
        "difficulty_en": "Medium · Course workload & schedule critique",
    },
    {
        "id": "07",
        "title_vi": "07 · Chủ đề khác - Trung tính: Thông báo học phí & mở cổng đăng ký môn",
        "title_en": "07 · Others - Neutral: Tuition deadline & registration portal notice",
        "text": "Nhà trường đã thông báo thời hạn đóng học phí và mở cổng đăng ký môn học học kỳ mới.",
        "en_translation": "The university announced the tuition payment deadline and opened the course registration portal for the new semester.",
        "expected_sent": "neutral",
        "expected_topic": "others",
        "difficulty_vi": "Lớp hiếm · Câu thông báo hành chính khách quan",
        "difficulty_en": "Minority Class · Objective administrative notice",
    },
    {
        "id": "08",
        "title_vi": "08 · Ngữ cảnh Ẩn ý & Mỉa mai: Trải nghiệm phòng học xông hơi giữa trưa",
        "title_en": "08 · Sarcasm & Implicit Context: 'Sauna classroom' under midday sun",
        "text": "Cảm ơn nhà trường đã tạo cơ hội cho sinh viên trải nghiệm phòng học xông hơi giữa trưa nắng.",
        "en_translation": "Thank the university for giving students the opportunity to experience a sauna classroom under the midday sun.",
        "expected_sent": "negative",
        "expected_topic": "facility",
        "difficulty_vi": "Thách thức cao · Từ bề mặt khen ngợi mang hàm ý chê trách",
        "difficulty_en": "High Difficulty · Positive surface words conveying negative irony",
    },
    {
        "id": "09",
        "title_vi": "09 · Tự nhập phản hồi mới: Nhập văn bản tùy chỉnh để kiểm thử",
        "title_en": "09 · Custom Feedback Input: Enter your own sentence for analysis",
        "text": "",
        "en_translation": "Custom Vietnamese student feedback entered directly by the user.",
        "expected_sent": "",
        "expected_topic": "",
        "difficulty_vi": "Tùy chỉnh · Kiểm thử tự do thời gian thực",
        "difficulty_en": "Custom · Free-form real-time evaluation",
    },
]

DEFAULT_CM_SENT_PHOBERT = [[1361, 21, 27], [27, 92, 48], [50, 18, 1522]]
DEFAULT_CM_SENT_BASELINE = [[1302, 63, 44], [58, 66, 43], [112, 66, 1412]]
DEFAULT_CM_TOPIC_PHOBERT = [
    [2126, 123, 4, 37],
    [87, 458, 4, 23],
    [2, 4, 138, 1],
    [37, 32, 3, 87],
]
DEFAULT_CM_TOPIC_BASELINE = [
    [1943, 238, 5, 104],
    [71, 454, 5, 42],
    [2, 5, 137, 1],
    [33, 41, 2, 83],
]


@st.cache_resource(show_spinner=False)
def load_models():
    """Nạp và lưu đệm mô hình Cơ sở (TF-IDF) và PhoBERT Đa nhiệm."""
    b_model = VietnameseFeedbackBaseline.load(BASELINE_PATH) if os.path.isfile(BASELINE_PATH) else None
    p_model = None
    has_weights = os.path.isfile(os.path.join(PHOBERT_DIR, "model.pt")) or os.path.isfile(
        os.path.join(PHOBERT_DIR, "pytorch_model.bin")
    )
    has_meta = os.path.isfile(os.path.join(PHOBERT_DIR, "model_meta.json"))
    if os.path.isdir(PHOBERT_DIR) and (has_weights or has_meta):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            p_model = PhoBERTMultiTask.from_pretrained(PHOBERT_DIR, device=device)
        except Exception:
            p_model = None
    return b_model, p_model


def load_metrics_data() -> Dict[str, Any]:
    """Đọc dữ liệu thực nghiệm từ results/metrics.json kèm cấu trúc dự phòng an toàn."""
    data: Dict[str, Any] = {}
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.setdefault("num_test_samples", 3166)
    data.setdefault(
        "baseline",
        {
            "sent_acc": 0.8781,
            "sent_wf1": 0.8806,
            "sent_mf1": 0.7276,
            "topic_acc": 0.8266,
            "topic_wf1": 0.8371,
            "topic_mf1": 0.7369,
            "mean_mf1": 0.7322,
            "latency_sec": 0.1325,
            "speed": 23887.95,
        },
    )
    data.setdefault(
        "phobert",
        {
            "sent_acc": 0.9397,
            "sent_wf1": 0.9377,
            "sent_mf1": 0.8429,
            "topic_acc": 0.8872,
            "topic_wf1": 0.8878,
            "topic_mf1": 0.8030,
            "mean_mf1": 0.8230,
            "latency_sec": 187.96,
            "speed": 16.84,
        },
    )
    data.setdefault(
        "dataset_stats",
        {
            "split_sizes": {"train": 11426, "validation": 1583, "test": 3166, "total": 16175},
            "sentiment_distribution": {"negative": 7439, "neutral": 698, "positive": 8038},
            "topic_distribution": {
                "lecturer": 11607,
                "training_program": 3040,
                "facility": 712,
                "others": 816,
            },
        },
    )
    return data


def _badge_color_for_sentiment(label: str) -> str:
    lbl = label.lower()
    if lbl == "positive":
        return "#10B981"
    if lbl == "negative":
        return "#F43F5E"
    return "#F59E0B"


def _badge_color_for_topic(label: str) -> str:
    lbl = label.lower()
    if lbl == "lecturer":
        return "#8B5CF6"
    if lbl == "training_program":
        return "#06B6D4"
    if lbl == "facility":
        return "#F59E0B"
    return "#94A3B8"


def _format_pair_label(raw_pair: str, lang: str) -> str:
    parts = [p.strip() for p in str(raw_pair).split("|") if p.strip()]
    if not parts:
        return ""
    return " · ".join(format_class_label(p, lang) for p in parts)


def render_scenario_metadata_box(scenario: Dict[str, str], lang: str) -> None:
    """Hiển thị hộp thông tin ngữ cảnh kịch bản (nhãn kỳ vọng, độ khó và bản dịch tiếng Anh khi ở chế độ EN)."""
    exp_sent = scenario.get("expected_sent", "")
    exp_topic = scenario.get("expected_topic", "")
    diff_text = scenario.get("difficulty_vi" if lang == "vi" else "difficulty_en", "")

    if exp_sent and exp_topic:
        sent_color = _badge_color_for_sentiment(exp_sent)
        topic_color = _badge_color_for_topic(exp_topic)
        sent_name = format_class_label(exp_sent, lang)
        topic_name = format_class_label(exp_topic, lang)
        sent_prefix = "Cảm xúc kỳ vọng" if lang == "vi" else "Expected Sentiment"
        topic_prefix = "Chủ đề kỳ vọng" if lang == "vi" else "Expected Topic"
        badges_html = (
            f'<span style="background:{sent_color}22; border:1px solid {sent_color}; color:{sent_color}; '
            f'padding:0.22rem 0.65rem; border-radius:999px; font-size:0.76rem; font-weight:700;">'
            f'{sent_prefix}: {html.escape(sent_name)}</span>'
            f'<span style="background:{topic_color}22; border:1px solid {topic_color}; color:{topic_color}; '
            f'padding:0.22rem 0.65rem; border-radius:999px; font-size:0.76rem; font-weight:700;">'
            f'{topic_prefix}: {html.escape(topic_name)}</span>'
        )
    else:
        custom_lbl = "Chế độ nhập tự do" if lang == "vi" else "Custom User Input Mode"
        badges_html = (
            f'<span style="background:rgba(34,211,238,0.16); border:1px solid #22D3EE; color:#22D3EE; '
            f'padding:0.22rem 0.65rem; border-radius:999px; font-size:0.76rem; font-weight:700;">'
            f'{custom_lbl}</span>'
        )

    diff_prefix = "Đặc trưng ngôn ngữ" if lang == "vi" else "Linguistic Profile"
    trans_html = ""
    if lang == "en" and scenario.get("en_translation"):
        trans_html = (
            f'<div style="margin-top:0.55rem; padding-top:0.45rem; border-top:1px solid rgba(148,163,184,0.18); '
            f'font-size:0.8rem; color:#CBD5E1;">'
            f'<b>English Meaning:</b> <i>“{html.escape(scenario["en_translation"])}”</i></div>'
        )

    st.markdown(
        f"""
        <div style="background:rgba(13,21,40,0.82); border:1px solid rgba(34,211,238,0.32);
                    border-radius:12px; padding:0.75rem 0.95rem; margin:0.35rem 0 0.85rem 0;">
            <div style="display:flex; flex-wrap:wrap; gap:0.45rem; align-items:center;">
                {badges_html}
                <span style="background:rgba(139,92,246,0.18); border:1px solid rgba(168,85,247,0.45); color:#C4B5FD;
                             padding:0.22rem 0.65rem; border-radius:999px; font-size:0.75rem; font-weight:600;">
                    {diff_prefix}: {html.escape(diff_text)}
                </span>
            </div>
            {trans_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_summary_card(
    title: str,
    arch_tag: str,
    result: Dict[str, Any],
    latency_sec: float,
    accent_color: str,
    lang: str,
) -> None:
    """Hiển thị thẻ thông số dự đoán, độ trễ, độ bất định Shannon Entropy và biên độ quyết định."""
    feats = compute_prediction_features(result)
    sent_raw = str(result["sentiment"]["label"])
    topic_raw = str(result["topic"]["label"])
    sent_name = format_class_label(sent_raw, lang)
    topic_name = format_class_label(topic_raw, lang)
    sent_color = _badge_color_for_sentiment(sent_raw)
    topic_color = _badge_color_for_topic(topic_raw)
    latency_ms = latency_sec * 1000.0

    lbl_sent = "Cảm xúc" if lang == "vi" else "Sentiment"
    lbl_topic = "Chủ đề" if lang == "vi" else "Topic"
    lbl_latency = "Độ trễ" if lang == "vi" else "Latency"
    lbl_joint = "Tin cậy Kép" if lang == "vi" else "Joint Conf."
    lbl_entropy = "Độ bất định (Cảm xúc / Chủ đề)" if lang == "vi" else "Shannon Entropy (Sent / Topic)"
    lbl_margin = "Biên độ Top-1 vs Top-2" if lang == "vi" else "Top-1 vs Top-2 Margin"
    unit_bits = "bit" if lang == "vi" else "bits"

    st.markdown(
        f"""
        <div class="bento-card" style="border-top:3px solid {accent_color}; padding:0.95rem 1.1rem; margin-bottom:0.75rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;">
                <div>
                    <div style="font-size:1.04rem; font-weight:800; color:#F8FAFC;">{html.escape(title)}</div>
                    <div style="font-size:0.76rem; color:#94A3B8;">{html.escape(arch_tag)}</div>
                </div>
                <span style="background:rgba(15,23,42,0.9); border:1px solid {accent_color}66; color:{accent_color};
                             padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem; font-weight:700;">
                    ⏱ {lbl_latency}: {latency_ms:.1f} ms
                </span>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.65rem 0;">
                <span style="background:{sent_color}22; border:1px solid {sent_color}; color:{sent_color};
                             padding:0.25rem 0.7rem; border-radius:999px; font-size:0.8rem; font-weight:700;">
                    {lbl_sent}: {html.escape(sent_name)} ({feats['sent_conf']:.1f}%)
                </span>
                <span style="background:{topic_color}22; border:1px solid {topic_color}; color:{topic_color};
                             padding:0.25rem 0.7rem; border-radius:999px; font-size:0.8rem; font-weight:700;">
                    {lbl_topic}: {html.escape(topic_name)} ({feats['topic_conf']:.1f}%)
                </span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:0.45rem; font-size:0.75rem;">
                <div style="background:rgba(7,9,19,0.65); padding:0.4rem 0.55rem; border-radius:8px; border:1px solid rgba(148,163,184,0.18);">
                    <div style="color:#94A3B8; font-size:0.68rem;">{lbl_joint}</div>
                    <div style="color:#F8FAFC; font-weight:700; font-family:'JetBrains Mono',monospace;">{feats['joint_conf']:.1f}%</div>
                </div>
                <div style="background:rgba(7,9,19,0.65); padding:0.4rem 0.55rem; border-radius:8px; border:1px solid rgba(148,163,184,0.18);">
                    <div style="color:#94A3B8; font-size:0.68rem;">{lbl_entropy}</div>
                    <div style="color:#22D3EE; font-weight:700; font-family:'JetBrains Mono',monospace;">{feats['sent_entropy']:.2f} / {feats['topic_entropy']:.2f} {unit_bits}</div>
                </div>
                <div style="background:rgba(7,9,19,0.65); padding:0.4rem 0.55rem; border-radius:8px; border:1px solid rgba(148,163,184,0.18);">
                    <div style="color:#94A3B8; font-size:0.68rem;">{lbl_margin}</div>
                    <div style="color:#34D399; font-weight:700; font-family:'JetBrains Mono',monospace;">{feats['sent_margin']:.1f}% / {feats['topic_margin']:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_architecture_comparison_box(lang: str) -> None:
    """Hiển thị bảng Bento đối sánh cấu hình huấn luyện và kiến trúc giữa PhoBERT và Mô hình Cơ sở."""
    if lang == "vi":
        p_title = "🚀 Kiến trúc PhoBERT Đa nhiệm (Mô hình Đề xuất)"
        b_title = "📊 Mô hình Cơ sở (TF-IDF + Hồi quy Logistic Kép)"
        p_rows = [
            ("Mạng nền tảng", "vinai/phobert-base-v2 (135 triệu tham số, gộp từ vị chuẩn BPE)"),
            ("Cơ chế Gộp đặc trưng", "Gộp trung bình có mặt nạ chú ý (Masked Mean Pooling)"),
            ("Nhánh Phân loại Kép", "2 nhánh mạng nơ-ron đa tầng độc lập (Hàm kích hoạt GELU, Dropout 0.2)"),
            ("Chiến lược Tinh chỉnh", "Mở khóa 6 tầng biến đổi trên cùng · Tối ưu hóa AdamW (3e-5 / 4.5e-4)"),
            ("Dung lượng Lưu trữ", "516,56 MB (Tự động tải trọng số từ kho phát hành đám mây)"),
        ]
        b_rows = [
            ("Tiền xử lý Ngôn ngữ", "Chuẩn hóa Unicode NFC & tách từ ghép tiếng Việt bằng thư viện PyVi"),
            ("Trích xuất Đặc trưng", "TF-IDF 25.000 cụm từ (1-2 âm tiết, thang đo tần suất logarit)"),
            ("Bộ Phân loại Tuyến tính", "2 bộ Hồi quy Logistic độc lập cho Cảm xúc (3 lớp) & Chủ đề (4 lớp)"),
            ("Xử lý Mất cân bằng", "Tự động cân bằng trọng số lớp học nghịch đảo tần suất (C = 3,0)"),
            ("Dung lượng & Tốc độ", "1,23 MB · Tốc độ suy luận siêu nhanh (> 23.000 câu/giây trên CPU)"),
        ]
        team_note = (
            "👥 <b>Nhóm Thực hiện 25</b> · Lê Văn Hoài Rin (<code>29211332113</code>) &amp; "
            "Nguyễn Trung Hiếu (<code>29211362521</code>) · Tập dữ liệu chuẩn <b>UIT-VSFC (16.175 câu)</b>"
        )
    else:
        p_title = "🚀 PhoBERT Multi-Task Architecture (Proposed Model)"
        b_title = "📊 Classical Baseline (TF-IDF + Dual Logistic Regression)"
        p_rows = [
            ("Backbone Encoder", "vinai/phobert-base-v2 (135M parameters, Vietnamese BPE subwords)"),
            ("Sequence Pooling", "Masked Mean Pooling across valid attention mask tokens"),
            ("Dual Task Heads", "2 independent 2-layer MLP heads (GELU activation, Dropout 0.2)"),
            ("Fine-Tuning Regimen", "Top 6 Transformer layers unfrozen · AdamW (lr = 3e-5 / 4.5e-4)"),
            ("Checkpoint Size", "516.56 MB (Automated weight provisioning via GitHub Releases)"),
        ]
        b_rows = [
            ("Text Preprocessing", "Unicode NFC normalization & PyVi Vietnamese compound word segmentation"),
            ("Feature Extraction", "TF-IDF Vectorizer with 25,000 n-grams (range 1–2, sublinear TF)"),
            ("Linear Classifiers", "Dual independent Logistic Regression heads (3 Sentiments & 4 Topics)"),
            ("Imbalance Handling", "Inverse class-frequency balanced weighting (Regularization C = 3.0)"),
            ("Footprint & Speed", "1.23 MB · Ultra-fast CPU inference (> 23,000 sentences/sec)"),
        ]
        team_note = (
            "👥 <b>Project Team 25</b> · Le Van Hoai Rin (<code>29211332113</code>) &amp; "
            "Nguyen Trung Hieu (<code>29211362521</code>) · Benchmark Corpus: <b>UIT-VSFC (16,175 sentences)</b>"
        )

    c1, c2 = st.columns(2)
    for col, title, rows, color in [(c1, p_title, p_rows, "#8B5CF6"), (c2, b_title, b_rows, "#06B6D4")]:
        rows_html = "".join(
            f'<div style="display:flex; justify-content:space-between; gap:0.75rem; padding:0.38rem 0; '
            f'border-bottom:1px solid rgba(148,163,184,0.14); font-size:0.81rem;">'
            f'<span style="color:#94A3B8; font-weight:600; min-width:34%;">{html.escape(k)}</span>'
            f'<span style="color:#F8FAFC; text-align:right;">{html.escape(v)}</span></div>'
            for k, v in rows
        )
        with col:
            st.markdown(
                f'<div class="bento-card" style="border-top:3px solid {color};">'
                f'<div style="font-size:1.02rem; font-weight:800; color:#F8FAFC; margin-bottom:0.55rem;">{html.escape(title)}</div>'
                f'{rows_html}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="background:rgba(13,21,40,0.75); border:1px solid rgba(139,92,246,0.28); '
        f'border-radius:12px; padding:0.6rem 1rem; margin-bottom:0.6rem; font-size:0.82rem; color:#CBD5E1; text-align:center;">'
        f'{team_note}</div>',
        unsafe_allow_html=True,
    )


def render_error_analysis_grid(samples: List[Dict[str, Any]], lang: str = "vi") -> None:
    """Hiển thị lưới Bento 2 cột cho các ca kiểm thử điển hình trên tập kiểm thử (100% theo ngôn ngữ đã chọn)."""
    if not samples:
        return
    hdr_text = (
        "🔍 Các Ca Kiểm thử Thực tế Điển hình (PhoBERT Vượt trội & Ngôn ngữ Ẩn ý / Đa nghĩa)"
        if lang == "vi"
        else "🔍 Representative Empirical Test Cases (PhoBERT Advantage & Ambiguous Context)"
    )
    st.markdown(
        f"<div style='margin: 0.75rem 0 0.5rem 0; font-size: 1.02rem; font-weight: 700; color: #E2E8F0;'>{hdr_text}</div>",
        unsafe_allow_html=True,
    )
    lbl_case = "CA KIỂM THỬ" if lang == "vi" else "TEST CASE"
    lbl_gt = "Nhãn Thực tế" if lang == "vi" else "Ground Truth"
    lbl_base = "Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline"
    lbl_pho = "PhoBERT Đa nhiệm" if lang == "vi" else "PhoBERT Multi-Task"

    cols = st.columns(2)
    for idx, item in enumerate(samples):
        text_quote = html.escape(str(item.get("text", "")))
        gt = html.escape(_format_pair_label(str(item.get("ground_truth", "")), lang))
        b_pred = html.escape(_format_pair_label(str(item.get("baseline_pred", "")), lang))
        p_pred = html.escape(_format_pair_label(str(item.get("phobert_pred", "")), lang))
        raw_type = str(item.get("type", ""))
        is_phobert_win = "PhoBERT" in raw_type
        if is_phobert_win:
            sample_type = "PhoBERT Vượt trội Mô hình Cơ sở" if lang == "vi" else "PhoBERT Outperforms Baseline"
            badge_color = "#10B981"
            border_color = "rgba(16, 185, 129, 0.38)"
        else:
            sample_type = "Thử thách Ngữ nghĩa & Đa nghĩa" if lang == "vi" else "Ambiguous & Implicit Context"
            badge_color = "#F59E0B"
            border_color = "rgba(245, 158, 11, 0.35)"

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="error-card bento-card" style="border:1px solid {border_color}; margin-bottom:0.85rem; padding:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.55rem;">
                        <span style="font-size:0.75rem; color:#94A3B8; font-weight:600;">{lbl_case} #{idx + 1}</span>
                        <span style="background:{badge_color}22; border:1px solid {badge_color}; color:{badge_color};
                                     padding:0.18rem 0.6rem; border-radius:999px; font-size:0.73rem; font-weight:700;">
                            {html.escape(sample_type)}
                        </span>
                    </div>
                    <div style="font-size:0.95rem; color:#F8FAFC; font-style:italic; margin-bottom:0.75rem;
                                padding:0.55rem 0.75rem; background:rgba(7,9,19,0.65); border-left:3px solid {badge_color}; border-radius:6px;">
                        “{text_quote}”
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:0.45rem; font-size:0.78rem;">
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(148,163,184,0.2);">
                            <div style="color:#94A3B8; font-size:0.7rem;">{lbl_gt}</div>
                            <div style="color:#E2E8F0; font-weight:700;">{gt}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(6,182,212,0.25);">
                            <div style="color:#22D3EE; font-size:0.7rem;">{lbl_base}</div>
                            <div style="color:#E2E8F0; font-weight:600;">{b_pred}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(139,92,246,0.35);">
                            <div style="color:#A855F7; font-size:0.7rem;">{lbl_pho}</div>
                            <div style="color:#F8FAFC; font-weight:700;">{p_pred}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_vibrant_theme()
    st.session_state.setdefault("lang", "vi")

    # ── THANH TIÊU ĐỀ TỐI GIẢN & CHUYỂN ĐỔI NGÔN NGỮ ─────────────────────────
    hdr_col, lang_col = st.columns([5.2, 1.3], vertical_alignment="center")
    with lang_col:
        b_vi, b_en = st.columns(2, gap="small")
        with b_vi:
            if st.button("🇻🇳 Tiếng Việt", key="btn_lang_vi", use_container_width=True):
                st.session_state["lang"] = "vi"
        with b_en:
            if st.button("🇬🇧 English", key="btn_lang_en", use_container_width=True):
                st.session_state["lang"] = "en"

    lang: str = st.session_state["lang"]
    with hdr_col:
        render_top_title_bar(lang=lang)

    baseline_model, phobert_model = load_models()
    metrics_data = load_metrics_data()

    tab1_label = "⚡ Demo Phân tích Phản hồi" if lang == "vi" else "⚡ Live Feedback Analyzer"
    tab2_label = "📊 Thông số Huấn luyện & Đánh giá Mô hình" if lang == "vi" else "📊 Model & Training Metrics"
    tab_demo, tab_metrics = st.tabs([tab1_label, tab2_label])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: DEMO PHÂN TÍCH PHẢN HỒI TRỰC TIẾP (2 CỘT TRÁI - PHẢI)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_demo:
        left_col, right_col = st.columns([1.05, 1.55], gap="large")

        with left_col:
            card_hdr = "🎯 Kịch bản & Đầu vào Phản hồi" if lang == "vi" else "🎯 Scenario & Feedback Input"
            card_sub = (
                "Chọn kịch bản kiểm thử chuẩn hoặc nhập câu phản hồi sinh viên để phân tích"
                if lang == "vi"
                else "Select a curated benchmark scenario or enter custom student feedback"
            )
            st.markdown(
                f"""
                <div class="bento-card" style="padding:0.85rem 1.1rem; margin-bottom:0.65rem; border-left:4px solid #22D3EE;">
                    <div style="font-size:1.08rem; font-weight:800; color:#F8FAFC;">{card_hdr}</div>
                    <div style="font-size:0.8rem; color:#94A3B8;">{card_sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            scenario_idx = st.selectbox(
                label=(
                    "Danh sách 9 kịch bản phản hồi sinh viên tiêu biểu:"
                    if lang == "vi"
                    else "Select from 9 benchmark student feedback scenarios:"
                ),
                options=list(range(len(SCENARIOS))),
                format_func=lambda i: SCENARIOS[i]["title_vi" if lang == "vi" else "title_en"],
                index=0,
                key=f"scenario_selector_{lang}",
            )
            active_scenario = SCENARIOS[scenario_idx]
            render_scenario_metadata_box(active_scenario, lang=lang)

            mode_options = (
                [
                    "So sánh song song (PhoBERT vs Cơ sở)",
                    "Chỉ PhoBERT Đa nhiệm",
                    "Chỉ Mô hình Cơ sở (TF-IDF)",
                ]
                if lang == "vi"
                else [
                    "Side-by-Side (PhoBERT vs Baseline)",
                    "PhoBERT Multi-Task Only",
                    "Classical Baseline Only",
                ]
            )
            mode_idx = st.radio(
                label="Chế độ suy luận mô hình:" if lang == "vi" else "Model inference mode:",
                options=[0, 1, 2],
                format_func=lambda i: mode_options[i],
                index=0,
                horizontal=True,
                key=f"inference_mode_radio_{lang}",
            )

            text_input = st.text_area(
                label=(
                    "Nội dung phản hồi sinh viên (tự động phân tích ngay khi nhập):"
                    if lang == "vi"
                    else "Vietnamese student feedback sentence (analyzed in real time):"
                ),
                value=active_scenario["text"],
                height=96,
                placeholder=(
                    "Nhập câu phản hồi sinh viên tại đây..."
                    if lang == "vi"
                    else "Enter a Vietnamese student feedback sentence here..."
                ),
            )

            st.button(
                "🚀 Thực hiện Phân tích Đa nhiệm Ngay" if lang == "vi" else "🚀 Run Multi-Task Analysis Now",
                type="primary",
                use_container_width=True,
                key="btn_analyze_now",
            )

            cleaned = text_input.strip()
            segmented = segment_text(cleaned) if cleaned else ""
            if segmented:
                render_segmented_tokens(segmented, lang=lang)

        with right_col:
            if not cleaned:
                empty_msg = (
                    "Vui lòng nhập nội dung phản hồi sinh viên hoặc chọn một kịch bản mẫu ở cột bên trái để xem hệ thống biểu đồ phân tích."
                    if lang == "vi"
                    else "Please enter a student feedback sentence or choose a preset scenario on the left to view detailed diagnostic charts."
                )
                st.info(empty_msg)
            else:
                res_p: Optional[Dict[str, Any]] = None
                res_b: Optional[Dict[str, Any]] = None
                dt_p = 0.0
                dt_b = 0.0

                if mode_idx in (0, 1) and phobert_model is not None:
                    t0 = time.perf_counter()
                    res_p = phobert_model.predict(cleaned, word_segment=True)
                    dt_p = time.perf_counter() - t0
                if mode_idx in (0, 2) and baseline_model is not None:
                    t1 = time.perf_counter()
                    res_b = baseline_model.predict(cleaned, word_segment=True)
                    dt_b = time.perf_counter() - t1

                p_title = "🚀 PhoBERT Đa nhiệm" if lang == "vi" else "🚀 PhoBERT Multi-Task"
                p_arch = (
                    "vinai/phobert-base-v2 · Gộp trung bình có mặt nạ · 2 Nhánh MLP"
                    if lang == "vi"
                    else "vinai/phobert-base-v2 · Masked Mean Pooling · Dual MLP Heads"
                )
                b_title = "📊 Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "📊 Classical Baseline (TF-IDF)"
                b_arch = (
                    "Tách từ PyVi · TF-IDF (1-2 âm tiết) · Hồi quy Logistic Cân bằng"
                    if lang == "vi"
                    else "PyVi Segmentation · TF-IDF (1-2 n-grams) · Balanced Logistic Regression"
                )

                # 1. Thẻ thông số dự đoán, Entropy & Margin
                if mode_idx == 0 and res_p is not None and res_b is not None:
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        render_feature_summary_card(p_title, p_arch, res_p, dt_p, "#8B5CF6", lang)
                    with fc2:
                        render_feature_summary_card(b_title, b_arch, res_b, dt_b, "#06B6D4", lang)
                elif res_p is not None:
                    render_feature_summary_card(p_title, p_arch, res_p, dt_p, "#8B5CF6", lang)
                elif res_b is not None:
                    render_feature_summary_card(b_title, b_arch, res_b, dt_b, "#06B6D4", lang)

                primary_res = res_p if res_p is not None else res_b
                compare_res = res_b if (mode_idx == 0 and res_p is not None and res_b is not None) else None
                primary_name = (
                    ("PhoBERT Đa nhiệm" if lang == "vi" else "PhoBERT Multi-Task")
                    if res_p is not None
                    else ("Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline")
                )
                compare_name = "Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline"

                if primary_res is not None:
                    # 2. Hàng biểu đồ 1: Đồng hồ đo độ tin cậy (Confidence Gauges)
                    if mode_idx == 0 and res_p is not None and res_b is not None:
                        g1, g2, g3, g4 = st.columns(4)
                        with g1:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Cảm xúc (PhoBERT)" if lang == "vi" else "PhoBERT Sentiment",
                                    res_p["sentiment"]["label"],
                                    res_p["sentiment"]["confidence"],
                                    accent_color="#8B5CF6",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_p_sent",
                            )
                        with g2:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Chủ đề (PhoBERT)" if lang == "vi" else "PhoBERT Topic",
                                    res_p["topic"]["label"],
                                    res_p["topic"]["confidence"],
                                    accent_color="#8B5CF6",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_p_topic",
                            )
                        with g3:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Cảm xúc (Cơ sở)" if lang == "vi" else "Baseline Sentiment",
                                    res_b["sentiment"]["label"],
                                    res_b["sentiment"]["confidence"],
                                    accent_color="#06B6D4",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_b_sent",
                            )
                        with g4:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Chủ đề (Cơ sở)" if lang == "vi" else "Baseline Topic",
                                    res_b["topic"]["label"],
                                    res_b["topic"]["confidence"],
                                    accent_color="#06B6D4",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_b_topic",
                            )
                    else:
                        g1, g2 = st.columns(2)
                        with g1:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Độ tin cậy Cảm xúc" if lang == "vi" else "Sentiment Confidence",
                                    primary_res["sentiment"]["label"],
                                    primary_res["sentiment"]["confidence"],
                                    accent_color="#8B5CF6",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_single_sent",
                            )
                        with g2:
                            st.plotly_chart(
                                make_confidence_gauge(
                                    "Độ tin cậy Chủ đề" if lang == "vi" else "Topic Confidence",
                                    primary_res["topic"]["label"],
                                    primary_res["topic"]["confidence"],
                                    accent_color="#06B6D4",
                                    lang=lang,
                                ),
                                use_container_width=True,
                                key="gauge_single_topic",
                            )

                    # 3. Hàng biểu đồ 2: Phân phối xác suất Cảm xúc (3 lớp) & Chủ đề (4 lớp)
                    pb1, pb2 = st.columns(2)
                    with pb1:
                        st.plotly_chart(
                            make_probability_bar_chart(
                                primary_res["sentiment"]["distribution"],
                                (
                                    "Phân phối Xác suất Cảm xúc (3 Lớp)"
                                    if lang == "vi"
                                    else "Sentiment Probability Distribution (3 Classes)"
                                ),
                                accent_color="#8B5CF6",
                                lang=lang,
                                compare_distribution=(
                                    compare_res["sentiment"]["distribution"] if compare_res is not None else None
                                ),
                                primary_name=primary_name,
                                compare_name=compare_name,
                            ),
                            use_container_width=True,
                            key="prob_bar_sent",
                        )
                    with pb2:
                        st.plotly_chart(
                            make_probability_bar_chart(
                                primary_res["topic"]["distribution"],
                                (
                                    "Phân phối Xác suất Chủ đề (4 Lớp)"
                                    if lang == "vi"
                                    else "Topic Probability Distribution (4 Classes)"
                                ),
                                accent_color="#06B6D4",
                                lang=lang,
                                compare_distribution=(
                                    compare_res["topic"]["distribution"] if compare_res is not None else None
                                ),
                                primary_name=primary_name,
                                compare_name=compare_name,
                            ),
                            use_container_width=True,
                            key="prob_bar_topic",
                        )

                    # 4. Hàng biểu đồ 3: Radar Đa nhiệm 7 lớp & Trọng số Từ khóa Đặc trưng
                    sent_lbl = str(primary_res["sentiment"]["label"])
                    topic_lbl = str(primary_res["topic"]["label"])
                    sent_id = SENTIMENT_LABELS.index(sent_lbl) if sent_lbl in SENTIMENT_LABELS else 0
                    topic_id = TOPIC_LABELS.index(topic_lbl) if topic_lbl in TOPIC_LABELS else 0
                    token_weights = extract_token_feature_importance(
                        baseline_model,
                        segmented,
                        sent_label_id=sent_id,
                        topic_label_id=topic_id,
                        top_k=8,
                    )

                    rd1, rd2 = st.columns(2)
                    with rd1:
                        st.plotly_chart(
                            make_sentence_multitask_radar_chart(
                                primary_res,
                                compare_res=compare_res,
                                lang=lang,
                                primary_name=primary_name,
                                compare_name=compare_name,
                            ),
                            use_container_width=True,
                            key="sentence_radar_7d",
                        )
                    with rd2:
                        st.plotly_chart(
                            make_token_importance_chart(token_weights, lang=lang),
                            use_container_width=True,
                            key="sentence_token_importance",
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: THÔNG SỐ HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP UIT-VSFC
    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        render_kpi_grid(metrics_data, lang=lang)

        render_section_header(
            "KIẾN TRÚC & CẤU HÌNH HUẤN LUYỆN" if lang == "vi" else "ARCHITECTURE & TRAINING CONFIGURATION",
            (
                "🧠 Đối sánh Kiến trúc Mô hình & Siêu tham số Huấn luyện"
                if lang == "vi"
                else "🧠 Model Architecture & Hyperparameter Comparison"
            ),
            (
                "So sánh cấu trúc mạng PhoBERT Đa nhiệm và Mô hình Cơ sở trên tập dữ liệu UIT-VSFC"
                if lang == "vi"
                else "Detailed structural comparison between PhoBERT Multi-Task and Classical Baseline on UIT-VSFC"
            ),
        )
        render_architecture_comparison_box(lang=lang)

        render_section_header(
            "ĐÁNH GIÁ THỰC NGHIỆM" if lang == "vi" else "EMPIRICAL BENCHMARK",
            (
                "📊 Đối sánh Hiệu năng Thực nghiệm (3.166 Mẫu Kiểm thử)"
                if lang == "vi"
                else "📊 Empirical Performance Benchmark (Test Set N = 3,166)"
            ),
            (
                "Đánh giá độc lập trên 3.166 câu kiểm thử UIT-VSFC theo Độ chính xác và Điểm F1 Vĩ mô"
                if lang == "vi"
                else "Independent evaluation on 3,166 held-out UIT-VSFC test sentences across Accuracy and Macro-F1"
            ),
        )

        row2a_c1, row2a_c2 = st.columns(2)
        with row2a_c1:
            st.plotly_chart(
                make_headline_comparison_chart(metrics_data, lang=lang),
                use_container_width=True,
                key="chart_headline",
            )
        with row2a_c2:
            st.plotly_chart(
                make_multimetric_radar_chart(metrics_data, lang=lang),
                use_container_width=True,
                key="chart_radar",
            )

        row2b_c1, row2b_c2 = st.columns(2)
        with row2b_c1:
            st.plotly_chart(
                make_per_class_f1_chart(metrics_data, "sentiment", lang=lang),
                use_container_width=True,
                key="chart_f1_sent",
            )
        with row2b_c2:
            st.plotly_chart(
                make_per_class_f1_chart(metrics_data, "topic", lang=lang),
                use_container_width=True,
                key="chart_f1_topic",
            )

        render_section_header(
            "CHẨN ĐOÁN & PHÂN TÍCH LỖI" if lang == "vi" else "DIAGNOSTICS & ERROR ANALYSIS",
            (
                "🎯 Ma trận Nhầm lẫn & Phân tích Ca Kiểm thử Thực tế"
                if lang == "vi"
                else "🎯 Confusion Matrices & Empirical Error Analysis"
            ),
            (
                "Trực quan hóa Ma trận Nhầm lẫn 3×3 (Cảm xúc) & 4×4 (Chủ đề) cùng các câu phản hồi ẩn ý"
                if lang == "vi"
                else "Visualizing 3×3 Sentiment and 4×4 Topic Confusion Matrices alongside implicit/sarcastic test cases"
            ),
        )

        cm_root = metrics_data.get("confusion_matrix", {})
        cm_p_sent = cm_root.get("phobert", {}).get("sentiment", DEFAULT_CM_SENT_PHOBERT)
        cm_b_sent = cm_root.get("baseline", {}).get("sentiment", DEFAULT_CM_SENT_BASELINE)
        cm_p_topic = cm_root.get("phobert", {}).get("topic", DEFAULT_CM_TOPIC_PHOBERT)
        cm_b_topic = cm_root.get("baseline", {}).get("topic", DEFAULT_CM_TOPIC_BASELINE)

        cm_top_c1, cm_top_c2 = st.columns(2)
        with cm_top_c1:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_p_sent,
                    SENTIMENT_LABELS,
                    (
                        "PhoBERT Đa nhiệm — Ma trận Nhầm lẫn Cảm xúc (3×3)"
                        if lang == "vi"
                        else "PhoBERT Multi-Task — Sentiment Confusion Matrix (3×3)"
                    ),
                    colorscale="Purples",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_phobert_sent",
            )
        with cm_top_c2:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_b_sent,
                    SENTIMENT_LABELS,
                    (
                        "Mô hình Cơ sở — Ma trận Nhầm lẫn Cảm xúc (3×3)"
                        if lang == "vi"
                        else "Classical Baseline — Sentiment Confusion Matrix (3×3)"
                    ),
                    colorscale="Blues",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_baseline_sent",
            )

        cm_bot_c1, cm_bot_c2 = st.columns(2)
        with cm_bot_c1:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_p_topic,
                    TOPIC_LABELS,
                    (
                        "PhoBERT Đa nhiệm — Ma trận Nhầm lẫn Chủ đề (4×4)"
                        if lang == "vi"
                        else "PhoBERT Multi-Task — Topic Confusion Matrix (4×4)"
                    ),
                    colorscale="Purples",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_phobert_topic",
            )
        with cm_bot_c2:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_b_topic,
                    TOPIC_LABELS,
                    (
                        "Mô hình Cơ sở — Ma trận Nhầm lẫn Chủ đề (4×4)"
                        if lang == "vi"
                        else "Classical Baseline — Topic Confusion Matrix (4×4)"
                    ),
                    colorscale="Blues",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_baseline_topic",
            )

        render_error_analysis_grid(metrics_data.get("error_analysis_samples", []), lang=lang)

        render_section_header(
            "THỐNG KÊ TẬP DỮ LIỆU" if lang == "vi" else "DATASET EXPLORER",
            (
                "🗂️ Toàn cảnh Tập dữ liệu Học thuật UIT-VSFC (16.175 Câu)"
                if lang == "vi"
                else "🗂️ UIT-VSFC Academic Corpus Overview (16,175 Sentences)"
            ),
            (
                "Phân bố 16.175 câu phản hồi sinh viên qua 3 tập dữ liệu và tỷ lệ phân bố nhãn"
                if lang == "vi"
                else "Distribution across Train / Validation / Test splits and class imbalance ratios"
            ),
        )

        dataset_stats = metrics_data.get("dataset_stats", {})
        ds_c1, ds_c2, ds_c3 = st.columns(3)
        with ds_c1:
            st.plotly_chart(
                make_split_donut_chart(dataset_stats, lang=lang),
                use_container_width=True,
                key="chart_split_donut",
            )
        with ds_c2:
            st.plotly_chart(
                make_label_distribution_chart(dataset_stats, "sentiment", lang=lang),
                use_container_width=True,
                key="chart_dist_sent",
            )
        with ds_c3:
            st.plotly_chart(
                make_label_distribution_chart(dataset_stats, "topic", lang=lang),
                use_container_width=True,
                key="chart_dist_topic",
            )


if __name__ == "__main__":
    main()
