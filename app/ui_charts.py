"""
app/ui_charts.py
Giao diện Vibrant AI Gradient CSS & hệ thống biểu đồ Plotly tương tác cho UIT-VSFC Dashboard.
"""

import html
import sys
sys.dont_write_bytecode = True
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

# Bảng màu chủ đạo Vibrant AI Gradient
BG_DEEP = "#070913"
BG_CARD = "rgba(13, 19, 38, 0.75)"
CYAN = "#06B6D4"
CYAN_BRIGHT = "#22D3EE"
VIOLET = "#8B5CF6"
VIOLET_BRIGHT = "#A855F7"
EMERALD = "#10B981"
ROSE = "#F43F5E"
AMBER = "#F59E0B"
SLATE = "#64748B"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

# Hằng số mặc định thực nghiệm từ UIT-VSFC (fallback an toàn khi thiếu khóa)
DEFAULT_PER_CLASS = {
    "baseline": {
        "sentiment": {
            "negative": {"precision": 0.8550, "recall": 0.9035, "f1": 0.8786, "support": 1409},
            "neutral": {"precision": 0.3413, "recall": 0.4251, "f1": 0.3787, "support": 167},
            "positive": {"precision": 0.9462, "recall": 0.9050, "f1": 0.9254, "support": 1590},
        },
        "topic": {
            "lecturer": {"precision": 0.9277, "recall": 0.8581, "f1": 0.8915, "support": 2290},
            "training_program": {"precision": 0.6147, "recall": 0.7448, "f1": 0.6735, "support": 572},
            "facility": {"precision": 0.8693, "recall": 0.9172, "f1": 0.8926, "support": 145},
            "others": {"precision": 0.4310, "recall": 0.5660, "f1": 0.4898, "support": 159},
        },
    },
    "phobert": {
        "sentiment": {
            "negative": {"precision": 0.9429, "recall": 0.9595, "f1": 0.9511, "support": 1409},
            "neutral": {"precision": 0.6818, "recall": 0.5389, "f1": 0.6020, "support": 167},
            "positive": {"precision": 0.9585, "recall": 0.9642, "f1": 0.9613, "support": 1590},
        },
        "topic": {
            "lecturer": {"precision": 0.9421, "recall": 0.9336, "f1": 0.9378, "support": 2290},
            "training_program": {"precision": 0.7796, "recall": 0.8007, "f1": 0.7900, "support": 572},
            "facility": {"precision": 0.9189, "recall": 0.9379, "f1": 0.9283, "support": 145},
            "others": {"precision": 0.5473, "recall": 0.5660, "f1": 0.5565, "support": 159},
        },
    },
}

DEFAULT_DATASET_STATS = {
    "split_sizes": {"train": 11426, "validation": 1583, "test": 3166, "total": 16175},
    "sentiment_distribution": {"negative": 7439, "neutral": 698, "positive": 8038},
    "topic_distribution": {"lecturer": 11606, "training_program": 2969, "facility": 705, "others": 895},
}

SENTIMENT_VI_NAMES = {
    "negative": "Tiêu cực (Negative)",
    "neutral": "Trung tính (Neutral)",
    "positive": "Tích cực (Positive)",
}

TOPIC_VI_NAMES = {
    "lecturer": "Giảng viên (Lecturer)",
    "training_program": "Chương trình ĐT (Program)",
    "facility": "Cơ sở vật chất (Facility)",
    "others": "Khác (Others)",
}

SEMANTIC_COLORS = {
    "positive": EMERALD,
    "neutral": AMBER,
    "negative": ROSE,
    "lecturer": VIOLET,
    "training_program": CYAN,
    "facility": AMBER,
    "others": SLATE,
}


def inject_vibrant_theme() -> None:
    """Tiêm CSS chủ đề Vibrant AI Gradient với nền tối Deep Space và thẻ Bento Glassmorphism."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(139, 92, 246, 0.16), transparent 40%),
                radial-gradient(circle at 85% 15%, rgba(6, 182, 212, 0.15), transparent 40%),
                #070913 !important;
            color: #F8FAFC !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 9, 19, 0.65) !important;
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: #0B0F1E !important;
            border-right: 1px solid rgba(139, 92, 246, 0.25) !important;
            color: #F8FAFC !important;
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            max-width: 1440px !important;
        }

        /* Form Inputs & Buttons Dark Cohesive Styling */
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: rgba(15, 23, 42, 0.85) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(139, 92, 246, 0.35) !important;
            border-radius: 12px !important;
        }
        .stTextArea textarea:focus {
            border-color: #22D3EE !important;
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.25) !important;
        }
        .stButton > button[kind="primary"], .stButton > button {
            background: linear-gradient(135deg, #8B5CF6 0%, #06B6D4 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.4rem !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35) !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 26px rgba(6, 182, 212, 0.45) !important;
        }

        /* 1. Hero Banner */
        .hero-banner {
            position: relative;
            background: linear-gradient(135deg, rgba(13, 19, 38, 0.88) 0%, rgba(15, 23, 42, 0.78) 100%);
            border: 1px solid rgba(139, 92, 246, 0.38);
            border-radius: 20px;
            padding: 1.65rem 2rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45), 0 0 28px rgba(139, 92, 246, 0.16);
            backdrop-filter: blur(16px);
            overflow: hidden;
        }
        .hero-banner::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, #06B6D4, #8B5CF6, #10B981);
        }
        .hero-top-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.65rem;
        }
        .hero-project-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            background: rgba(139, 92, 246, 0.18);
            color: #C4B5FD;
            border: 1px solid rgba(168, 85, 247, 0.45);
        }
        .hero-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.78rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.14);
            color: #6EE7B7;
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
        }
        .status-pill.offline {
            background: rgba(245, 158, 11, 0.14);
            color: #FCD34D;
            border-color: rgba(245, 158, 11, 0.4);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10B981;
            box-shadow: 0 0 8px #10B981;
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0.25rem 0 0.45rem 0;
            background: linear-gradient(90deg, #F8FAFC 0%, #22D3EE 52%, #A855F7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 0.95rem;
            margin-bottom: 0.85rem;
            line-height: 1.5;
        }
        .hero-team-bar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
            font-size: 0.84rem;
            color: #CBD5E1;
        }
        .team-badge {
            background: rgba(6, 182, 212, 0.15);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.38);
            padding: 0.2rem 0.68rem;
            border-radius: 8px;
            font-weight: 700;
        }

        /* 2. KPI Grid (6 Columns Responsive) */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1.75rem;
        }
        @media (max-width: 1280px) {
            .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 768px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .kpi-card {
            position: relative;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(139, 92, 246, 0.26);
            border-radius: 16px;
            padding: 1.05rem 1.1rem;
            backdrop-filter: blur(12px);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            overflow: hidden;
        }
        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #06B6D4, #8B5CF6);
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            border-color: rgba(34, 211, 238, 0.6);
            box-shadow: 0 12px 28px rgba(6, 182, 212, 0.2), 0 0 18px rgba(139, 92, 246, 0.22);
        }
        .kpi-label {
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.68rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
            margin-bottom: 0.45rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .kpi-delta {
            display: inline-block;
            font-size: 0.74rem;
            font-weight: 700;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.16);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .kpi-delta.corpus {
            background: rgba(6, 182, 212, 0.15);
            color: #22D3EE;
            border-color: rgba(6, 182, 212, 0.35);
        }

        /* 3. Bento Card */
        .bento-card {
            background: rgba(13, 19, 38, 0.74);
            border: 1px solid rgba(139, 92, 246, 0.28);
            border-radius: 16px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.32);
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        }
        .bento-card:hover {
            border-color: rgba(6, 182, 212, 0.52);
            box-shadow: 0 12px 30px rgba(6, 182, 212, 0.15), 0 0 20px rgba(139, 92, 246, 0.16);
        }

        /* 4. Section Header */
        .section-hdr {
            margin: 1.7rem 0 1rem 0;
            padding: 0.95rem 1.25rem;
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.85) 0%, rgba(13, 19, 38, 0.45) 100%);
            border-left: 4px solid #8B5CF6;
            border-radius: 0 14px 14px 0;
            border-top: 1px solid rgba(139, 92, 246, 0.2);
            border-right: 1px solid rgba(6, 182, 212, 0.15);
            border-bottom: 1px solid rgba(6, 182, 212, 0.15);
        }
        .section-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(6, 182, 212, 0.2), rgba(139, 92, 246, 0.25));
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.4);
            margin-bottom: 0.35rem;
        }
        .section-title {
            font-size: 1.32rem;
            font-weight: 800;
            color: #F8FAFC;
            margin: 0.1rem 0 0.2rem 0;
        }
        .section-subtitle {
            font-size: 0.86rem;
            color: #94A3B8;
            margin: 0;
        }

        /* 5. PyVi Segmented Tokens */
        .token-box {
            background: rgba(13, 19, 38, 0.78);
            border: 1px solid rgba(6, 182, 212, 0.32);
            border-radius: 14px;
            padding: 0.9rem 1.15rem;
            margin: 0.75rem 0 1.15rem 0;
        }
        .token-hdr {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.78rem;
            color: #94A3B8;
            margin-bottom: 0.6rem;
            font-weight: 600;
        }
        .token-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            align-items: center;
        }
        .token-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.24rem 0.62rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            background: rgba(30, 41, 59, 0.75);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .token-chip.compound {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.22), rgba(139, 92, 246, 0.26));
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.55);
            font-weight: 600;
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
        }

        /* 6. Prediction Badges */
        .pred-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .pred-badge.positive, .pred-positive {
            background: rgba(16, 185, 129, 0.18);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.5);
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.22);
        }
        .pred-badge.neutral, .pred-neutral {
            background: rgba(245, 158, 11, 0.18);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.5);
            box-shadow: 0 0 14px rgba(245, 158, 11, 0.22);
        }
        .pred-badge.negative, .pred-negative {
            background: rgba(244, 63, 94, 0.18);
            color: #FB7185;
            border: 1px solid rgba(244, 63, 94, 0.5);
            box-shadow: 0 0 14px rgba(244, 63, 94, 0.22);
        }
        .pred-badge.lecturer, .pred-lecturer {
            background: rgba(139, 92, 246, 0.2);
            color: #C4B5FD;
            border: 1px solid rgba(168, 85, 247, 0.5);
            box-shadow: 0 0 14px rgba(139, 92, 246, 0.22);
        }
        .pred-badge.training_program, .pred-training_program {
            background: rgba(6, 182, 212, 0.2);
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.5);
            box-shadow: 0 0 14px rgba(6, 182, 212, 0.22);
        }
        .pred-badge.facility, .pred-facility {
            background: rgba(245, 158, 11, 0.2);
            color: #FCD34D;
            border: 1px solid rgba(245, 158, 11, 0.5);
            box-shadow: 0 0 14px rgba(245, 158, 11, 0.22);
        }
        .pred-badge.others, .pred-others {
            background: rgba(100, 116, 139, 0.22);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.45);
        }

        /* 7. Error Analysis Card */
        .error-card {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(139, 92, 246, 0.28);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .error-card:hover {
            transform: translateY(-2px);
            border-color: rgba(34, 211, 238, 0.5);
        }
        .error-quote {
            font-size: 0.94rem;
            font-weight: 600;
            color: #F8FAFC;
            font-style: italic;
            margin: 0.45rem 0 0.7rem 0;
            padding-left: 0.75rem;
            border-left: 3px solid #22D3EE;
        }
        .error-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            font-size: 0.78rem;
            background: rgba(7, 9, 19, 0.55);
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner(baseline_ready: bool = True, phobert_ready: bool = True) -> None:
    """Hiển thị Hero Banner học thuật cho Project 1 — UIT-VSFC và thông tin Team 25."""
    p_cls = "status-pill" if phobert_ready else "status-pill offline"
    b_cls = "status-pill" if baseline_ready else "status-pill offline"
    p_txt = "PhoBERT Multi-Task (vinai/phobert-base-v2): Sẵn sàng" if phobert_ready else "PhoBERT Multi-Task: Chưa nạp"
    b_txt = "Classical Baseline (TF-IDF + LR): Sẵn sàng" if baseline_ready else "Classical Baseline: Chưa nạp"

    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-top-row">
                <span class="hero-project-tag">🎓 Project 1 · Trí Tuệ Nhân Tạo Nâng Cao · UIT-VSFC Benchmark</span>
                <div class="hero-status-row">
                    <span class="{p_cls}"><span class="status-dot"></span>{p_txt}</span>
                    <span class="{b_cls}"><span class="status-dot"></span>{b_txt}</span>
                </div>
            </div>
            <div class="hero-title">Vietnamese Student Feedback AI Analyzer (UIT-VSFC)</div>
            <div class="hero-subtitle">
                Hệ thống Phân loại Đa nhiệm (Multi-Task Learning) đồng thời <b>Cảm xúc (3 lớp)</b> &amp; <b>Chủ đề (4 lớp)</b>
                trên <b>16.175 câu</b> phản hồi sinh viên Việt Nam — Đối sánh trực tiếp <b>PhoBERT-base-v2 + Masked Mean Pooling</b>
                với <b>Classical Baseline (TF-IDF + Dual Logistic Regression)</b>.
            </div>
            <div class="hero-team-bar">
                <span class="team-badge">TEAM 25</span>
                <span>👨‍💻 <b>Lê Văn Hoài Rin</b> — MSSV: <code>29211332113</code></span>
                <span>·</span>
                <span>👨‍💻 <b>Nguyễn Trung Hiếu</b> — MSSV: <code>29211362521</code></span>
                <span>·</span>
                <span>🏛️ Tập dữ liệu chuẩn: <b>UIT-VSFC (11.426 Train / 1.583 Val / 3.166 Test)</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_grid(metrics: Dict[str, Any]) -> None:
    """Hiển thị lưới 6 thẻ Bento KPI từ số liệu thực nghiệm trong metrics.json."""
    b = metrics.get("baseline", {})
    p = metrics.get("phobert", {})

    p_s_acc = float(p.get("sent_acc", 0.93967)) * 100
    b_s_acc = float(b.get("sent_acc", 0.87808)) * 100
    p_s_mf1 = float(p.get("sent_mf1", 0.84289)) * 100
    b_s_mf1 = float(b.get("sent_mf1", 0.72757)) * 100

    p_t_acc = float(p.get("topic_acc", 0.88724)) * 100
    b_t_acc = float(b.get("topic_acc", 0.82660)) * 100
    p_t_mf1 = float(p.get("topic_mf1", 0.80302)) * 100
    b_t_mf1 = float(b.get("topic_mf1", 0.73686)) * 100

    p_mean = float(p.get("mean_mf1", (p_s_mf1 + p_t_mf1) / 200.0)) * 100
    b_mean = float(b.get("mean_mf1", (b_s_mf1 + b_t_mf1) / 200.0)) * 100

    ds = metrics.get("dataset_stats", DEFAULT_DATASET_STATS).get("split_sizes", DEFAULT_DATASET_STATS["split_sizes"])
    total_cnt = int(ds.get("total", 16175))
    tr_cnt = int(ds.get("train", 11426))
    va_cnt = int(ds.get("validation", 1583))
    te_cnt = int(ds.get("test", 3166))

    cards = [
        ("Sentiment Accuracy", f"{p_s_acc:.2f}%", f"+{p_s_acc - b_s_acc:.2f}% vs Baseline", "kpi-delta"),
        ("Sentiment Macro-F1", f"{p_s_mf1:.2f}%", f"+{p_s_mf1 - b_s_mf1:.2f}% vs Baseline", "kpi-delta"),
        ("Topic Accuracy", f"{p_t_acc:.2f}%", f"+{p_t_acc - b_t_acc:.2f}% vs Baseline", "kpi-delta"),
        ("Topic Macro-F1", f"{p_t_mf1:.2f}%", f"+{p_t_mf1 - b_t_mf1:.2f}% vs Baseline", "kpi-delta"),
        ("Mean Macro-F1", f"{p_mean:.2f}%", f"+{p_mean - b_mean:.2f}% vs Baseline", "kpi-delta"),
        ("UIT-VSFC Corpus", f"{total_cnt:,} câu", f"Train {tr_cnt:,} · Val {va_cnt:,} · Test {te_cnt:,}", "kpi-delta corpus"),
    ]

    cards_html = "".join(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
            <span class="{cls}">{delta}</span>
        </div>
        """
        for lbl, val, delta, cls in cards
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_section_header(badge: str, title: str, subtitle: str) -> None:
    """Hiển thị tiêu đề phân khu Bento có huy hiệu Neon."""
    st.markdown(
        f"""
        <div class="section-hdr">
            <span class="section-badge">{html.escape(badge)}</span>
            <div class="section-title">{html.escape(title)}</div>
            <p class="section-subtitle">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_segmented_tokens(segmented_text: str) -> None:
    """Hiển thị các token đã tách từ bằng PyVi, làm nổi bật từ ghép có dấu gạch dưới '_'."""
    tokens = [tok for tok in segmented_text.strip().split() if tok]
    if not tokens:
        return
    compound_count = sum(1 for tok in tokens if "_" in tok)
    chips_html = "".join(
        f'<span class="token-chip {"compound" if "_" in tok else ""}">{html.escape(tok)}</span>'
        for tok in tokens
    )
    st.markdown(
        f"""
        <div class="token-box">
            <div class="token-hdr">
                <span>🧩 <b>Tiền xử lý Tách từ Tiếng Việt (PyVi Word Segmentation cho PhoBERT BPE)</b></span>
                <span>Tổng: <b>{len(tokens)}</b> token · Từ ghép (<code>_</code>): <b style="color:#22D3EE;">{compound_count}</b></span>
            </div>
            <div class="token-list">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_dark_layout(fig: go.Figure, height: int = 320, title: str = "") -> go.Figure:
    """Áp dụng chuẩn nền tối trong suốt và phông chữ sáng cho mọi biểu đồ Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_LIGHT, family="Inter, sans-serif"), x=0.02, y=0.95) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,38,0.45)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=28, r=24, t=46 if title else 28, b=28),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor=VIOLET, font=dict(color=TEXT_LIGHT, size=12)),
    )
    return fig


def make_confidence_gauge(
    title: str, label: str, confidence: float, accent_color: str = "#8B5CF6"
) -> go.Figure:
    """Biểu đồ đồng hồ đo độ tin cậy dự đoán (0-100%)."""
    pct = confidence * 100.0 if confidence <= 1.0 else float(confidence)
    clean_label = label.replace("_", " ").upper()
    bar_color = SEMANTIC_COLORS.get(label.lower(), accent_color)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            number={"suffix": "%", "font": {"size": 26, "color": TEXT_LIGHT, "family": "JetBrains Mono, monospace"}},
            title={
                "text": f"<span style='font-size:12px;color:#94A3B8'>{title}</span><br><b style='font-size:14px;color:{bar_color}'>{clean_label}</b>",
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"size": 10}},
                "bar": {"color": bar_color, "thickness": 0.72},
                "bgcolor": "rgba(15, 23, 42, 0.7)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.25)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(30, 41, 59, 0.55)"},
                    {"range": [50, 80], "color": "rgba(51, 65, 85, 0.45)"},
                    {"range": [80, 100], "color": "rgba(71, 85, 105, 0.35)"},
                ],
            },
        )
    )
    return _apply_dark_layout(fig, height=195)


def make_probability_bar_chart(
    distribution: Dict[str, float], title: str, accent_color: str = "#06B6D4"
) -> go.Figure:
    """Biểu đồ thanh ngang phân phối xác suất các lớp (0-100%)."""
    labels = list(distribution.keys())
    display_labels = [lbl.replace("_", " ").title() for lbl in labels]
    values = [float(v) * 100.0 if float(v) <= 1.0 else float(v) for v in distribution.values()]
    colors = [SEMANTIC_COLORS.get(lbl.lower(), accent_color) for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=display_labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.22)", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(color=TEXT_LIGHT, size=11, family="JetBrains Mono, monospace"),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=210, title=title)
    fig.update_layout(
        margin=dict(l=18, r=24, t=40, b=20),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(148,163,184,0.14)", ticksuffix="%"),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return fig


def make_headline_comparison_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Biểu đồ cột ghép so sánh 5 chỉ số chính giữa Classical Baseline và PhoBERT Multi-Task."""
    b = metrics.get("baseline", {})
    p = metrics.get("phobert", {})

    categories = ["Sentiment Acc", "Sentiment Macro-F1", "Topic Acc", "Topic Macro-F1", "Mean Macro-F1"]
    keys = ["sent_acc", "sent_mf1", "topic_acc", "topic_mf1", "mean_mf1"]
    b_defaults = [0.8781, 0.7276, 0.8266, 0.7369, 0.7322]
    p_defaults = [0.9397, 0.8429, 0.8872, 0.8030, 0.8230]

    b_vals = [float(b.get(k, d)) * 100.0 for k, d in zip(keys, b_defaults)]
    p_vals = [float(p.get(k, d)) * 100.0 for k, d in zip(keys, p_defaults)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Classical Baseline (TF-IDF + LR)",
            x=categories,
            y=b_vals,
            marker_color=CYAN,
            text=[f"{v:.2f}%" for v in b_vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Baseline: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="PhoBERT Multi-Task (Proposed)",
            x=categories,
            y=p_vals,
            marker_color=VIOLET,
            text=[f"{v:.2f}%" for v in p_vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>PhoBERT: %{y:.2f}%<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=360, title="Đối sánh 5 Chỉ số Hiệu năng Tổng thể (Test Set N = 3,166)")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[60, 102], ticksuffix="%", gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        margin=dict(l=32, r=20, t=62, b=32),
    )
    return fig


def make_multimetric_radar_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Biểu đồ Radar 6 trục đối sánh toàn diện Baseline vs PhoBERT (bao gồm lớp hiếm Neutral & Others)."""
    b = metrics.get("baseline", {})
    p = metrics.get("phobert", {})
    per_class = metrics.get("per_class", DEFAULT_PER_CLASS)

    b_pc = per_class.get("baseline", DEFAULT_PER_CLASS["baseline"])
    p_pc = per_class.get("phobert", DEFAULT_PER_CLASS["phobert"])

    b_neutral_f1 = float(b_pc.get("sentiment", {}).get("neutral", {}).get("f1", 0.3787)) * 100.0
    p_neutral_f1 = float(p_pc.get("sentiment", {}).get("neutral", {}).get("f1", 0.6020)) * 100.0
    b_others_f1 = float(b_pc.get("topic", {}).get("others", {}).get("f1", 0.4898)) * 100.0
    p_others_f1 = float(p_pc.get("topic", {}).get("others", {}).get("f1", 0.5565)) * 100.0

    axes = [
        "Sentiment Acc",
        "Sentiment Macro-F1",
        "Topic Acc",
        "Topic Macro-F1",
        "F1 Lớp hiếm (Neutral)",
        "F1 Chủ đề Khác (Others)",
    ]
    b_r = [
        float(b.get("sent_acc", 0.8781)) * 100.0,
        float(b.get("sent_mf1", 0.7276)) * 100.0,
        float(b.get("topic_acc", 0.8266)) * 100.0,
        float(b.get("topic_mf1", 0.7369)) * 100.0,
        b_neutral_f1,
        b_others_f1,
    ]
    p_r = [
        float(p.get("sent_acc", 0.9397)) * 100.0,
        float(p.get("sent_mf1", 0.8429)) * 100.0,
        float(p.get("topic_acc", 0.8872)) * 100.0,
        float(p.get("topic_mf1", 0.8030)) * 100.0,
        p_neutral_f1,
        p_others_f1,
    ]

    # Khép kín đa giác radar
    axes_closed = axes + [axes[0]]
    b_closed = b_r + [b_r[0]]
    p_closed = p_r + [p_r[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=b_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.20)",
            line=dict(color=CYAN, width=2.2),
            name="Classical Baseline",
            hovertemplate="<b>%{theta}</b>: %{r:.2f}%<extra>Baseline</extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=p_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.28)",
            line=dict(color=VIOLET_BRIGHT, width=2.6),
            name="PhoBERT Multi-Task",
            hovertemplate="<b>%{theta}</b>: %{r:.2f}%<extra>PhoBERT</extra>",
        )
    )
    _apply_dark_layout(fig, height=360, title="Biểu đồ Radar Đa chiều & Năng lực Xử lý Lớp Hiếm")
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(13, 19, 38, 0.55)",
            radialaxis=dict(visible=True, range=[30, 100], gridcolor="rgba(148,163,184,0.18)", ticksuffix="%"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=42, r=42, t=62, b=32),
    )
    return fig


def make_per_class_f1_chart(metrics: Dict[str, Any], task: str = "sentiment") -> go.Figure:
    """Biểu đồ cột ghép so sánh F1-score từng lớp giữa Baseline và PhoBERT cho Sentiment hoặc Topic."""
    per_class = metrics.get("per_class", DEFAULT_PER_CLASS)
    b_task = per_class.get("baseline", DEFAULT_PER_CLASS["baseline"]).get(task, DEFAULT_PER_CLASS["baseline"][task])
    p_task = per_class.get("phobert", DEFAULT_PER_CLASS["phobert"]).get(task, DEFAULT_PER_CLASS["phobert"][task])

    if task == "sentiment":
        keys = ["negative", "neutral", "positive"]
        label_map = SENTIMENT_VI_NAMES
        chart_title = "F1-Score Theo Từng Lớp Cảm Xúc (Sentiment Per-Class F1)"
    else:
        keys = ["lecturer", "training_program", "facility", "others"]
        label_map = TOPIC_VI_NAMES
        chart_title = "F1-Score Theo Từng Lớp Chủ Đề (Topic Per-Class F1)"

    x_labels = [label_map.get(k, k) for k in keys]
    b_f1 = [float(b_task.get(k, DEFAULT_PER_CLASS["baseline"][task][k]).get("f1", 0.0)) * 100.0 for k in keys]
    p_f1 = [float(p_task.get(k, DEFAULT_PER_CLASS["phobert"][task][k]).get("f1", 0.0)) * 100.0 for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Classical Baseline",
            x=x_labels,
            y=b_f1,
            marker_color=CYAN,
            text=[f"{v:.1f}%" for v in b_f1],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Baseline F1: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="PhoBERT Multi-Task",
            x=x_labels,
            y=p_f1,
            marker_color=VIOLET,
            text=[f"{v:.1f}%" for v in p_f1],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>PhoBERT F1: %{y:.2f}%<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=320, title=chart_title)
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="rgba(148,163,184,0.14)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        margin=dict(l=30, r=20, t=58, b=30),
    )
    return fig


def make_confusion_matrix_heatmap(
    cm: List[List[int]], labels: List[str], title: str, colorscale: str = "Purples"
) -> go.Figure:
    """Biểu đồ Heatmap Ma trận Nhầm lẫn có chú thích số lượng mẫu và tỷ lệ % theo hàng."""
    text_matrix = []
    norm_matrix = []
    for row in cm:
        row_sum = max(1, sum(row))
        text_row = []
        norm_row = []
        for val in row:
            pct = (val / row_sum) * 100.0
            text_row.append(f"{val}<br>({pct:.1f}%)")
            norm_row.append(pct)
        text_matrix.append(text_row)
        norm_matrix.append(norm_row)

    clean_labels = [lbl.replace("_", " ").title() for lbl in labels]

    fig = go.Figure(
        go.Heatmap(
            z=norm_matrix,
            x=clean_labels,
            y=clean_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 11, "family": "JetBrains Mono, monospace"},
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="%", ticksuffix="%", thickness=12),
            hovertemplate="Thực tế: <b>%{y}</b><br>Dự đoán: <b>%{x}</b><br>Số mẫu: %{text}<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=330, title=title)
    fig.update_layout(
        xaxis=dict(title="Nhãn Dự đoán (Predicted)", side="bottom"),
        yaxis=dict(title="Nhãn Thực tế (Actual)", autorange="reversed"),
        margin=dict(l=40, r=20, t=48, b=40),
    )
    return fig


def _extract_dataset_stats(dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa đầu vào dataset_stats (hỗ trợ cả truyền trực tiếp hoặc truyền qua dict metrics)."""
    if not dataset_stats:
        return DEFAULT_DATASET_STATS
    if "dataset_stats" in dataset_stats and isinstance(dataset_stats["dataset_stats"], dict):
        return dataset_stats["dataset_stats"]
    return dataset_stats


def make_split_donut_chart(dataset_stats: Dict[str, Any]) -> go.Figure:
    """Biểu đồ Donut tỷ lệ phân chia 3 tập Train / Validation / Test của UIT-VSFC."""
    stats = _extract_dataset_stats(dataset_stats)
    splits = stats.get("split_sizes", DEFAULT_DATASET_STATS["split_sizes"])

    train_n = int(splits.get("train", 11426))
    val_n = int(splits.get("validation", 1583))
    test_n = int(splits.get("test", 3166))
    total_n = int(splits.get("total", train_n + val_n + test_n))

    labels = ["Train Set", "Test Set", "Validation Set"]
    values = [train_n, test_n, val_n]
    colors = [VIOLET, CYAN, EMERALD]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=BG_DEEP, width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color=TEXT_LIGHT),
            hovertemplate="<b>%{label}</b><br>Số câu: %{value:,} (%{percent})<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=320, title="Cấu trúc Phân tập UIT-VSFC (Train / Val / Test)")
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        annotations=[
            dict(
                text=f"<b>{total_n:,}</b><br><span style='font-size:11px;color:#94A3B8'>Câu phản hồi</span>",
                x=0.5,
                y=0.5,
                font=dict(size=16, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
                showarrow=False,
            )
        ],
    )
    return fig


def make_label_distribution_chart(dataset_stats: Dict[str, Any], task: str = "sentiment") -> go.Figure:
    """Biểu đồ phân bố số lượng và tỷ lệ nhãn Cảm xúc (3 lớp) hoặc Chủ đề (4 lớp) trên toàn bộ 16.175 câu."""
    stats = _extract_dataset_stats(dataset_stats)
    if task == "sentiment":
        dist = stats.get("sentiment_distribution", DEFAULT_DATASET_STATS["sentiment_distribution"])
        keys = ["positive", "negative", "neutral"]
        names = ["Positive (Tích cực)", "Negative (Tiêu cực)", "Neutral (Trung tính)"]
        colors = [EMERALD, ROSE, AMBER]
        title = "Phân bố Nhãn Cảm Xúc (Sentiment Imbalance)"
    else:
        dist = stats.get("topic_distribution", DEFAULT_DATASET_STATS["topic_distribution"])
        keys = ["lecturer", "training_program", "others", "facility"]
        names = ["Lecturer (Giảng viên)", "Program (Đào tạo)", "Others (Khác)", "Facility (CSVC)"]
        colors = [VIOLET, CYAN, SLATE, AMBER]
        title = "Phân bố Nhãn Chủ Đề (Topic Imbalance)"

    counts = [int(dist.get(k, 0)) for k in keys]
    total = max(1, sum(counts))
    texts = [f"{c:,} ({c / total * 100:.1f}%)" for c in counts]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=counts,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
            hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,} câu<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=320, title=title)
    fig.update_layout(
        yaxis=dict(range=[0, max(counts) * 1.22], gridcolor="rgba(148,163,184,0.14)", title="Số câu"),
        xaxis=dict(showgrid=False),
        margin=dict(l=36, r=20, t=52, b=36),
    )
    return fig
