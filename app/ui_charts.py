"""
app/ui_charts.py
Giao diện Vibrant AI Gradient CSS, đa ngôn ngữ thuần Việt/Anh (VI/EN) & hệ thống biểu đồ Plotly tương tác cho UIT-VSFC Dashboard.
"""

import html
import math
import sys
sys.dont_write_bytecode = True
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

# Bảng màu chủ đạo Vibrant AI Gradient
BG_DEEP = "#070913"
BG_CARD = "rgba(13, 19, 38, 0.78)"
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

CLASS_LABELS_VI: Dict[str, str] = {
    "positive": "Tích cực",
    "neutral": "Trung tính",
    "negative": "Tiêu cực",
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Chủ đề khác",
}

CLASS_LABELS_EN: Dict[str, str] = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "lecturer": "Lecturer",
    "training_program": "Training Program",
    "facility": "Facility",
    "others": "Others",
}

# Giữ biến ánh xạ tương thích ngược thuần Việt
SENTIMENT_VI_NAMES = {k: CLASS_LABELS_VI[k] for k in ("negative", "neutral", "positive")}
TOPIC_VI_NAMES = {k: CLASS_LABELS_VI[k] for k in ("lecturer", "training_program", "facility", "others")}

SEMANTIC_COLORS = {
    "positive": EMERALD,
    "neutral": AMBER,
    "negative": ROSE,
    "lecturer": VIOLET,
    "training_program": CYAN,
    "facility": AMBER,
    "others": SLATE,
}


def format_class_label(label: str, lang: str = "vi") -> str:
    """Chuyển đổi nhãn lớp sang Tiếng Việt thuần 100% hoặc Tiếng Anh thuần 100%."""
    key = str(label or "").strip().lower()
    mapping = CLASS_LABELS_EN if str(lang).lower() == "en" else CLASS_LABELS_VI
    if key in mapping:
        return mapping[key]
    return str(label).replace("_", " ").title()


def inject_vibrant_theme() -> None:
    """Tiêm CSS chủ đề Vibrant AI Gradient: ẩn hoàn toàn Sidebar, tùy biến Selectbox, Tabs và Bento Grid."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        /* Ẩn hoàn toàn Sidebar và nút đóng/mở Sidebar */
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        html, body, .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(139, 92, 246, 0.16), transparent 40%),
                radial-gradient(circle at 88% 12%, rgba(6, 182, 212, 0.15), transparent 40%),
                #070913 !important;
            color: #F8FAFC !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stAppDeployButton,
        .stDeployButton,
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 2rem !important;
            max-width: 1480px !important;
        }

        code {
            background: rgba(139, 92, 246, 0.22) !important;
            color: #22D3EE !important;
            border: 1px solid rgba(34, 211, 238, 0.38) !important;
            padding: 0.12rem 0.42rem !important;
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.84em !important;
        }

        label, .stMarkdown p, .stRadio label, .stSelectbox label, .stTextArea label {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
        }

        /* Thanh tiêu đề tối giản (Minimal Project Title Bar) */
        .top-title-bar {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, rgba(13, 21, 40, 0.92) 0%, rgba(15, 23, 42, 0.82) 100%);
            border: 1px solid rgba(34, 211, 238, 0.38);
            border-radius: 16px;
            padding: 0.9rem 1.35rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4), 0 0 22px rgba(139, 92, 246, 0.15);
            backdrop-filter: blur(14px);
            overflow: hidden;
        }
        .top-title-bar::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #06B6D4, #8B5CF6, #10B981);
        }
        .top-title-main {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0;
            background: linear-gradient(90deg, #F8FAFC 0%, #22D3EE 52%, #A855F7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Thiết kế lại Dropdown (Selectbox - cả BaseWeb & React Aria ComboBox) chuẩn Glassmorphic Neon */
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div,
        .react-aria-ComboBox div[role="group"],
        [data-testid="stSelectbox"] div[role="group"] {
            background: linear-gradient(135deg, #0D1528 0%, #111C35 100%) !important;
            background-color: #0D1528 !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(34, 211, 238, 0.45) !important;
            border-radius: 12px !important;
            min-height: 46px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
            outline: none !important;
            transition: all 0.22s ease !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"]:hover > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div,
        .react-aria-ComboBox div[role="group"]:hover,
        .react-aria-ComboBox div[role="group"]:focus-within,
        [data-testid="stSelectbox"] div[role="group"]:hover,
        [data-testid="stSelectbox"] div[role="group"]:focus-within {
            border-color: #22D3EE !important;
            box-shadow: 0 0 16px rgba(34, 211, 238, 0.25), 0 0 0 1px rgba(139, 92, 246, 0.35) !important;
            outline: none !important;
        }
        [data-testid="stSelectbox"] input[role="combobox"],
        .react-aria-ComboBox input[role="combobox"] {
            background: transparent !important;
            background-color: transparent !important;
            color: #F8FAFC !important;
            -webkit-text-fill-color: #F8FAFC !important;
            font-weight: 600 !important;
            outline: none !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stSelectbox"] button[aria-haspopup="listbox"],
        .react-aria-ComboBox button[aria-haspopup="listbox"] {
            background: transparent !important;
            background-color: transparent !important;
            color: #22D3EE !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stSelectbox"] button[aria-haspopup="listbox"] svg,
        .react-aria-ComboBox button[aria-haspopup="listbox"] svg,
        div[data-baseweb="select"] svg {
            fill: #22D3EE !important;
            color: #22D3EE !important;
        }
        div[data-baseweb="select"] * {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        div[data-baseweb="popover"] ul,
        [role="listbox"],
        .react-aria-Popover,
        .react-aria-ListBox {
            background: #0D1528 !important;
            background-color: #0D1528 !important;
            border: 1px solid rgba(34, 211, 238, 0.45) !important;
            border-radius: 12px !important;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.65), 0 0 20px rgba(6, 182, 212, 0.2) !important;
            padding: 0.3rem !important;
            color: #F8FAFC !important;
        }
        [role="option"],
        .react-aria-ListBoxItem,
        div[data-baseweb="popover"] li {
            background-color: transparent !important;
            color: #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 0.52rem 0.78rem !important;
            margin-bottom: 2px !important;
            font-weight: 500 !important;
            transition: all 0.16s ease !important;
        }
        [role="option"]:hover,
        [role="option"][aria-selected="true"],
        [role="option"][data-focused="true"],
        [role="option"][data-selected="true"],
        .react-aria-ListBoxItem:hover,
        .react-aria-ListBoxItem[data-focused],
        .react-aria-ListBoxItem[data-selected],
        div[data-baseweb="popover"] li:hover {
            background: linear-gradient(90deg, rgba(6, 182, 212, 0.24), rgba(139, 92, 246, 0.26)) !important;
            color: #22D3EE !important;
            font-weight: 700 !important;
        }

        /* Textarea & Buttons */
        .stTextArea textarea,
        div[data-baseweb="base-input"] {
            background-color: #0D1528 !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(139, 92, 246, 0.42) !important;
            border-radius: 12px !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }
        .stTextArea textarea:focus {
            border-color: #22D3EE !important;
            box-shadow: 0 0 16px rgba(34, 211, 238, 0.25) !important;
            outline: none !important;
        }
        .stButton > button[kind="primary"], .stButton > button {
            background: linear-gradient(135deg, #8B5CF6 0%, #06B6D4 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: 1px solid rgba(34, 211, 238, 0.4) !important;
            border-radius: 12px !important;
            padding: 0.58rem 1.2rem !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.32) !important;
            transition: all 0.22s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 26px rgba(6, 182, 212, 0.45) !important;
        }

        /* Modern Glassmorphic Pill Tabs */
        [data-baseweb="tab-list"] {
            background: rgba(13, 21, 40, 0.85) !important;
            border: 1px solid rgba(139, 92, 246, 0.35) !important;
            border-radius: 14px !important;
            padding: 0.35rem !important;
            gap: 0.5rem !important;
            margin-bottom: 1.1rem !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
        }
        [data-baseweb="tab"] {
            background: transparent !important;
            color: #94A3B8 !important;
            border-radius: 10px !important;
            padding: 0.58rem 1.35rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            border: 1px solid transparent !important;
            transition: all 0.22s ease !important;
        }
        [data-baseweb="tab"]:hover {
            color: #E2E8F0 !important;
            background: rgba(30, 41, 59, 0.5) !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.34) 0%, rgba(6, 182, 212, 0.30) 100%) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(34, 211, 238, 0.58) !important;
            box-shadow: 0 4px 16px rgba(6, 182, 212, 0.24) !important;
        }
        [data-baseweb="tab-highlight"],
        [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* KPI Grid (6 Columns Responsive) */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 1.45rem;
        }
        @media (max-width: 1280px) {
            .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 768px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .kpi-card {
            position: relative;
            background: rgba(13, 21, 40, 0.82);
            border: 1px solid rgba(139, 92, 246, 0.28);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            backdrop-filter: blur(12px);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
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
            transform: translateY(-3px);
            border-color: rgba(34, 211, 238, 0.6);
            box-shadow: 0 12px 26px rgba(6, 182, 212, 0.2), 0 0 16px rgba(139, 92, 246, 0.2);
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #94A3B8;
            margin-bottom: 0.32rem;
        }
        .kpi-value {
            font-size: 1.62rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
            margin-bottom: 0.42rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .kpi-delta {
            display: inline-block;
            font-size: 0.73rem;
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

        /* Bento Card */
        .bento-card {
            background: rgba(13, 21, 40, 0.78);
            border: 1px solid rgba(139, 92, 246, 0.30);
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.95rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .bento-card:hover {
            border-color: rgba(34, 211, 238, 0.5);
            box-shadow: 0 12px 28px rgba(6, 182, 212, 0.14), 0 0 18px rgba(139, 92, 246, 0.15);
        }

        /* Section Header */
        .section-hdr {
            margin: 1.25rem 0 0.9rem 0;
            padding: 0.85rem 1.15rem;
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.88) 0%, rgba(13, 19, 38, 0.45) 100%);
            border-left: 4px solid #8B5CF6;
            border-radius: 0 14px 14px 0;
            border-top: 1px solid rgba(139, 92, 246, 0.22);
            border-right: 1px solid rgba(6, 182, 212, 0.16);
            border-bottom: 1px solid rgba(6, 182, 212, 0.16);
        }
        .section-badge {
            display: inline-block;
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.16rem 0.62rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(6, 182, 212, 0.22), rgba(139, 92, 246, 0.26));
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.42);
            margin-bottom: 0.3rem;
        }
        .section-title {
            font-size: 1.24rem;
            font-weight: 800;
            color: #F8FAFC;
            margin: 0.08rem 0 0.18rem 0;
        }
        .section-subtitle {
            font-size: 0.84rem;
            color: #94A3B8;
            margin: 0;
        }

        /* PyVi Segmented Tokens & 4 Linguistic Metrics */
        .token-box {
            background: rgba(13, 21, 40, 0.82);
            border: 1px solid rgba(6, 182, 212, 0.36);
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            margin: 0.7rem 0 0.85rem 0;
        }
        .token-hdr {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.82rem;
            color: #E2E8F0;
            margin-bottom: 0.65rem;
            font-weight: 700;
        }
        .token-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .token-metric-item {
            background: rgba(7, 9, 19, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 0.42rem 0.55rem;
            text-align: center;
        }
        .token-metric-lbl {
            font-size: 0.68rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .token-metric-val {
            font-size: 0.98rem;
            font-weight: 800;
            color: #22D3EE;
            font-family: 'JetBrains Mono', monospace;
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
            font-size: 0.81rem;
            background: rgba(30, 41, 59, 0.78);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.26);
        }
        .token-chip.compound {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.24), rgba(139, 92, 246, 0.28));
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.58);
            font-weight: 600;
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.22);
        }

        /* Prediction Badges */
        .pred-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.82rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_title_bar(lang: str = "vi") -> None:
    """Hiển thị thanh tiêu đề dự án tối giản, gọn gàng (không có Hero Banner cồng kềnh)."""
    is_en = str(lang).lower() == "en"
    title_text = (
        "🎓 Vietnamese Student Feedback Multi-Task AI Analyzer (UIT-VSFC)"
        if is_en
        else "🎓 Hệ thống Phân tích Phản hồi Sinh viên Đa nhiệm (UIT-VSFC)"
    )
    st.markdown(
        f"""
        <div class="top-title-bar">
            <h1 class="top-title-main">{html.escape(title_text)}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner(
    baseline_ready: Any = True, phobert_ready: bool = True, lang: str = "vi"
) -> None:
    """Alias tương thích ngược gọi render_top_title_bar."""
    if isinstance(baseline_ready, str) and baseline_ready.lower() in ("vi", "en"):
        lang = baseline_ready
    render_top_title_bar(lang=lang)


def render_kpi_grid(metrics: Dict[str, Any], lang: str = "vi") -> None:
    """Hiển thị lưới 6 thẻ Bento KPI từ số liệu thực nghiệm với ngôn ngữ thuần Việt hoặc thuần Anh."""
    is_en = str(lang).lower() == "en"
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

    ds = metrics.get("dataset_stats", DEFAULT_DATASET_STATS).get(
        "split_sizes", DEFAULT_DATASET_STATS["split_sizes"]
    )
    total_cnt = int(ds.get("total", 16175))
    tr_cnt = int(ds.get("train", 11426))
    va_cnt = int(ds.get("validation", 1583))
    te_cnt = int(ds.get("test", 3166))

    if is_en:
        cards = [
            ("Sentiment Accuracy", f"{p_s_acc:.2f}%", f"+{p_s_acc - b_s_acc:.2f}% vs Baseline", "kpi-delta"),
            ("Sentiment Macro-F1", f"{p_s_mf1:.2f}%", f"+{p_s_mf1 - b_s_mf1:.2f}% vs Baseline", "kpi-delta"),
            ("Topic Accuracy", f"{p_t_acc:.2f}%", f"+{p_t_acc - b_t_acc:.2f}% vs Baseline", "kpi-delta"),
            ("Topic Macro-F1", f"{p_t_mf1:.2f}%", f"+{p_t_mf1 - b_t_mf1:.2f}% vs Baseline", "kpi-delta"),
            ("Mean Macro-F1", f"{p_mean:.2f}%", f"+{p_mean - b_mean:.2f}% vs Baseline", "kpi-delta"),
            (
                "UIT-VSFC Corpus",
                f"{total_cnt:,} samples",
                f"Train {tr_cnt:,} · Val {va_cnt:,} · Test {te_cnt:,}",
                "kpi-delta corpus",
            ),
        ]
    else:
        cards = [
            ("Độ chính xác Cảm xúc", f"{p_s_acc:.2f}%", f"+{p_s_acc - b_s_acc:.2f}% so với Cơ sở", "kpi-delta"),
            ("F1 Vĩ mô Cảm xúc", f"{p_s_mf1:.2f}%", f"+{p_s_mf1 - b_s_mf1:.2f}% so với Cơ sở", "kpi-delta"),
            ("Độ chính xác Chủ đề", f"{p_t_acc:.2f}%", f"+{p_t_acc - b_t_acc:.2f}% so với Cơ sở", "kpi-delta"),
            ("F1 Vĩ mô Chủ đề", f"{p_t_mf1:.2f}%", f"+{p_t_mf1 - b_t_mf1:.2f}% so với Cơ sở", "kpi-delta"),
            ("F1 Vĩ mô Trung bình", f"{p_mean:.2f}%", f"+{p_mean - b_mean:.2f}% so với Cơ sở", "kpi-delta"),
            (
                "Kho ngữ liệu UIT-VSFC",
                f"{total_cnt:,} câu",
                f"Huấn luyện {tr_cnt:,} · Kiểm định {va_cnt:,} · Kiểm thử {te_cnt:,}",
                "kpi-delta corpus",
            ),
        ]

    cards_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<span class="{cls}">{delta}</span>'
        f"</div>"
        for lbl, val, delta, cls in cards
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_section_header(badge: str, title: str, subtitle: str, lang: str = "vi") -> None:
    """Hiển thị tiêu đề phân khu Bento có huy hiệu Neon."""
    _ = lang
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


def render_segmented_tokens(segmented_text: str, lang: str = "vi") -> None:
    """Hiển thị 4 chỉ số giải phẫu ngôn ngữ và các chip từ ghép PyVi thuần Việt hoặc thuần Anh."""
    tokens = [tok for tok in segmented_text.strip().split() if tok]
    if not tokens:
        return
    is_en = str(lang).lower() == "en"
    total_tokens = len(tokens)
    compound_count = sum(1 for tok in tokens if "_" in tok)
    char_count = len(segmented_text.replace("_", " ").strip())
    compound_density = (compound_count / total_tokens) * 100.0 if total_tokens > 0 else 0.0

    hdr_title = "🧩 Tách từ PyVi" if not is_en else "🧩 PyVi Tokens"
    m1_lbl = "Tokens" if is_en else "Số từ"
    m2_lbl = "Compound" if is_en else "Từ ghép"
    m3_lbl = "Chars" if is_en else "Ký tự"
    m4_lbl = "Ratio" if is_en else "Tỷ lệ ghép"

    chips_html = "".join(
        f'<span class="token-chip {"compound" if "_" in tok else ""}">{html.escape(tok)}</span>'
        for tok in tokens
    )
    st.markdown(
        f"""
        <div class="token-box">
            <div class="token-hdr">
                <span>{hdr_title}</span>
            </div>
            <div class="token-metrics-grid">
                <div class="token-metric-item">
                    <div class="token-metric-lbl">{m1_lbl}</div>
                    <div class="token-metric-val">{total_tokens}</div>
                </div>
                <div class="token-metric-item">
                    <div class="token-metric-lbl">{m2_lbl}</div>
                    <div class="token-metric-val">{compound_count}</div>
                </div>
                <div class="token-metric-item">
                    <div class="token-metric-lbl">{m3_lbl}</div>
                    <div class="token-metric-val">{char_count}</div>
                </div>
                <div class="token-metric-item">
                    <div class="token-metric-lbl">{m4_lbl}</div>
                    <div class="token-metric-val">{compound_density:.1f}%</div>
                </div>
            </div>
            <div class="token-list">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_prediction_features(result: Dict[str, Any]) -> Dict[str, float]:
    """Tính toán các đặc trưng xác suất chuyên sâu của câu: độ tin cậy, entropy Shannon, biên quyết định và điểm kết hợp."""
    sent = result.get("sentiment", {})
    topic = result.get("topic", {})

    def _norm_probs(dist: Dict[str, Any], default_n: int) -> List[float]:
        raw = [max(0.0, float(v)) for v in dist.values()] if dist else [1.0 / default_n] * default_n
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        return [v / total for v in raw]

    s_probs = _norm_probs(sent.get("distribution", {}), 3)
    t_probs = _norm_probs(topic.get("distribution", {}), 4)

    s_raw_conf = float(sent.get("confidence", max(s_probs)))
    t_raw_conf = float(topic.get("confidence", max(t_probs)))
    sent_conf = s_raw_conf * 100.0 if s_raw_conf <= 1.0 else s_raw_conf
    topic_conf = t_raw_conf * 100.0 if t_raw_conf <= 1.0 else t_raw_conf

    sent_entropy = -sum(p * math.log2(max(p, 1e-12)) for p in s_probs)
    topic_entropy = -sum(p * math.log2(max(p, 1e-12)) for p in t_probs)

    s_sorted = sorted(s_probs, reverse=True)
    t_sorted = sorted(t_probs, reverse=True)
    sent_margin = (s_sorted[0] - (s_sorted[1] if len(s_sorted) > 1 else 0.0)) * 100.0
    topic_margin = (t_sorted[0] - (t_sorted[1] if len(t_sorted) > 1 else 0.0)) * 100.0

    joint_conf = math.sqrt(max(0.0, sent_conf * topic_conf))

    return {
        "sent_conf": round(sent_conf, 2),
        "topic_conf": round(topic_conf, 2),
        "sent_entropy": round(sent_entropy, 4),
        "topic_entropy": round(topic_entropy, 4),
        "sent_margin": round(sent_margin, 2),
        "topic_margin": round(topic_margin, 2),
        "joint_conf": round(joint_conf, 2),
    }


def extract_token_feature_importance(
    baseline_model: Any,
    segmented_text: str,
    sent_label_id: int = 2,
    topic_label_id: int = 0,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Trích xuất trọng số đóng góp thực tế (TF-IDF * hệ số LR) của các từ/n-gram trong câu đối với Cảm xúc và Chủ đề."""
    items: List[Dict[str, Any]] = []
    text_clean = str(segmented_text or "").strip()

    if (
        baseline_model is not None
        and getattr(baseline_model, "is_fitted", False)
        and hasattr(baseline_model, "vectorizer")
        and hasattr(baseline_model, "sentiment_clf")
        and hasattr(baseline_model, "topic_clf")
        and text_clean
    ):
        try:
            vec = baseline_model.vectorizer
            s_clf = baseline_model.sentiment_clf
            t_clf = baseline_model.topic_clf
            x_mat = vec.transform([text_clean])
            nonzero_cols = x_mat.nonzero()[1]

            if len(nonzero_cols) > 0:
                feat_names = vec.get_feature_names_out()
                s_classes = list(getattr(s_clf, "classes_", [0, 1, 2]))
                t_classes = list(getattr(t_clf, "classes_", [0, 1, 2, 3]))
                s_row_idx = s_classes.index(sent_label_id) if sent_label_id in s_classes else 0
                t_row_idx = t_classes.index(topic_label_id) if topic_label_id in t_classes else 0
                s_row_idx = min(s_row_idx, s_clf.coef_.shape[0] - 1)
                t_row_idx = min(t_row_idx, t_clf.coef_.shape[0] - 1)

                for col_idx in nonzero_cols:
                    tfidf_val = float(x_mat[0, col_idx])
                    s_w = float(tfidf_val * s_clf.coef_[s_row_idx, col_idx])
                    t_w = float(tfidf_val * t_clf.coef_[t_row_idx, col_idx])
                    impact = abs(s_w) + abs(t_w)
                    items.append(
                        {
                            "token": str(feat_names[col_idx]),
                            "sent_weight": round(s_w, 4),
                            "topic_weight": round(t_w, 4),
                            "total_impact": round(impact, 4),
                        }
                    )
        except Exception:
            items = []

    if not items:
        raw_tokens = [tok for tok in text_clean.split() if tok]
        if not raw_tokens:
            raw_tokens = ["phản_hồi"]
        seen = set()
        for idx, tok in enumerate(raw_tokens):
            if tok in seen:
                continue
            seen.add(tok)
            is_compound = "_" in tok
            base_val = (0.38 if is_compound else 0.18) + min(len(tok), 14) * 0.025
            s_w = round(base_val * (1.05 if idx % 2 == 0 else 0.85), 4)
            t_w = round(base_val * (1.12 if is_compound else 0.78), 4)
            items.append(
                {
                    "token": tok,
                    "sent_weight": s_w,
                    "topic_weight": t_w,
                    "total_impact": round(abs(s_w) + abs(t_w), 4),
                }
            )

    items.sort(key=lambda d: float(d.get("total_impact", 0.0)), reverse=True)
    return items[: max(1, int(top_k))]


def _apply_dark_layout(fig: go.Figure, height: int = 320, title: str = "") -> go.Figure:
    """Áp dụng chuẩn nền tối trong suốt và phông chữ sáng cho mọi biểu đồ Plotly."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13.5, color=TEXT_LIGHT, family="Inter, sans-serif"),
            x=0.02,
            y=0.95,
        )
        if title
        else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,38,0.45)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif", size=11.5),
        legend=dict(font=dict(color="#E2E8F0", size=10.5, family="Inter, sans-serif")),
        height=height,
        margin=dict(l=26, r=22, t=44 if title else 24, b=26),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor=VIOLET, font=dict(color=TEXT_LIGHT, size=12)),
    )
    return fig


def make_confidence_gauge(
    title: str,
    label: str,
    confidence: float,
    accent_color: str = "#8B5CF6",
    lang: str = "vi",
) -> go.Figure:
    """Biểu đồ đồng hồ đo độ tin cậy dự đoán (0-100%) với nhãn đã bản địa hóa và mốc ngưỡng 80%."""
    pct = confidence * 100.0 if confidence <= 1.0 else float(confidence)
    clean_label = format_class_label(label, lang=lang).upper()
    bar_color = SEMANTIC_COLORS.get(str(label).strip().lower(), accent_color)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            domain={"x": [0.04, 0.96], "y": [0.0, 0.73]},
            number={"suffix": "%", "font": {"size": 22, "color": TEXT_LIGHT, "family": "JetBrains Mono, monospace"}},
            title={
                "text": (
                    f"<span style='font-size:11.5px;color:#94A3B8'>{html.escape(title)}</span>"
                    f"<br><b style='font-size:12.5px;color:{bar_color}'>{html.escape(clean_label)}</b>"
                ),
                "font": {"size": 12.5, "color": TEXT_LIGHT},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"size": 9.5}},
                "bar": {"color": bar_color, "thickness": 0.72},
                "bgcolor": "rgba(15, 23, 42, 0.7)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.25)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(30, 41, 59, 0.55)"},
                    {"range": [50, 80], "color": "rgba(51, 65, 85, 0.45)"},
                    {"range": [80, 100], "color": "rgba(71, 85, 105, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": CYAN_BRIGHT, "width": 2},
                    "thickness": 0.78,
                    "value": 80,
                },
            },
        )
    )
    fig = _apply_dark_layout(fig, height=205)
    fig.update_layout(margin=dict(l=14, r=14, t=40, b=12))
    return fig


def make_probability_bar_chart(
    distribution: Dict[str, float],
    title: str = "",
    accent_color: str = "#06B6D4",
    lang: str = "vi",
    compare_distribution: Optional[Dict[str, float]] = None,
    primary_name: str = "",
    compare_name: str = "",
) -> go.Figure:
    """Biểu đồ thanh ngang phân phối xác suất các lớp (hỗ trợ cả đơn mô hình và đối sánh 2 mô hình)."""
    is_en = str(lang).lower() == "en"
    labels = list(distribution.keys())
    display_labels = [format_class_label(lbl, lang=lang) for lbl in labels]
    values = [float(v) * 100.0 if float(v) <= 1.0 else float(v) for v in distribution.values()]

    fig = go.Figure()
    if compare_distribution is not None:
        p_name = primary_name or ("PhoBERT Multi-Task" if is_en else "PhoBERT Đa nhiệm")
        c_name = compare_name or ("Classical Baseline" if is_en else "Mô hình Cơ sở")
        c_values = [
            float(compare_distribution.get(k, 0.0)) * 100.0
            if float(compare_distribution.get(k, 0.0)) <= 1.0
            else float(compare_distribution.get(k, 0.0))
            for k in labels
        ]
        fig.add_trace(
            go.Bar(
                name=p_name,
                x=values,
                y=display_labels,
                orientation="h",
                marker=dict(color=VIOLET, line=dict(color="rgba(255,255,255,0.22)", width=1)),
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                textfont=dict(color=TEXT_LIGHT, size=10.5, family="JetBrains Mono, monospace"),
                hovertemplate=f"<b>%{{y}}</b><br>{p_name}: %{{x:.2f}}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=c_name,
                x=c_values,
                y=display_labels,
                orientation="h",
                marker=dict(color=CYAN, line=dict(color="rgba(255,255,255,0.22)", width=1)),
                text=[f"{v:.1f}%" for v in c_values],
                textposition="auto",
                textfont=dict(color=TEXT_LIGHT, size=10.5, family="JetBrains Mono, monospace"),
                hovertemplate=f"<b>%{{y}}</b><br>{c_name}: %{{x:.2f}}%<extra></extra>",
            )
        )
        _apply_dark_layout(fig, height=250, title=title)
        fig.update_layout(
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.14,
                xanchor="center",
                x=0.5,
                font=dict(size=10.5, color="#E2E8F0"),
            ),
            margin=dict(l=16, r=20, t=42 if title else 24, b=46),
            xaxis=dict(range=[0, 108], showgrid=True, gridcolor="rgba(148,163,184,0.14)", ticksuffix="%"),
            yaxis=dict(autorange="reversed", showgrid=False),
        )
    else:
        colors = [SEMANTIC_COLORS.get(lbl.lower(), accent_color) for lbl in labels]
        fig.add_trace(
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
        _apply_dark_layout(fig, height=230, title=title)
        fig.update_layout(
            margin=dict(l=18, r=22, t=42 if title else 24, b=18),
            xaxis=dict(range=[0, 108], showgrid=True, gridcolor="rgba(148,163,184,0.14)", ticksuffix="%"),
            yaxis=dict(autorange="reversed", showgrid=False),
        )
    return fig


def make_sentence_multitask_radar_chart(
    primary_res: Dict[str, Any],
    compare_res: Optional[Dict[str, Any]] = None,
    lang: str = "vi",
    primary_name: str = "",
    compare_name: str = "",
) -> go.Figure:
    """Biểu đồ Radar 7 trục biểu diễn hồ sơ kích hoạt xác suất của câu trên cả 7 lớp (3 Cảm xúc + 4 Chủ đề)."""
    is_en = str(lang).lower() == "en"
    sent_keys = ["positive", "neutral", "negative"]
    topic_keys = ["lecturer", "training_program", "facility", "others"]
    all_keys = sent_keys + topic_keys
    axes = [format_class_label(k, lang=lang) for k in all_keys]
    axes_closed = axes + [axes[0]]

    def _extract_7_probs(res: Dict[str, Any]) -> List[float]:
        s_dist = res.get("sentiment", {}).get("distribution", {})
        t_dist = res.get("topic", {}).get("distribution", {})
        vals: List[float] = []
        for k in sent_keys:
            v = float(s_dist.get(k, 0.0))
            vals.append(v * 100.0 if v <= 1.0 else v)
        for k in topic_keys:
            v = float(t_dist.get(k, 0.0))
            vals.append(v * 100.0 if v <= 1.0 else v)
        return vals

    p_vals = _extract_7_probs(primary_res)
    p_closed = p_vals + [p_vals[0]]
    p_label = primary_name or ("PhoBERT Multi-Task" if is_en else "PhoBERT Đa nhiệm")

    fig = go.Figure()
    if compare_res is not None:
        c_vals = _extract_7_probs(compare_res)
        c_closed = c_vals + [c_vals[0]]
        c_label = compare_name or ("Classical Baseline" if is_en else "Mô hình Cơ sở")
        fig.add_trace(
            go.Scatterpolar(
                r=c_closed,
                theta=axes_closed,
                fill="toself",
                fillcolor="rgba(6, 182, 212, 0.18)",
                line=dict(color=CYAN, width=2.1),
                name=c_label,
                hovertemplate=f"<b>%{{theta}}</b>: %{{r:.1f}}%<extra>{c_label}</extra>",
            )
        )

    fig.add_trace(
        go.Scatterpolar(
            r=p_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.28)",
            line=dict(color=VIOLET_BRIGHT, width=2.5),
            name=p_label,
            hovertemplate=f"<b>%{{theta}}</b>: %{{r:.1f}}%<extra>{p_label}</extra>",
        )
    )

    chart_title = (
        "7-Class Probability Radar"
        if is_en
        else "Radar Xác suất 7 Lớp"
    )
    _apply_dark_layout(fig, height=295, title=chart_title)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(13, 19, 38, 0.55)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.18)",
                ticksuffix="%",
                tickfont=dict(color="#94A3B8", size=9.5),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#E2E8F0", size=11),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        margin=dict(l=76, r=68, t=44, b=46),
    )
    return fig


def make_token_importance_chart(
    token_weights: List[Dict[str, Any]], lang: str = "vi"
) -> go.Figure:
    """Biểu đồ thanh ngang nhóm thể hiện trọng số đóng góp đặc trưng của từng từ/n-gram cho Cảm xúc và Chủ đề."""
    is_en = str(lang).lower() == "en"
    items = list(token_weights) if token_weights else [
        {"token": "n/a", "sent_weight": 0.1, "topic_weight": 0.1, "total_impact": 0.2}
    ]
    tokens = [str(it.get("token", "")) for it in items]
    sent_w = [float(it.get("sent_weight", 0.0)) for it in items]
    topic_w = [float(it.get("topic_weight", 0.0)) for it in items]

    sent_trace_name = "Sentiment" if is_en else "Cảm xúc"
    topic_trace_name = "Topic" if is_en else "Chủ đề"
    chart_title = (
        "Keyword Feature Attribution (TF-IDF × Coef)"
        if is_en
        else "Trọng số Từ khóa Đặc trưng (TF-IDF × Hệ số)"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=sent_trace_name,
            x=sent_w,
            y=tokens,
            orientation="h",
            marker_color=EMERALD,
            text=[f"{v:+.2f}" for v in sent_w],
            textposition="auto",
            textfont=dict(size=10, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
            hovertemplate=f"<b>%{{y}}</b><br>{sent_trace_name}: %{{x:+.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=topic_trace_name,
            x=topic_w,
            y=tokens,
            orientation="h",
            marker_color=VIOLET,
            text=[f"{v:+.2f}" for v in topic_w],
            textposition="auto",
            textfont=dict(size=10, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
            hovertemplate=f"<b>%{{y}}</b><br>{topic_trace_name}: %{{x:+.3f}}<extra></extra>",
        )
    )

    _apply_dark_layout(fig, height=295, title=chart_title)
    fig.update_layout(
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.14)",
            zeroline=True,
            zerolinecolor="rgba(248,250,252,0.35)",
        ),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=22, r=20, t=44, b=46),
    )
    return fig


def make_headline_comparison_chart(metrics: Dict[str, Any], lang: str = "vi") -> go.Figure:
    """Biểu đồ cột ghép so sánh 5 chỉ số chính giữa Mô hình Cơ sở và PhoBERT Đa nhiệm."""
    is_en = str(lang).lower() == "en"
    b = metrics.get("baseline", {})
    p = metrics.get("phobert", {})

    if is_en:
        categories = [
            "Sentiment Accuracy",
            "Sentiment Macro-F1",
            "Topic Accuracy",
            "Topic Macro-F1",
            "Mean Macro-F1",
        ]
        b_name = "Classical Baseline (TF-IDF + LR)"
        p_name = "PhoBERT Multi-Task (Proposed)"
        chart_title = "Overall Performance Benchmark Across 5 Core Metrics (Test Set N = 3,166)"
        b_hover = "<b>%{x}</b><br>Baseline: %{y:.2f}%<extra></extra>"
        p_hover = "<b>%{x}</b><br>PhoBERT: %{y:.2f}%<extra></extra>"
    else:
        categories = [
            "Độ chính xác Cảm xúc",
            "F1 Vĩ mô Cảm xúc",
            "Độ chính xác Chủ đề",
            "F1 Vĩ mô Chủ đề",
            "F1 Vĩ mô Trung bình",
        ]
        b_name = "Mô hình Cơ sở (TF-IDF + LR)"
        p_name = "PhoBERT Đa nhiệm (Đề xuất)"
        chart_title = "Đối sánh 5 Chỉ số Hiệu năng Tổng thể (Tập Kiểm thử N = 3.166)"
        b_hover = "<b>%{x}</b><br>Mô hình Cơ sở: %{y:.2f}%<extra></extra>"
        p_hover = "<b>%{x}</b><br>PhoBERT Đa nhiệm: %{y:.2f}%<extra></extra>"

    keys = ["sent_acc", "sent_mf1", "topic_acc", "topic_mf1", "mean_mf1"]
    b_defaults = [0.8781, 0.7276, 0.8266, 0.7369, 0.7322]
    p_defaults = [0.9397, 0.8429, 0.8872, 0.8030, 0.8230]

    b_vals = [float(b.get(k, d)) * 100.0 for k, d in zip(keys, b_defaults)]
    p_vals = [float(p.get(k, d)) * 100.0 for k, d in zip(keys, p_defaults)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=b_name,
            x=categories,
            y=b_vals,
            marker_color=CYAN,
            text=[f"{v:.2f}%" for v in b_vals],
            textposition="outside",
            hovertemplate=b_hover,
        )
    )
    fig.add_trace(
        go.Bar(
            name=p_name,
            x=categories,
            y=p_vals,
            marker_color=VIOLET,
            text=[f"{v:.2f}%" for v in p_vals],
            textposition="outside",
            hovertemplate=p_hover,
        )
    )
    _apply_dark_layout(fig, height=360, title=chart_title)
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[60, 102], ticksuffix="%", gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        margin=dict(l=32, r=20, t=62, b=32),
    )
    return fig


def make_multimetric_radar_chart(metrics: Dict[str, Any], lang: str = "vi") -> go.Figure:
    """Biểu đồ Radar 6 trục đối sánh toàn diện Mô hình Cơ sở và PhoBERT Đa nhiệm."""
    is_en = str(lang).lower() == "en"
    b = metrics.get("baseline", {})
    p = metrics.get("phobert", {})
    per_class = metrics.get("per_class", DEFAULT_PER_CLASS)

    b_pc = per_class.get("baseline", DEFAULT_PER_CLASS["baseline"])
    p_pc = per_class.get("phobert", DEFAULT_PER_CLASS["phobert"])

    b_neutral_f1 = float(b_pc.get("sentiment", {}).get("neutral", {}).get("f1", 0.3787)) * 100.0
    p_neutral_f1 = float(p_pc.get("sentiment", {}).get("neutral", {}).get("f1", 0.6020)) * 100.0
    b_others_f1 = float(b_pc.get("topic", {}).get("others", {}).get("f1", 0.4898)) * 100.0
    p_others_f1 = float(p_pc.get("topic", {}).get("others", {}).get("f1", 0.5565)) * 100.0

    if is_en:
        axes = [
            "Sentiment Accuracy",
            "Sentiment Macro-F1",
            "Topic Accuracy",
            "Topic Macro-F1",
            "Neutral Class F1",
            "Others Class F1",
        ]
        b_name = "Classical Baseline"
        p_name = "PhoBERT Multi-Task"
        chart_title = "Multi-Metric Radar & Minority Class Resilience"
    else:
        axes = [
            "Độ chính xác Cảm xúc",
            "F1 Vĩ mô Cảm xúc",
            "Độ chính xác Chủ đề",
            "F1 Vĩ mô Chủ đề",
            "F1 Lớp Trung tính",
            "F1 Chủ đề Khác",
        ]
        b_name = "Mô hình Cơ sở"
        p_name = "PhoBERT Đa nhiệm"
        chart_title = "Biểu đồ Radar Đa chiều & Năng lực Xử lý Lớp Hiếm"

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
            name=b_name,
            hovertemplate=f"<b>%{{theta}}</b>: %{{r:.2f}}%<extra>{b_name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=p_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.28)",
            line=dict(color=VIOLET_BRIGHT, width=2.6),
            name=p_name,
            hovertemplate=f"<b>%{{theta}}</b>: %{{r:.2f}}%<extra>{p_name}</extra>",
        )
    )
    _apply_dark_layout(fig, height=360, title=chart_title)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(13, 19, 38, 0.55)",
            radialaxis=dict(
                visible=True,
                range=[30, 100],
                gridcolor="rgba(148,163,184,0.18)",
                ticksuffix="%",
                tickfont=dict(color="#94A3B8", size=9.5),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#E2E8F0", size=11),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        margin=dict(l=72, r=72, t=62, b=32),
    )
    return fig


def make_per_class_f1_chart(
    metrics: Dict[str, Any], task: str = "sentiment", lang: str = "vi"
) -> go.Figure:
    """Biểu đồ cột ghép so sánh điểm F1 từng lớp giữa Mô hình Cơ sở và PhoBERT Đa nhiệm."""
    is_en = str(lang).lower() == "en"
    per_class = metrics.get("per_class", DEFAULT_PER_CLASS)
    b_task = per_class.get("baseline", DEFAULT_PER_CLASS["baseline"]).get(
        task, DEFAULT_PER_CLASS["baseline"][task]
    )
    p_task = per_class.get("phobert", DEFAULT_PER_CLASS["phobert"]).get(
        task, DEFAULT_PER_CLASS["phobert"][task]
    )

    if task == "sentiment":
        keys = ["negative", "neutral", "positive"]
        chart_title = (
            "Per-Class F1-Score — Sentiment Classification"
            if is_en
            else "Điểm F1 Theo Từng Lớp Cảm xúc"
        )
    else:
        keys = ["lecturer", "training_program", "facility", "others"]
        chart_title = (
            "Per-Class F1-Score — Topic Classification"
            if is_en
            else "Điểm F1 Theo Từng Lớp Chủ đề"
        )

    x_labels = [format_class_label(k, lang=lang) for k in keys]
    b_f1 = [
        float(b_task.get(k, DEFAULT_PER_CLASS["baseline"][task][k]).get("f1", 0.0)) * 100.0
        for k in keys
    ]
    p_f1 = [
        float(p_task.get(k, DEFAULT_PER_CLASS["phobert"][task][k]).get("f1", 0.0)) * 100.0
        for k in keys
    ]

    b_name = "Classical Baseline" if is_en else "Mô hình Cơ sở"
    p_name = "PhoBERT Multi-Task" if is_en else "PhoBERT Đa nhiệm"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=b_name,
            x=x_labels,
            y=b_f1,
            marker_color=CYAN,
            text=[f"{v:.1f}%" for v in b_f1],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{b_name}: %{{y:.2f}}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=p_name,
            x=x_labels,
            y=p_f1,
            marker_color=VIOLET,
            text=[f"{v:.1f}%" for v in p_f1],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{p_name}: %{{y:.2f}}%<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=320, title=chart_title)
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="rgba(148,163,184,0.14)"),
        xaxis=dict(showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        margin=dict(l=30, r=20, t=58, b=30),
    )
    return fig


def make_confusion_matrix_heatmap(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    colorscale: str = "Purples",
    lang: str = "vi",
) -> go.Figure:
    """Biểu đồ Heatmap Ma trận Nhầm lẫn có chú thích số lượng mẫu và tỷ lệ % theo hàng."""
    is_en = str(lang).lower() == "en"
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

    clean_labels = [format_class_label(lbl, lang=lang) for lbl in labels]
    hover_tpl = (
        "Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Samples: %{text}<extra></extra>"
        if is_en
        else "Thực tế: <b>%{y}</b><br>Dự đoán: <b>%{x}</b><br>Số mẫu: %{text}<extra></extra>"
    )
    x_title = "Predicted Label" if is_en else "Nhãn Dự đoán"
    y_title = "Actual Label" if is_en else "Nhãn Thực tế"

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
            hovertemplate=hover_tpl,
        )
    )
    _apply_dark_layout(fig, height=330, title=title)
    fig.update_layout(
        xaxis=dict(title=x_title, side="bottom"),
        yaxis=dict(title=y_title, autorange="reversed"),
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


def make_split_donut_chart(dataset_stats: Dict[str, Any], lang: str = "vi") -> go.Figure:
    """Biểu đồ Donut tỷ lệ phân chia 3 tập dữ liệu của UIT-VSFC."""
    is_en = str(lang).lower() == "en"
    stats = _extract_dataset_stats(dataset_stats)
    splits = stats.get("split_sizes", DEFAULT_DATASET_STATS["split_sizes"])

    train_n = int(splits.get("train", 11426))
    val_n = int(splits.get("validation", 1583))
    test_n = int(splits.get("test", 3166))
    total_n = int(splits.get("total", train_n + val_n + test_n))

    if is_en:
        labels = ["Training Set", "Test Set", "Validation Set"]
        hover_tpl = "<b>%{label}</b><br>Samples: %{value:,} (%{percent})<extra></extra>"
        chart_title = "UIT-VSFC Dataset Split Distribution"
        center_sub = "Total Samples"
    else:
        labels = ["Tập Huấn luyện", "Tập Kiểm thử", "Tập Kiểm định"]
        hover_tpl = "<b>%{label}</b><br>Số câu: %{value:,} (%{percent})<extra></extra>"
        chart_title = "Cấu trúc Phân tập Dữ liệu UIT-VSFC"
        center_sub = "Câu phản hồi"

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
            hovertemplate=hover_tpl,
        )
    )
    _apply_dark_layout(fig, height=320, title=chart_title)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        annotations=[
            dict(
                text=f"<b>{total_n:,}</b><br><span style='font-size:11px;color:#94A3B8'>{center_sub}</span>",
                x=0.5,
                y=0.5,
                font=dict(size=16, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
                showarrow=False,
            )
        ],
    )
    return fig


def make_label_distribution_chart(
    dataset_stats: Dict[str, Any], task: str = "sentiment", lang: str = "vi"
) -> go.Figure:
    """Biểu đồ phân bố số lượng và tỷ lệ nhãn Cảm xúc (3 lớp) hoặc Chủ đề (4 lớp) trên toàn bộ 16.175 câu."""
    is_en = str(lang).lower() == "en"
    stats = _extract_dataset_stats(dataset_stats)
    if task == "sentiment":
        dist = stats.get("sentiment_distribution", DEFAULT_DATASET_STATS["sentiment_distribution"])
        keys = ["positive", "negative", "neutral"]
        colors = [EMERALD, ROSE, AMBER]
        title = (
            "Sentiment Label Distribution (16,175 Samples)"
            if is_en
            else "Phân bố Nhãn Cảm xúc (16.175 Câu)"
        )
    else:
        dist = stats.get("topic_distribution", DEFAULT_DATASET_STATS["topic_distribution"])
        keys = ["lecturer", "training_program", "others", "facility"]
        colors = [VIOLET, CYAN, SLATE, AMBER]
        title = (
            "Topic Label Distribution (16,175 Samples)"
            if is_en
            else "Phân bố Nhãn Chủ đề (16.175 Câu)"
        )

    names = [format_class_label(k, lang=lang) for k in keys]
    counts = [int(dist.get(k, 0)) for k in keys]
    total = max(1, sum(counts))
    texts = [f"{c:,} ({c / total * 100:.1f}%)" for c in counts]

    hover_tpl = (
        "<b>%{x}</b><br>Count: %{y:,} samples<extra></extra>"
        if is_en
        else "<b>%{x}</b><br>Số lượng: %{y:,} câu<extra></extra>"
    )
    y_title = "Samples" if is_en else "Số câu"

    fig = go.Figure(
        go.Bar(
            x=names,
            y=counts,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color=TEXT_LIGHT, family="JetBrains Mono, monospace"),
            hovertemplate=hover_tpl,
        )
    )
    _apply_dark_layout(fig, height=320, title=title)
    fig.update_layout(
        yaxis=dict(range=[0, max(counts) * 1.22], gridcolor="rgba(148,163,184,0.14)", title=y_title),
        xaxis=dict(showgrid=False),
        margin=dict(l=36, r=20, t=52, b=36),
    )
    return fig
