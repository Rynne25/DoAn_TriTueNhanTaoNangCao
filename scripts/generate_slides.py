# -*- coding: utf-8 -*-
"""
scripts/generate_slides.py
Generates an executive, anti-slop, modern 5-slide presentation for the UIT-VSFC Multi-Task project.
Follows high-end design standards: Bento grid, big data metrics, elegant typography, and smooth Morph transitions.
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Sophisticated Dark / Carbon Palette (Taste Skill inspired)
BG_CANVAS    = RGBColor(10, 14, 23)      # #0A0E17 Deep Obsidian Canvas
CARD_BG      = RGBColor(17, 24, 39)      # #111827 Elevated Card Surface
CARD_HOVER   = RGBColor(26, 36, 56)      # #1A2438 Highlight Surface
BORDER_SUBTLE= RGBColor(38, 51, 77)      # #26334D Clean Hairline Border
BORDER_ACTIVE= RGBColor(99, 102, 241)    # #6366F1 Accent border
WHITE        = RGBColor(255, 255, 255)   # #FFFFFF Primary Ink
TEXT_MUTED   = RGBColor(148, 163, 184)   # #94A3B8 Secondary Text
TEXT_DIM     = RGBColor(100, 116, 139)   # #64748B Subtle Notes
ACCENT_BLUE  = RGBColor(59, 130, 246)    # #3B82F6 Core Blue
ACCENT_INDIGO= RGBColor(99, 102, 241)    # #6366F1 Indigo Accent
ACCENT_EMERALD=RGBColor(16, 185, 129)    # #10B981 Emerald Metric
ACCENT_AMBER = RGBColor(245, 158, 11)    # #F59E0B Warning / Neutral
ACCENT_CYAN  = RGBColor(6, 182, 212)     # #06B6D4 Cyan Tech
CHIP_BG      = RGBColor(30, 41, 59)      # #1E293B Pill Tag Background

def add_bg(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_CANVAS
    bg.line.fill.background()
    return bg

def add_panel(slide, left, top, width, height, bg=CARD_BG, border=BORDER_SUBTLE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_header(slide, title, category="UIT-VSFC · ADVANCED ARTIFICIAL INTELLIGENCE"):
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf = tag_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = category.upper()
    p.font.name = "Segoe UI"
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.6))
    tf2 = title_box.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = WHITE

# ==============================================================================
# SLIDE 1: Executive Dark Theme Cover
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
add_bg(s1)

# Subtle decorative background grid card
add_panel(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), bg=CARD_BG, border=BORDER_SUBTLE)

# Top Bar / Monogram & Institution
top_box = s1.shapes.add_textbox(Inches(1.3), Inches(1.2), Inches(10.7), Inches(0.6))
tf = top_box.text_frame
p = tf.paragraphs[0]
p.text = "TRƯỜNG ĐẠI HỌC DUY TÂN  •  TRƯỜNG KHOA HỌC MÁY TÍNH  •  BỘ MÔN AI"
p.font.name = "Segoe UI"
p.font.size = Pt(9.5)
p.font.bold = True
p.font.color.rgb = ACCENT_CYAN

# Main Title Section
title_box = s1.shapes.add_textbox(Inches(1.3), Inches(1.85), Inches(10.7), Inches(2.2))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "PHÂN LOẠI ĐA NHIỆM PHẢN HỒI SINH VIÊN"
p.font.name = "Segoe UI"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = WHITE
p.space_after = Pt(8)

p2 = tf.add_paragraph()
p2.text = "Kiến Trúc PhoBERT Multi-Task v2 (Shared CLS) & Classical Baseline trên Bộ Dữ Liệu UIT-VSFC"
p2.font.name = "Segoe UI"
p2.font.size = Pt(14)
p2.font.color.rgb = TEXT_MUTED

# 3 Executive Value Chips
chips = [
    ("⚡ 0.05 ms Baseline", "TF-IDF + Dual Logistic Regression"),
    ("🧠 81.50% Mean F1", "PhoBERT-v2 Shared Representation"),
    ("🌐 Song Ngữ VI - EN", "Giao diện Streamlit Taste-Skill")
]
for idx, (head, sub) in enumerate(chips):
    left_c = Inches(1.3 + idx * 3.65)
    add_panel(s1, left_c, Inches(3.8), Inches(3.4), Inches(0.9), bg=CARD_HOVER, border=BORDER_SUBTLE)
    c_box = s1.shapes.add_textbox(left_c + Inches(0.15), Inches(3.88), Inches(3.1), Inches(0.75))
    ctf = c_box.text_frame
    cp1 = ctf.paragraphs[0]
    cp1.text = head
    cp1.font.name = "Segoe UI"
    cp1.font.size = Pt(11)
    cp1.font.bold = True
    cp1.font.color.rgb = WHITE
    cp2 = ctf.add_paragraph()
    cp2.text = sub
    cp2.font.name = "Segoe UI"
    cp2.font.size = Pt(8.5)
    cp2.font.color.rgb = TEXT_MUTED

# Authors & Instructor Banner
add_panel(s1, Inches(1.3), Inches(5.0), Inches(10.733), Inches(1.3), bg=RGBColor(15, 23, 42), border=BORDER_SUBTLE)
foot_box = s1.shapes.add_textbox(Inches(1.5), Inches(5.12), Inches(10.3), Inches(1.05))
ftf = foot_box.text_frame
fp1 = ftf.paragraphs[0]
fp1.text = "GIẢNG VIÊN HƯỚNG DẪN:  TS. VIKRAM PURI     |     LỚP:  CS 467 C"
fp1.font.name = "Segoe UI"
fp1.font.size = Pt(10)
fp1.font.bold = True
fp1.font.color.rgb = ACCENT_INDIGO
fp1.space_after = Pt(4)

fp2 = ftf.add_paragraph()
fp2.text = "Sinh viên thực hiện:  Nguyễn Trung Hiếu (29211362521)   •   Lê Văn Hoài Rin (29211332113)"
fp2.font.name = "Segoe UI"
fp2.font.size = Pt(11.5)
fp2.font.bold = True
fp2.font.color.rgb = WHITE

# ==============================================================================
# SLIDE 2: Bento Grid — Đặt Vấn Đề & Dữ Liệu UIT-VSFC
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
add_bg(s2)
add_header(s2, "1. Bối Cảnh Bài Toán & Tập Dữ Liệu UIT-VSFC")

# Top Stat Row (3 Large Metric Cards)
stats = [
    ("16.175", "TỔNG CÂU PHẢN HỒI", "Train: 11.426 | Val: 1.583 | Test: 3.166", ACCENT_CYAN),
    ("4.01%", "LỚP TRUNG TÍNH (NEUTRAL)", "Mất cân bằng cực đoan (Lớp thiểu số)", ACCENT_AMBER),
    ("2 TASKS", "HỌC ĐA NHIỆM ĐỒNG THỜI", "3 Cảm xúc (Sentiment) × 4 Chủ đề (Topic)", ACCENT_EMERALD)
]
for i, (metric, label, desc, color) in enumerate(stats):
    left_pos = Inches(0.8 + i * 4.0)
    add_panel(s2, left_pos, Inches(1.4), Inches(3.733), Inches(1.4), bg=CARD_BG, border=BORDER_SUBTLE)
    box = s2.shapes.add_textbox(left_pos + Inches(0.2), Inches(1.45), Inches(3.33), Inches(1.25))
    btf = box.text_frame
    p1 = btf.paragraphs[0]
    p1.text = metric
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = color

    p2 = btf.add_paragraph()
    p2.text = label
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(8.5)
    p2.font.bold = True
    p2.font.color.rgb = WHITE

    p3 = btf.add_paragraph()
    p3.text = desc
    p3.font.name = "Segoe UI"
    p3.font.size = Pt(8)
    p3.font.color.rgb = TEXT_MUTED

# Bottom Row: 2 Structured Bento Panels
# Left Panel: Linguistic Challenges
add_panel(s2, Inches(0.8), Inches(3.0), Inches(5.7), Inches(3.9), bg=CARD_BG, border=BORDER_SUBTLE)
b1_box = s2.shapes.add_textbox(Inches(1.05), Inches(3.2), Inches(5.2), Inches(3.5))
tf = b1_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "ĐẶC THÙ NGÔN NGỮ TIẾNG VIỆT"
p.font.name = "Segoe UI"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = WHITE
p.space_after = Pt(8)

points_l = [
    ("Ranh giới từ ghép không tường minh", "Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ ghép (giảng_viên, cơ_sở_vật_chất) quyết định ngữ nghĩa, bắt buộc chuẩn hóa tách từ PyVi."),
    ("Cấu trúc phủ định đa tầng", "Phủ định kép và phủ định đảo vị trí ('không phải không tốt', 'chẳng thể chê') dễ làm sụp đổ các bộ phân loại từ khóa đơn lẻ."),
    ("Khẩu ngữ & Mỉa mai (Sarcasm)", "Sinh viên dùng tiếng lóng học đường, khen bề mặt nhưng mang ý chê ('cảm ơn trường cho trải nghiệm phòng học xông hơi giữa trưa nắng').")
]
for title, text in points_l:
    p = tf.add_paragraph()
    p.text = "▸ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(2)
    p_body = tf.add_paragraph()
    p_body.text = text
    p_body.font.name = "Segoe UI"
    p_body.font.size = Pt(8.5)
    p_body.font.color.rgb = TEXT_MUTED
    p_body.space_after = Pt(6)

# Right Panel: Label Distribution Details
add_panel(s2, Inches(6.8), Inches(3.0), Inches(5.733), Inches(3.9), bg=CARD_BG, border=BORDER_SUBTLE)
b2_box = s2.shapes.add_textbox(Inches(7.05), Inches(3.2), Inches(5.2), Inches(3.5))
tf = b2_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "PHÂN BỐ 2 TÁC VỤ PHÂN LOẠI"
p.font.name = "Segoe UI"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = WHITE
p.space_after = Pt(8)

points_r = [
    ("Tác vụ 1: Cảm xúc (Sentiment - 3 Lớp)", "• Tích cực (Positive): 48.2% — Khen ngợi, hài lòng về môn học\n• Tiêu cực (Negative): 47.8% — Bất mãn, phàn nàn, góp ý cải thiện\n• Trung tính (Neutral): 4.0% — Khách quan, cân bằng (Thách thức F1)"),
    ("Tác vụ 2: Chủ đề (Topic - 4 Lớp)", "• Giảng viên (Lecturer): 72.33% — Phương pháp, tác phong, truyền đạt\n• Chương trình đào tạo (Curriculum): 18.07% — Môn học, tiến độ, đồ án\n• Cơ sở vật chất (Facility): 4.58% — Phòng máy, điều hòa, Wi-Fi\n• Khác (Others): 5.02% — Học phí, hành chính, thủ tục một cửa"),
    ("Quy cách dữ liệu", "Phân đoạn câu dài tối đa 118 từ; 95% câu tập trung dưới 32 từ; tối ưu hóa kích thước chuỗi max_length = 128.")
]
for title, text in points_r:
    p = tf.add_paragraph()
    p.text = "▸ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO
    p.space_after = Pt(2)
    p_body = tf.add_paragraph()
    p_body.text = text
    p_body.font.name = "Segoe UI"
    p_body.font.size = Pt(8.5)
    p_body.font.color.rgb = TEXT_MUTED
    p_body.space_after = Pt(6)

# ==============================================================================
# SLIDE 3: Kiến Trúc Hai Luồng (Dual-Track System Architecture)
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
add_bg(s3)
add_header(s3, "2. Thiết Kế Kiến Trúc Hệ Thống Hai Luồng (Dual-Track)")

# Top Architecture Diagram Bar
add_panel(s3, Inches(0.8), Inches(1.4), Inches(11.733), Inches(0.9), bg=CARD_HOVER, border=BORDER_SUBTLE)
diag_box = s3.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(11.3), Inches(0.8))
tf = diag_box.text_frame
p = tf.paragraphs[0]
p.text = "SƠ ĐỒ LUỒNG DỮ LIỆU ĐỒNG THỜI (DUAL-TRACK PIPELINE)"
p.font.name = "Segoe UI"
p.font.size = Pt(9.5)
p.font.bold = True
p.font.color.rgb = ACCENT_CYAN

p2 = tf.add_paragraph()
p2.text = "Văn bản thô  ➔  Chuẩn hóa & PyVi  ➔  [Track 1: TF-IDF 10k → Dual LR]  &  [Track 2: PhoBERT-v2 → Shared CLS → Dual Heads]"
p2.font.name = "Segoe UI"
p2.font.size = Pt(11)
p2.font.bold = True
p2.font.color.rgb = WHITE

# Left Bento Card: Track 1 Classical Baseline
add_panel(s3, Inches(0.8), Inches(2.5), Inches(5.7), Inches(4.4), bg=CARD_BG, border=BORDER_SUBTLE)
b1 = s3.shapes.add_textbox(Inches(1.05), Inches(2.7), Inches(5.2), Inches(4.0))
tf = b1.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "LUỒNG 1: CLASSICAL BASELINE"
p.font.name = "Segoe UI"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_after = Pt(6)

b_points = [
    ("Đặc trưng TF-IDF n-gram (1, 2):", "Từ điển 10.000 chiều, áp dụng giãn tần suất sublinear_tf=True nhằm kiềm chế các từ xuất hiện quá dày đặc."),
    ("Dual Logistic Regression:", "2 bộ phân loại độc lập tối ưu bằng L-BFGS, điều chuẩn L2 (C=1.0) và class_weight='balanced' trợ lực lớp thiểu số."),
    ("Tốc độ siêu tốc 0.05 ms / câu:", "Đạt thông lượng 21.962 câu/giây trên CPU, kích thước tệp chỉ 0.95 MB, phản hồi tức thời cho bài toán phân luồng sơ bộ."),
    ("Triển khai Độc lập & Tinh gọn:", "Hoạt động thuần túy trên Scikit-Learn/PyVi, không cần GPU hay thư viện Deep Learning cồng kềnh, lý tưởng cho môi trường máy chủ biên (Edge/Microservice).")
]
for title, body in b_points:
    p = tf.add_paragraph()
    p.text = "✔ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(1)
    pb = tf.add_paragraph()
    pb.text = body
    pb.font.name = "Segoe UI"
    pb.font.size = Pt(8.5)
    pb.font.color.rgb = TEXT_MUTED
    pb.space_after = Pt(6)

# Right Bento Card: Track 2 PhoBERT Multi-Task
add_panel(s3, Inches(6.8), Inches(2.5), Inches(5.733), Inches(4.4), bg=CARD_BG, border=BORDER_ACTIVE)
b2 = s3.shapes.add_textbox(Inches(7.05), Inches(2.7), Inches(5.2), Inches(4.0))
tf = b2.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "LUỒNG 2: PHOBERT MULTI-TASK v2 (ĐỀ XUẤT)"
p.font.name = "Segoe UI"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_INDIGO
p.space_after = Pt(6)

p_points = [
    ("Backbone RoBERTa Chia Sẻ [CLS]:", "vinai/phobert-base-v2 (12 layers, 768-d). Biểu diễn đặc trưng ngữ cảnh sâu được chia sẻ trọn vẹn cho cả 2 nhiệm vụ trong 1 lượt tính."),
    ("Đầu Phân Loại Tuyến Tính Kép (Dual Heads):", "Linear(768 → 3) cho Cảm xúc và Linear(768 → 4) cho Chủ đề; triệt tiêu 60% RAM so với chạy 2 mô hình đơn nhiệm riêng lẻ."),
    ("Hàm Mất Mát Điều Hòa Lớp Thiểu Số:", "L_total = L_sent + L_topic kết hợp Trọng số Nghịch đảo Làm mịn (power=0.35) và Label Smoothing (ε=0.03) chống over-confidence."),
    ("Tối Ưu Huấn Luyện CPU Độc Quyền:", "Đóng băng 6 tầng đáy [0..5] + Hidden-state FP16 caching, tăng tốc 40% trên máy chủ không trang bị GPU rời.")
]
for title, body in p_points:
    p = tf.add_paragraph()
    p.text = "★ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(1)
    pb = tf.add_paragraph()
    pb.text = body
    pb.font.name = "Segoe UI"
    pb.font.size = Pt(8.5)
    pb.font.color.rgb = TEXT_MUTED
    pb.space_after = Pt(6)

# ==============================================================================
# SLIDE 4: Kết Quả Thực Nghiệm & Đối Đầu Hugging Face (Benchmark)
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
add_bg(s4)
add_header(s4, "3. Kết Quả Thực Nghiệm & Bảng Đối Đầu Hugging Face (Benchmark)")

# Table Bento Card
tbl_card = add_panel(s4, Inches(0.8), Inches(1.4), Inches(11.733), Inches(2.8), bg=CARD_BG, border=BORDER_SUBTLE)
tbl_shape = s4.shapes.add_table(5, 7, Inches(0.95), Inches(1.5), Inches(11.433), Inches(2.55))
table = tbl_shape.table

table.columns[0].width = Inches(2.9)
table.columns[1].width = Inches(1.3)
table.columns[2].width = Inches(1.3)
table.columns[3].width = Inches(1.4)
table.columns[4].width = Inches(1.3)
table.columns[5].width = Inches(1.4)
table.columns[6].width = Inches(1.833)

headers = ["MÔ HÌNH / PHƯƠNG PHÁP", "SENT F1", "TOPIC F1", "MEAN F1", "RAM", "ĐỘ TRỄ CPU", "ĐÁNH GIÁ TRIỂN KHAI"]
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Segoe UI"
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

bench_data = [
    ("Mô hình Cơ sở (TF-IDF + LR)", "71.90%", "72.72%", "72.31%", "~45 MB", "0.05 ms", "Siêu nhẹ, xử lý thời gian thực"),
    ("Pipeline HF Đơn nhiệm (zeroday117 + CafeBERT)", "87.82%", "88.41%", "88.12%", "~2.8 GB", "115.4 ms", "2 mô hình tuần tự, tốn RAM"),
    ("Zero-Shot NLI (DeBERTa-v3 Large)", "83.50%", "78.20%", "80.85%", "~1.6 GB", "1,250 ms", "Suy luận 7 giả thuyết, rất chậm"),
    ("PhoBERT Multi-Task (Đề xuất)", "82.89%", "80.10%", "81.50%", "~1.1 GB", "30.1 ms", "Tối ưu RAM/Tốc độ vượt trội")
]

for i, row in enumerate(bench_data, start=1):
    is_proposed = (i == 4)
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.fill.solid()
        if is_proposed:
            cell.fill.fore_color.rgb = RGBColor(30, 41, 75)
        else:
            cell.fill.fore_color.rgb = CARD_BG if i % 2 == 1 else RGBColor(15, 23, 38)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Segoe UI"
        p.font.size = Pt(9)
        p.font.bold = is_proposed
        if is_proposed:
            p.font.color.rgb = WHITE if j == 0 else ACCENT_CYAN
        else:
            p.font.color.rgb = WHITE if j == 0 else TEXT_MUTED
        if j in [1, 2, 3, 4, 5]:
            p.alignment = PP_ALIGN.CENTER

# 3 Big Value Cards Below Table
values = [
    ("TIẾT KIỆM 60% RAM", "~1.1 GB vs ~2.8 GB", "Chia sẻ toàn bộ 12 tầng Transformer chung, loại bỏ nhu cầu tải đồng thời PhoBERT v2 và CafeBERT riêng rẽ, vận hành bền bỉ trên máy chủ phổ thông.", ACCENT_CYAN),
    ("NHANH GẤP 3.8 LẦN", "30.1 ms vs 115.4 ms", "Chỉ tính toán ma trận Attention một lần duy nhất cho cả hai tác vụ; nhanh hơn 41.5 lần so với tiếp cận Zero-Shot NLI (1,250 ms) trên cùng phần cứng CPU.", ACCENT_EMERALD),
    ("BỨT PHÁ LỚP NEUTRAL", "+23.07% F1-Score", "Cơ chế Smoothed Class Weights giải quyết xuất sắc điểm nghẽn lớp thiểu số: F1 lớp Trung tính tăng từ 35.12% lên 58.19%, nâng Mean Macro-F1 toàn hệ thống lên 81.50%.", ACCENT_AMBER)
]

for idx, (title, stat, desc, col) in enumerate(values):
    left_c = Inches(0.8 + idx * 4.0)
    add_panel(s4, left_c, Inches(4.35), Inches(3.733), Inches(2.55), bg=CARD_BG, border=BORDER_SUBTLE)
    vbox = s4.shapes.add_textbox(left_c + Inches(0.2), Inches(4.45), Inches(3.33), Inches(2.35))
    tf = vbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Segoe UI"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = col
    p.space_after = Pt(2)

    p_st = tf.add_paragraph()
    p_st.text = stat
    p_st.font.name = "Segoe UI"
    p_st.font.size = Pt(14)
    p_st.font.bold = True
    p_st.font.color.rgb = WHITE
    p_st.space_after = Pt(4)

    pd = tf.add_paragraph()
    pd.text = desc
    pd.font.name = "Segoe UI"
    pd.font.size = Pt(8.5)
    pd.font.color.rgb = TEXT_MUTED

# ==============================================================================
# SLIDE 5: Triển Khai Web Demo & Tổng Kết
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
add_bg(s5)
add_header(s5, "4. Triển Khai Ứng Dụng Thực Tế & Kết Luận")

# Left Bento Card: Web App Streamlit
add_panel(s5, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.5), bg=CARD_BG, border=BORDER_SUBTLE)
wbox = s5.shapes.add_textbox(Inches(1.05), Inches(1.6), Inches(5.2), Inches(5.1))
tf = wbox.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "ỨNG DỤNG STREAMLIT DASHBOARD"
p.font.name = "Segoe UI"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(8)

w_items = [
    ("Top Control Deck chuẩn Taste-Skill", "Thanh điều khiển nằm ngang ở đỉnh trang, loại bỏ thanh trượt sidebar rối mắt, tối ưu chiều cao hiển thị và căn khớp 38px hoàn hảo."),
    ("Song ngữ Việt – Anh Toàn diện", "Tích hợp sẵn bộ từ vựng và 2 thư viện mẫu câu độc lập (10 câu VI & 10 câu EN), hỗ trợ kiểm thử đối ứng song song."),
    ("Chế độ Dự đoán Đa năng", "Cho phép lựa chọn linh hoạt: Chế độ đơn (Trung bình hoặc Cao) hoặc So sánh song song (Cả hai) hiển thị phân bố xác suất trực quan."),
    ("Consensus Engine & Biểu đồ Donut", "Tự động so khớp kết quả giữa Baseline và PhoBERT, phân tích điểm bất đồng ngữ nghĩa, tỷ lệ phân tán và chênh lệch tốc độ thực tế."),
    ("Cơ chế Dự phòng Lỗi CUDA OOM", "Tự động phát hiện tràn bộ nhớ GPU và chuyển tải thông minh sang CPU, đảm bảo độ ổn định 100% trong môi trường demo.")
]
for title, text in w_items:
    p = tf.add_paragraph()
    p.text = "◆ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(1)
    pb = tf.add_paragraph()
    pb.text = text
    pb.font.name = "Segoe UI"
    pb.font.size = Pt(8.5)
    pb.font.color.rgb = TEXT_MUTED
    pb.space_after = Pt(5)

# Right Bento Card: Deliverables & Roadmap
add_panel(s5, Inches(6.8), Inches(1.4), Inches(5.733), Inches(5.5), bg=CARD_BG, border=BORDER_SUBTLE)
cbox = s5.shapes.add_textbox(Inches(7.05), Inches(1.6), Inches(5.2), Inches(5.1))
tf = cbox.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "TỔNG KẾT ĐÓNG GÓP & HƯỚNG PHÁT TRIỂN"
p.font.name = "Segoe UI"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_INDIGO
p.space_after = Pt(8)

c_items = [
    ("Đóng góp kỹ thuật hoàn chỉnh", "Xây dựng trọn vẹn đường ống ML/DL hai luồng từ chuẩn hóa Parquet, huấn luyện cân bằng lớp, đối chiếu benchmark công bằng đến đóng gói sản phẩm."),
    ("Hiệu quả kiến trúc Đa nhiệm", "Chứng minh học đa nhiệm (MTL) chia sẻ biểu diễn CLS vừa giúp nâng cao chất lượng mô hình (+9.19 F1) vừa tiết kiệm tối đa chi phí hạ tầng CPU (30 ms)."),
    ("Bảo đảm chất lượng phần mềm", "Vận hành bộ kiểm thử tự động 12 module (327 test cases: 185 unit tests + 142 E2E tests) đạt tỉ lệ đạt 100% PASS."),
    ("Định hướng tối ưu hóa thời gian thực", "Xuất mô hình sang định dạng ONNX Runtime INT8 lượng tử hóa nhằm hạ thấp độ trễ xuống dưới 8 ms/câu trên CPU thông dụng."),
    ("Mở rộng sang Aspect-Based ABSA", "Nâng cấp từ bài toán phân loại chủ đề đơn lẻ sang trích xuất khía cạnh đa nhãn và đo lường cảm xúc tương ứng cho các câu nhận xét phức hợp.")
]
for title, text in c_items:
    p = tf.add_paragraph()
    p.text = "★ " + title
    p.font.name = "Segoe UI"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(1)
    pb = tf.add_paragraph()
    pb.text = text
    pb.font.name = "Segoe UI"
    pb.font.size = Pt(8.5)
    pb.font.color.rgb = TEXT_MUTED
    pb.space_after = Pt(5)

output_path = "Slide_BaoCao_DoAn_UIT_VSFC.pptx"
prs.save(output_path)
print(f"SUCCESS: Created presentation {output_path}")
