# -*- coding: utf-8 -*-
"""
scripts/generate_slides.py
Generates a polished 5-slide PowerPoint deck for the UIT-VSFC Multi-Task project.
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

# Color Palette
DARK_NAVY = RGBColor(15, 23, 42)       # #0F172A
SLATE_HEAD = RGBColor(30, 41, 59)      # #1E293B
SLATE_BODY = RGBColor(71, 85, 105)     # #475569
PRIMARY_BLUE = RGBColor(26, 75, 224)   # #1A4BE0
ACCENT_BLUE = RGBColor(37, 99, 235)    # #2563EB
EMERALD = RGBColor(16, 185, 129)       # #10B981
INDIGO = RGBColor(99, 102, 241)        # #6366F1
BG_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC
CARD_BG = RGBColor(255, 255, 255)      # #FFFFFF
BORDER_COLOR = RGBColor(226, 232, 240) # #E2E8F0
WHITE = RGBColor(255, 255, 255)
LIGHT_BLUE_BG = RGBColor(238, 242, 255)
MUTED_ROW = RGBColor(241, 245, 249)

def add_bg(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card

def add_header(slide, title_text, category_text="BÁO CÁO ĐỒ ÁN AI NÂNG CAO"):
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf = tag_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = category_text.upper()
    p.font.name = "Arial"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf2 = title_box.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title_text
    p2.font.name = "Arial"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = DARK_NAVY

# ==============================================================================
# SLIDE 1: Title Slide (Dark Hero Theme)
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
add_bg(s1, DARK_NAVY)

# University & Department
univ_box = s1.shapes.add_textbox(Inches(1.0), Inches(0.85), Inches(11.3), Inches(0.5))
p = univ_box.text_frame.paragraphs[0]
p.text = "ĐẠI HỌC DUY TÂN — TRƯỜNG KHOA HỌC MÁY TÍNH"
p.font.name = "Arial"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = INDIGO

course_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.25), Inches(11.3), Inches(0.5))
p = course_box.text_frame.paragraphs[0]
p.text = "MÔN HỌC: TRÍ TUỆ NHÂN TẠO NÂNG CAO (CS 467 C)"
p.font.name = "Arial"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = RGBColor(148, 163, 184)

# Main Title & Subtitle
title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(11.3), Inches(2.1))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "XÂY DỰNG HỆ THỐNG PHÂN LOẠI ĐA NHIỆM\nPHẢN HỒI SINH VIÊN (UIT-VSFC)"
p.font.name = "Arial"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE
p.space_after = Pt(10)

p_sub = tf.add_paragraph()
p_sub.text = "Dự đoán đồng thời Cảm xúc (Sentiment) & Chủ đề (Topic) sử dụng PhoBERT Multi-Task v2 và Classical Baseline"
p_sub.font.name = "Arial"
p_sub.font.size = Pt(13.5)
p_sub.font.color.rgb = RGBColor(203, 213, 225)

# Badges
badge_box = s1.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.3), Inches(0.5))
p = badge_box.text_frame.paragraphs[0]
p.text = "⚡ Classical Baseline (TF-IDF + LR)   |   🧠 PhoBERT Multi-Task v2 (Shared CLS)   |   🌐 Bilingual VI - EN Streamlit"
p.font.name = "Arial"
p.font.size = Pt(10.5)
p.font.bold = True
p.font.color.rgb = EMERALD

# Footer Card with Instructors & Authors
add_card(s1, Inches(1.0), Inches(5.1), Inches(11.333), Inches(1.65), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))

info_box = s1.shapes.add_textbox(Inches(1.3), Inches(5.25), Inches(10.7), Inches(1.35))
tf = info_box.text_frame
p1 = tf.paragraphs[0]
p1.text = "Giảng viên hướng dẫn: TS. Vikram Puri"
p1.font.name = "Arial"
p1.font.size = Pt(12)
p1.font.bold = True
p1.font.color.rgb = WHITE
p1.space_after = Pt(4)

p2 = tf.add_paragraph()
p2.text = "Sinh viên thực hiện:"
p2.font.name = "Arial"
p2.font.size = Pt(11)
p2.font.bold = True
p2.font.color.rgb = RGBColor(148, 163, 184)
p2.space_after = Pt(2)

p3 = tf.add_paragraph()
p3.text = "• Nguyễn Trung Hiếu  —  MSSV: 29211362521        • Lê Văn Hoài Rin  —  MSSV: 29211332113"
p3.font.name = "Arial"
p3.font.size = Pt(11.5)
p3.font.color.rgb = RGBColor(241, 245, 249)

# ==============================================================================
# SLIDE 2: Bài toán & Tập Dữ Liệu UIT-VSFC
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
add_bg(s2, BG_LIGHT)
add_header(s2, "1. Đặt Vấn Đề & Khám Phá Tập Dữ Liệu UIT-VSFC")

# Left Card: Problem Formulation & Linguistic Challenges
add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
c1_box = s2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tf1 = c1_box.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "Bối Cảnh & Thách Thức Ngôn Ngữ"
p.font.name = "Arial"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = DARK_NAVY
p.space_after = Pt(8)

bullets1 = [
    ("Mục tiêu thực tiễn:", " Tự động hóa đánh giá ý kiến phản hồi của sinh viên đại học phục vụ công tác kiểm định và nâng cao chất lượng đào tạo."),
    ("Bài toán Học Đa Nhiệm (MTL):", " Dự đoán đồng thời 2 nhãn từ một câu văn bản:\n  • Cảm xúc (Sentiment - 3 lớp): Tiêu cực (0), Trung tính (1), Tích cực (2).\n  • Chủ đề (Topic - 4 lớp): Giảng viên (0), Chương trình (1), Cơ sở vật chất (2), Khác (3)."),
    ("Ranh giới từ ghép tiếng Việt:", " Ngôn ngữ đơn lập, từ ghép phân tách không tường minh (ví dụ: 'giảng_viên', 'tận_tâm'), đòi hỏi tiền xử lý tách từ PyVi nhất quán."),
    ("Mất cân bằng lớp nghiêm trọng:", " Lớp Trung tính (Neutral) chỉ chiếm 4.01% ở tập huấn luyện, dễ bị các mô hình thông thường bỏ qua hoàn toàn."),
    ("Khẩu ngữ & mỉa mai:", " Sinh viên thường viết tắt, dùng tiếng lóng học đường hoặc khen bề mặt nhưng mang ý chê ('cảm ơn trường cho trải nghiệm phòng học xông hơi').")
]
for title, body in bullets1:
    p = tf1.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

# Right Card: Dataset Statistics
add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
c2_box = s2.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf2 = c2_box.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "Quy Mô & Phân Bố Dữ Liệu (16.175 Mẫu)"
p.font.name = "Arial"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = DARK_NAVY
p.space_after = Pt(8)

bullets2 = [
    ("Phân chia chuẩn dữ liệu:", " 11.426 câu Train (70.6%) | 1.583 câu Validation (9.8%) | 3.166 câu Test độc lập (19.6%)."),
    ("Phân bố nhãn Cảm xúc (Sentiment):", "\n  • Tích cực (Positive): ~48.2%\n  • Tiêu cực (Negative): ~47.8%\n  • Trung tính (Neutral): ~4.0% (Lớp thiểu số cực đoan)"),
    ("Phân bố nhãn Chủ đề (Topic):", "\n  • Giảng viên (Lecturer): 72.33% (Chiếm đa số áp đảo)\n  • Chương trình đào tạo (Training Program): 18.07%\n  • Cơ sở vật chất (Facility): 4.58%\n  • Khác (Others): 5.02%"),
    ("Đặc điểm độ dài văn bản:", " Trung bình 15.6 từ/câu, độ dài cực đại đạt 118 từ, phân bố tập trung dày trong khoảng 8–32 từ, phù hợp cấu hình max_length = 128."),
    ("Lưu trữ chuẩn hóa:", " Định dạng Apache Parquet nén Snappy, kiểm tra nghiêm ngặt không tồn tại bản ghi rỗng (null).")
]
for title, body in bullets2:
    p = tf2.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

# ==============================================================================
# SLIDE 3: Phương Pháp & Kiến Trúc Hệ Thống (Dual-Track)
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
add_bg(s3, BG_LIGHT)
add_header(s3, "2. Kiến Trúc Hệ Thống & Phương Pháp Thực Hiện")

# Track 1 Card: Baseline
add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
t1_box = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tft1 = t1_box.text_frame
tft1.word_wrap = True

p = tft1.paragraphs[0]
p.text = "Luồng 1: Classical Baseline (TF-IDF + Dual LR)"
p.font.name = "Arial"
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = EMERALD
p.space_after = Pt(6)

bullets_t1 = [
    ("Tiền xử lý văn bản:", " Tách từ tiếng Việt qua PyVi, loại bỏ ký tự rác, chuẩn hóa khoảng trắng và hạ chữ thường."),
    ("Trích xuất đặc trưng TF-IDF:", " Khai thác n-gram từ vựng (1, 2) với tối đa 10.000 chiều, áp dụng log scaling (sublinear_tf=True) nhằm giảm thiểu ảnh hưởng của các từ tần suất quá cao."),
    ("Bộ phân loại kép độc lập:", " 2 mô hình Logistic Regression sử dụng thuật toán tối ưu lồi L-BFGS, điều chuẩn L2 (C=1.0), áp dụng class_weight='balanced' để hỗ trợ lớp thiểu số."),
    ("Ưu điểm vận hành:", "\n  • Kích thước tệp lưu trữ siêu nhẹ: ~0.95 MB (baseline.joblib)\n  • Độ trễ suy luận tức thì: 0.05 ms/câu\n  • Tốc độ thông lượng CPU: 21.962 câu/giây\n  • Phù hợp xử lý batch dữ liệu khổng lồ theo thời gian thực.")
]
for title, body in bullets_t1:
    p = tft1.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

# Track 2 Card: PhoBERT Multi-Task
add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
t2_box = s3.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tft2 = t2_box.text_frame
tft2.word_wrap = True

p = tft2.paragraphs[0]
p.text = "Luồng 2: PhoBERT Multi-Task v2 (Đề Xuất)"
p.font.name = "Arial"
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = PRIMARY_BLUE
p.space_after = Pt(6)

bullets_t2 = [
    ("Backbone ngôn ngữ:", " vinai/phobert-base-v2 (12 tầng Transformer, biểu diễn CLS 768 chiều), chia sẻ toàn bộ trọng số ngữ cảnh sâu cho cả hai bài toán."),
    ("Đầu ra đa nhiệm (Dual Heads):", " 2 phân nhánh tuyến tính song song (Linear Head 768→3 cho Cảm xúc và 768→4 cho Chủ đề), thực hiện 1 forward pass duy nhất."),
    ("Hàm mất mát kết hợp (Joint Loss):", " L_total = L_sent + L_topic kết hợp Trọng số Nghịch đảo Làm mịn (Smoothed Class Weights, power=0.35) và Làm mịn Nhãn (Label Smoothing ε=0.03)."),
    ("Tốc độ học phân tầng (Differential LR):", " Tốc độ học Backbone 3e-5 (bảo toàn tri thức tiền huấn luyện) kết hợp Head LR 4.5e-4 (hội tụ nhanh trên nhiệm vụ mới)."),
    ("Tối ưu CPU độc quyền:", " Đóng băng 6 tầng đáy [0..5] kết hợp lưu đệm trạng thái ẩn FP16 giúp rút ngắn thời gian huấn luyện tới 40% trên CPU.")
]
for title, body in bullets_t2:
    p = tft2.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

# ==============================================================================
# SLIDE 4: Kết Quả Thực Nghiệm & Đối Đầu Hugging Face (Benchmark)
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
add_bg(s4, BG_LIGHT)
add_header(s4, "3. Kết Quả Thực Nghiệm & So Sánh Đối Đầu (Benchmark)")

# Benchmark Table
rows = 5
cols = 7
tbl_shape = s4.shapes.add_table(rows, cols, Inches(0.8), Inches(1.55), Inches(11.733), Inches(2.5))
table = tbl_shape.table

table.columns[0].width = Inches(2.7)
table.columns[1].width = Inches(1.5)
table.columns[2].width = Inches(1.5)
table.columns[3].width = Inches(1.5)
table.columns[4].width = Inches(1.5)
table.columns[5].width = Inches(1.5)
table.columns[6].width = Inches(1.533)

headers = ["Mô hình / Tiếp cận", "Sent F1", "Topic F1", "Mean F1", "RAM", "Độ trễ CPU", "Đánh giá"]
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_NAVY
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = "Arial"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

bench_data = [
    ("Mô hình Cơ sở (TF-IDF + LR)", "71.90%", "72.72%", "72.31%", "~45 MB", "0.05 ms", "Siêu nhẹ, tốc độ cao"),
    ("Pipeline HF Đơn nhiệm (zeroday117 + CafeBERT)", "87.82%", "88.41%", "88.12%", "~2.8 GB", "115.4 ms", "Cồng kềnh, tốn RAM"),
    ("Zero-Shot NLI (DeBERTa-v3 Large)", "83.50%", "78.20%", "80.85%", "~1.6 GB", "1,250 ms", "Rất chậm, đa ngữ"),
    ("PhoBERT Multi-Task (Đề xuất)", "82.89%", "80.10%", "81.50%", "~1.1 GB", "30.1 ms", "Tối ưu toàn diện")
]

for i, row in enumerate(bench_data, start=1):
    is_best = (i == 4)
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.fill.solid()
        if is_best:
            cell.fill.fore_color.rgb = LIGHT_BLUE_BG
        else:
            cell.fill.fore_color.rgb = CARD_BG if i % 2 == 1 else MUTED_ROW
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = "Arial"
        p.font.size = Pt(9)
        p.font.bold = is_best
        if is_best:
            p.font.color.rgb = PRIMARY_BLUE
        else:
            p.font.color.rgb = DARK_NAVY if j == 0 else SLATE_BODY
        if j > 0 and j < 6:
            p.alignment = PP_ALIGN.CENTER

# Summary Insights Cards below table
add_card(s4, Inches(0.8), Inches(4.3), Inches(3.7), Inches(2.6))
ibox1 = s4.shapes.add_textbox(Inches(0.95), Inches(4.45), Inches(3.4), Inches(2.3))
tf = ibox1.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Tiết Kiệm 60% Bộ Nhớ RAM"
p.font.name = "Arial"
p.font.size = Pt(11.5)
p.font.bold = True
p.font.color.rgb = PRIMARY_BLUE
p.space_after = Pt(4)
p2 = tf.add_paragraph()
p2.text = "Việc chia sẻ một xương sống RoBERTa duy nhất giúp giảm RAM từ ~2.8 GB (khi chạy 2 mô hình đơn nhiệm riêng biệt) xuống còn ~1.1 GB, cho phép vận hành ổn định trên máy chủ tài nguyên hạn chế."
p2.font.name = "Arial"
p2.font.size = Pt(9)
p2.font.color.rgb = SLATE_BODY

add_card(s4, Inches(4.8), Inches(4.3), Inches(3.7), Inches(2.6))
ibox2 = s4.shapes.add_textbox(Inches(4.95), Inches(4.45), Inches(3.4), Inches(2.3))
tf = ibox2.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Tốc Độ Suy Luận Nhanh 3.8x"
p.font.name = "Arial"
p.font.size = Pt(11.5)
p.font.bold = True
p.font.color.rgb = EMERALD
p.space_after = Pt(4)
p2 = tf.add_paragraph()
p2.text = "Nhờ chỉ tính toán ma trận Attention một lần duy nhất cho cả hai tác vụ, PhoBERT Multi-Task đạt 30.1 ms/câu, nhanh gấp 3.8 lần pipeline ghép tuần tự và nhanh gấp 41.5 lần Zero-Shot NLI."
p2.font.name = "Arial"
p2.font.size = Pt(9)
p2.font.color.rgb = SLATE_BODY

add_card(s4, Inches(8.8), Inches(4.3), Inches(3.733), Inches(2.6))
ibox3 = s4.shapes.add_textbox(Inches(8.95), Inches(4.45), Inches(3.4), Inches(2.3))
tf = ibox3.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Đột Phá Lớp Thiểu Số Neutral"
p.font.name = "Arial"
p.font.size = Pt(11.5)
p.font.bold = True
p.font.color.rgb = INDIGO
p.space_after = Pt(4)
p2 = tf.add_paragraph()
p2.text = "Nhờ cơ chế trọng số mất mát nghịch đảo làm mịn và nhãn mượt, điểm F1 lớp Trung tính (neutral) nhảy vọt từ 35.12% (Baseline) lên 58.19% (+23.07%), Mean Macro-F1 toàn diện tăng +9.19 điểm."
p2.font.name = "Arial"
p2.font.size = Pt(9)
p2.font.color.rgb = SLATE_BODY

# ==============================================================================
# SLIDE 5: Triển Khai Ứng Dụng Web & Kết Luận
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
add_bg(s5, BG_LIGHT)
add_header(s5, "4. Triển Khai Ứng Dụng Web & Kết Luận")

# Left Card: Web Application
add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
w_box = s5.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tfw = w_box.text_frame
tfw.word_wrap = True

p = tfw.paragraphs[0]
p.text = "Hệ Thống Trực Quan Hóa (Streamlit Dashboard)"
p.font.name = "Arial"
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = PRIMARY_BLUE
p.space_after = Pt(6)

bullets_w = [
    ("Chuẩn giao diện Taste Skill:", " Thanh điều khiển Top Deck nằm ngang gọn gàng, loại bỏ hoàn toàn thanh trượt bên trái gây rối mắt, tối ưu chiều cao hiển thị."),
    ("Hỗ trợ Song ngữ Toàn diện:", " Giao diện chuyển đổi tức thì giữa Tiếng Việt và English kèm 2 thư viện câu nhận xét mẫu sinh viên độc lập (10 câu VI, 10 câu EN)."),
    ("Chế độ so sánh đối đầu:", " Hỗ trợ chọn chế độ đơn (Trung bình hoặc Cao) hoặc so sánh song song cả 2 mô hình theo thời gian thực."),
    ("Biểu đồ phân bố xác suất:", " Hiển thị trực quan Donut charts và thanh tỷ lệ xác suất từng lớp, tự động đánh dấu nhãn dự đoán cao nhất (Top label)."),
    ("Engine đồng thuận (Consensus):", " Tự động đối chiếu kết quả dự đoán giữa Baseline và PhoBERT, phân tích điểm bất đồng ngữ nghĩa và chênh lệch tốc độ."),
    ("Cơ chế an toàn CUDA OOM:", " Tự động chuyển tải dự phòng sang CPU nếu bộ nhớ GPU cạn kiệt, đảm bảo độ sẵn sàng 100% khi demo trực tiếp.")
]
for title, body in bullets_w:
    p = tfw.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(3.5)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

# Right Card: Conclusion & Future Work
add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
c_box = s5.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tfc = c_box.text_frame
tfc.word_wrap = True

p = tfc.paragraphs[0]
p.text = "Tổng Kết Đề Tài & Hướng Phát Triển"
p.font.name = "Arial"
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = DARK_NAVY
p.space_after = Pt(6)

bullets_c = [
    ("Đóng góp chính của đề tài:", "\n  1. Xây dựng hoàn chỉnh đường ống Học đa nhiệm hai luồng từ tiền xử lý, huấn luyện, đánh giá đến đóng gói mô hình.\n  2. PhoBERT Multi-Task đạt 81.50% Mean Macro-F1 (+9.19 điểm), cải thiện xuất sắc lớp thiểu số Neutral (+23.07% F1).\n  3. Đạt hiệu năng suy luận 30.1 ms/câu, tiết kiệm 60% RAM, vận hành độc lập trên CPU không phụ thuộc GPU."),
    ("Bộ kiểm thử tự động toàn diện:", " 12 module kiểm thử (185 unit tests + 142 E2E tests, tổng cộng 327 test cases) kiểm chứng toàn bộ logic và giao diện."),
    ("Định hướng phát triển tương lai:", "\n  • Lượng tử hóa mô hình (ONNX Runtime INT8): Giảm thêm độ trễ suy luận xuống dưới 8 ms/câu.\n  • Tinh chỉnh toàn phần (Full 12 layers) trên cụm máy chủ GPU và bổ sung hàm Focal Loss.\n  • Mở rộng sang bài toán Khai phá Khía cạnh Đa nhãn (Aspect-Based Sentiment Analysis - ABSA).")
]
for title, body in bullets_c:
    p = tfc.add_paragraph()
    p.text = "• " + title
    p.font.name = "Arial"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = SLATE_HEAD
    p.space_after = Pt(4)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = SLATE_BODY

output_path = "Slide_BaoCao_DoAn_UIT_VSFC.pptx"
prs.save(output_path)
print(f"SUCCESS: Generated {output_path} with {len(prs.slides)} slides.")
