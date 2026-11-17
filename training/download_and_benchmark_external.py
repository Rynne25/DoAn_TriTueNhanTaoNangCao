"""
training/download_and_benchmark_external.py
Script tải xuống, kiểm tra và benchmark các mô hình Hugging Face bên ngoài:
- Nhóm A (Tiếng Việt): zeroday117/phobert-vsfc-sentiment3, thnhan3, BaoNhan/cafebert-UITVSFC-T
- Nhóm B (Tiếng Anh): MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
"""

import argparse
import time
import json
import os
from typing import Dict, Any


def test_vietnamese_models(sample_text: str = "Thầy dạy rất tận tâm và dễ hiểu, nhưng phòng học hơi nóng."):
    print("\n" + "=" * 65)
    print("  ĐÁNH GIÁ MÔ HÌNH TIẾNG VIỆT HUGGING FACE (NHÓM A)")
    print("=" * 65)
    print(f"Câu kiểm tra: '{sample_text}'\n")

    from transformers import pipeline

    # 1. zeroday117 Sentiment
    print("[1/3] Đang tải & dự đoán: zeroday117/phobert-vsfc-sentiment3...")
    t0 = time.perf_counter()
    p_zero = pipeline("text-classification", model="zeroday117/phobert-vsfc-sentiment3", device=-1)
    res_zero = p_zero(sample_text)[0]
    dt_zero = (time.perf_counter() - t0) * 1000
    print(f"  -> Kết quả Cảm xúc: {res_zero['label']} (độ tin cậy: {res_zero['score']:.4f}) | Thời gian: {dt_zero:.1f}ms")

    # 2. thnhan3 Sentiment
    print("\n[2/3] Đang tải & dự đoán: thnhan3/phobert-vietnamese-students-feedback...")
    t0 = time.perf_counter()
    p_thnhan = pipeline("text-classification", model="thnhan3/phobert-vietnamese-students-feedback", device=-1)
    res_thnhan = p_thnhan(sample_text)[0]
    dt_thnhan = (time.perf_counter() - t0) * 1000
    print(f"  -> Kết quả Cảm xúc: {res_thnhan['label']} (độ tin cậy: {res_thnhan['score']:.4f}) | Thời gian: {dt_thnhan:.1f}ms")

    # 3. BaoNhan CafeBERT Topic
    print("\n[3/3] Đang tải & dự đoán: BaoNhan/cafebert-UITVSFC-T (Dung lượng ~2.1GB)...")
    t0 = time.perf_counter()
    p_nhan = pipeline("text-classification", model="BaoNhan/cafebert-UITVSFC-T", device=-1)
    res_nhan = p_nhan(sample_text)[0]
    dt_nhan = (time.perf_counter() - t0) * 1000
    print(f"  -> Kết quả Chủ đề: {res_nhan['label']} (độ tin cậy: {res_nhan['score']:.4f}) | Thời gian: {dt_nhan:.1f}ms")


def test_english_zero_shot(sample_text: str = "The professor is knowledgeable and helpful, but the classroom Wi-Fi is unstable."):
    print("\n" + "=" * 65)
    print("  ĐÁNH GIÁ MÔ HÌNH TIẾNG ANH ZERO-SHOT DEBERTA-V3 (NHÓM B)")
    print("=" * 65)
    print(f"Câu kiểm tra: '{sample_text}'\n")

    from transformers import pipeline

    model_id = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    print(f"Đang tải & dự đoán Zero-Shot: {model_id} (Dung lượng ~830MB)...")
    t0 = time.perf_counter()
    classifier = pipeline("zero-shot-classification", model=model_id, device=-1)

    s_res = classifier(
        sample_text,
        candidate_labels=["positive", "neutral", "negative"],
        hypothesis_template="The student sentiment of this feedback is {}.",
    )
    t_res = classifier(
        sample_text,
        candidate_labels=["lecturer", "training program", "facility", "others"],
        hypothesis_template="This feedback discusses {}.",
    )
    dt = (time.perf_counter() - t0) * 1000

    print(f"  -> Cảm xúc Zero-Shot: {s_res['labels'][0]} ({s_res['scores'][0]:.4f})")
    print(f"  -> Chủ đề Zero-Shot:  {t_res['labels'][0]} ({t_res['scores'][0]:.4f})")
    print(f"  -> Tổng thời gian suy luận: {dt:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Tải và benchmark mô hình Hugging Face bên ngoài.")
    parser.add_argument("--group", choices=["vi", "en", "all"], default="vi", help="Nhóm mô hình cần test (vi, en, all)")
    args = parser.parse_args()

    if args.group in ("vi", "all"):
        test_vietnamese_models()
    if args.group in ("en", "all"):
        test_english_zero_shot()


if __name__ == "__main__":
    main()
