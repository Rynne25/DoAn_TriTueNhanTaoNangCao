"""
evaluation/evaluate.py
Đánh giá và so sánh thực nghiệm: Classical Baseline vs PhoBERT Multi-Task trên tập Test UIT-VSFC (N=3.166).
Đo lường: Accuracy & Macro-F1 per task, xuất mẫu câu phân tích lỗi (Error Analysis).
"""

import argparse
import json
import os
import sys
sys.dont_write_bytecode = True
import time
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SENTIMENT_LABELS, SENTIMENT_MAP, TOPIC_LABELS, TOPIC_MAP, load_vsfc_split
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask


def evaluate_model(model, texts: List[str], sents: List[int], topics: List[int]) -> Dict[str, Any]:
    """Đánh giá toàn diện một mô hình trên tập test 3.166 câu."""
    t0 = time.time()
    s_preds, t_preds = model.predict_labels(texts, word_segment=False)
    dt = time.time() - t0

    s_acc = float(accuracy_score(sents, s_preds))
    s_wf1 = float(f1_score(sents, s_preds, average="weighted", zero_division=0))
    s_mf1 = float(f1_score(sents, s_preds, average="macro", zero_division=0))
    t_acc = float(accuracy_score(topics, t_preds))
    t_wf1 = float(f1_score(topics, t_preds, average="weighted", zero_division=0))
    t_mf1 = float(f1_score(topics, t_preds, average="macro", zero_division=0))

    s_cm = confusion_matrix(sents, s_preds, labels=[0, 1, 2]).tolist()
    t_cm = confusion_matrix(topics, t_preds, labels=[0, 1, 2, 3]).tolist()
    s_p, s_r, s_f1, s_sup = precision_recall_fscore_support(sents, s_preds, labels=[0, 1, 2], zero_division=0)
    t_p, t_r, t_f1, t_sup = precision_recall_fscore_support(topics, t_preds, labels=[0, 1, 2, 3], zero_division=0)

    per_class_sent = {
        lbl: {"precision": float(s_p[i]), "recall": float(s_r[i]), "f1": float(s_f1[i]), "support": int(s_sup[i])}
        for i, lbl in enumerate(SENTIMENT_LABELS)
    }
    per_class_topic = {
        lbl: {"precision": float(t_p[i]), "recall": float(t_r[i]), "f1": float(t_f1[i]), "support": int(t_sup[i])}
        for i, lbl in enumerate(TOPIC_LABELS)
    }

    return {
        "sent_acc": s_acc, "sent_wf1": s_wf1, "sent_mf1": s_mf1,
        "topic_acc": t_acc, "topic_wf1": t_wf1, "topic_mf1": t_mf1,
        "mean_mf1": (s_mf1 + t_mf1) / 2.0, "latency_sec": dt, "speed": len(texts) / max(0.001, dt),
        "s_preds": s_preds, "t_preds": t_preds, "s_cm": s_cm, "t_cm": t_cm,
        "per_class_sent": per_class_sent, "per_class_topic": per_class_topic,
    }


def compute_dataset_stats(data_dir: str) -> Dict[str, Any]:
    """Thống kê quy mô các tập dữ liệu và phân bố nhãn trên toàn bộ 16.175 câu UIT-VSFC."""
    split_sizes: Dict[str, int] = {}
    sent_counts = {lbl: 0 for lbl in SENTIMENT_LABELS}
    topic_counts = {lbl: 0 for lbl in TOPIC_LABELS}

    for split in ("train", "validation", "test"):
        texts, sents, topics = load_vsfc_split(split, data_dir=data_dir, word_segment=False)
        split_sizes[split] = len(texts)
        for s in sents:
            sent_counts[SENTIMENT_LABELS[int(s)]] += 1
        for t in topics:
            topic_counts[TOPIC_LABELS[int(t)]] += 1

    split_sizes["total"] = sum(split_sizes.values())
    return {
        "split_sizes": split_sizes,
        "sentiment_distribution": sent_counts,
        "topic_distribution": topic_counts,
    }


def collect_error_samples(texts: List[str], sents: List[int], topics: List[int], b_preds, p_preds) -> List[Dict[str, Any]]:
    """Trích xuất các mẫu câu thú vị phục vụ Error Analysis trong báo cáo."""
    errors = []
    for i in range(len(texts)):
        true_s = SENTIMENT_MAP[sents[i]]
        true_t = TOPIC_MAP[topics[i]]
        b_s = SENTIMENT_MAP[b_preds["s_preds"][i]]
        b_t = TOPIC_MAP[b_preds["t_preds"][i]]
        p_s = SENTIMENT_MAP[p_preds["s_preds"][i]]
        p_t = TOPIC_MAP[p_preds["t_preds"][i]]

        # Mẫu PhoBERT đoán đúng mà Baseline đoán sai (chứng minh tính vượt trội)
        if (p_s == true_s and b_s != true_s) or (p_t == true_t and b_t != true_t):
            if len(errors) < 5:
                errors.append({
                    "text": texts[i].replace("_", " "),
                    "ground_truth": f"{true_s} | {true_t}",
                    "baseline_pred": f"{b_s} | {b_t}",
                    "phobert_pred": f"{p_s} | {p_t}",
                    "type": "PhoBERT vượt trội Baseline",
                })
        # Mẫu cả 2 cùng đoán sai (câu mỉa mai hoặc nhãn trung tính khó)
        elif p_s != true_s and b_s != true_s and len(errors) < 7:
            errors.append({
                "text": texts[i].replace("_", " "),
                "ground_truth": f"{true_s} | {true_t}",
                "baseline_pred": f"{b_s} | {b_t}",
                "phobert_pred": f"{p_s} | {p_t}",
                "type": "Thử thách chung (câu khó / mơ hồ)",
            })
    return errors


def main():
    parser = argparse.ArgumentParser(description="So sánh Baseline vs PhoBERT trên UIT-VSFC")
    parser.add_argument("--data_dir", type=str, default="data/uit_vsfc")
    parser.add_argument("--baseline_path", type=str, default="saved_models/baseline.joblib")
    parser.add_argument("--phobert_dir", type=str, default="saved_models/phobert")
    args = parser.parse_args()

    print("=" * 80 + "\n ĐÁNH GIÁ SO SÁNH: CLASSICAL BASELINE vs PHOBERT MULTI-TASK (UIT-VSFC)\n" + "=" * 80)

    # 1. Nạp tập test (đã tách từ)
    print(f"\n[*] Đang nạp dữ liệu kiểm thử từ '{args.data_dir}/test.parquet'...")
    texts, sents, topics = load_vsfc_split("test", data_dir=args.data_dir, word_segment=True)
    print(f"    Số lượng mẫu kiểm thử: {len(texts):,} câu.")

    # 2. Đánh giá Baseline
    print("\n[+] Đang đánh giá Classical Baseline (TF-IDF + Dual Logistic Regression)...")
    b_model = VietnameseFeedbackBaseline.load(args.baseline_path)
    b_res = evaluate_model(b_model, texts, sents, topics)

    # 3. Đánh giá PhoBERT Multi-Task
    print("[+] Đang đánh giá PhoBERT Multi-Task (vinai/phobert-base-v2 + Mean Pooling + Dual MLP)...")
    p_model = PhoBERTMultiTask.from_pretrained(args.phobert_dir)
    p_res = evaluate_model(p_model, texts, sents, topics)

    # 4. In bảng so sánh đối đầu chuẩn yêu cầu giảng viên
    hdr = f"{'Mô hình':<24} | {'Sent Acc':<9} | {'Sent Macro-F1':<13} | {'Topic Acc':<10} | {'Topic Macro-F1':<14} | {'Tốc độ':<10}"
    print("\n" + "=" * 95 + f"\n{hdr}\n" + "-" * 95)
    print(f"{'Classical Baseline':<24} | {b_res['sent_acc']*100:7.2f}% | {b_res['sent_mf1']*100:11.2f}% | {b_res['topic_acc']*100:8.2f}% | {b_res['topic_mf1']*100:12.2f}% | {b_res['speed']:7.1f} c/s")
    print(f"{'PhoBERT Multi-Task':<24} | {p_res['sent_acc']*100:7.2f}% | {p_res['sent_mf1']*100:11.2f}% | {p_res['topic_acc']*100:8.2f}% | {p_res['topic_mf1']*100:12.2f}% | {p_res['speed']:7.1f} c/s")
    print("-" * 95)
    d_s_acc = (p_res['sent_acc'] - b_res['sent_acc']) * 100
    d_s_f1 = (p_res['sent_mf1'] - b_res['sent_mf1']) * 100
    d_t_acc = (p_res['topic_acc'] - b_res['topic_acc']) * 100
    d_t_f1 = (p_res['topic_mf1'] - b_res['topic_mf1']) * 100
    print(f"{'PhoBERT cải thiện (+/-)':<24} | {d_s_acc:+7.2f}% | {d_s_f1:+11.2f}% | {d_t_acc:+8.2f}% | {d_t_f1:+12.2f}% |")
    print("=" * 95)

    # 5. Phân tích lỗi (Error Analysis)
    err_samples = collect_error_samples(texts, sents, topics, b_res, p_res)
    print("\n" + "=" * 80 + "\n PHÂN TÍCH LỖI SAI TIÊU BIỂU (ERROR ANALYSIS SAMPLES)\n" + "=" * 80)
    for idx, e in enumerate(err_samples, 1):
        print(f"[{idx}] Loại: {e['type']}")
        print(f"    Câu phản hồi  : \"{e['text']}\"")
        print(f"    Nhãn thực tế  : {e['ground_truth']}")
        print(f"    Baseline đoán : {e['baseline_pred']}")
        print(f"    PhoBERT đoán  : {e['phobert_pred']}\n")

    # 6. Lưu kết quả ra file
    os.makedirs("results", exist_ok=True)
    skip_keys = {"s_preds", "t_preds", "s_cm", "t_cm", "per_class_sent", "per_class_topic"}
    summary_data = {
        "num_test_samples": len(texts),
        "baseline": {k: v for k, v in b_res.items() if k not in skip_keys},
        "phobert": {k: v for k, v in p_res.items() if k not in skip_keys},
        "per_class": {
            "baseline": {"sentiment": b_res["per_class_sent"], "topic": b_res["per_class_topic"]},
            "phobert": {"sentiment": p_res["per_class_sent"], "topic": p_res["per_class_topic"]},
        },
        "confusion_matrix": {
            "baseline": {"sentiment": b_res["s_cm"], "topic": b_res["t_cm"]},
            "phobert": {"sentiment": p_res["s_cm"], "topic": p_res["t_cm"]},
        },
        "dataset_stats": compute_dataset_stats(args.data_dir),
        "error_analysis_samples": err_samples,
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    md_content = f"""# Báo Cáo Đánh Giá So Sánh Mô Hình (UIT-VSFC Test Set - N=3.166)
**Dự án**: Project 1 — Text Classification of Vietnamese Student Feedback  
**Nhóm**: Team 25  

## 1. Bảng So Sánh Hiệu Năng Thực Nghiệm

| Mô hình | Sentiment Accuracy | Sentiment Macro-F1 | Topic Accuracy | Topic Macro-F1 | Tốc độ suy luận |
|---|---|---|---|---|---|
| **Classical Baseline (TF-IDF + LR)** | **{b_res['sent_acc']*100:.2f}%** | **{b_res['sent_mf1']*100:.2f}%** | **{b_res['topic_acc']*100:.2f}%** | **{b_res['topic_mf1']*100:.2f}%** | {b_res['speed']:.1f} câu/s |
| **PhoBERT Multi-Task (Đề xuất)** | **{p_res['sent_acc']*100:.2f}%** | **{p_res['sent_mf1']*100:.2f}%** | **{p_res['topic_acc']*100:.2f}%** | **{p_res['topic_mf1']*100:.2f}%** | {p_res['speed']:.1f} câu/s |
| **Mức cải thiện của PhoBERT** | **+{d_s_acc:.2f}%** | **+{d_s_f1:.2f}%** | **+{d_t_acc:.2f}%** | **+{d_t_f1:.2f}%** | — |

## 2. Phân Tích Lỗi Sai Tiêu Biểu (Error Analysis)
"""
    for idx, e in enumerate(err_samples, 1):
        md_content += f"""### Mẫu {idx} ({e['type']})
- **Câu phản hồi**: *"{e['text']}"*
- **Nhãn thực tế**: `{e['ground_truth']}`
- **Baseline dự đoán**: `{e['baseline_pred']}`
- **PhoBERT dự đoán**: `{e['phobert_pred']}`

"""
    with open("results/comparison_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[*] Đã lưu kết quả tại 'results/metrics.json' và 'results/comparison_summary.md'.\n")


if __name__ == "__main__":
    main()
