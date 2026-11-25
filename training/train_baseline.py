"""
training/train_baseline.py
Huấn luyện mô hình Baseline cổ điển (TF-IDF + Logistic Regression) trên 11.426 câu UIT-VSFC.
"""

import argparse
import os
import sys
sys.dont_write_bytecode = True
import time
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import load_vsfc_split
from models.baseline_model import VietnameseFeedbackBaseline


def evaluate_split(model, texts, sents, topics):
    """Tính toán Accuracy, Weighted-F1 và Macro-F1 cho 2 tác vụ Sentiment và Topic."""
    sent_preds, topic_preds = model.predict_labels(texts, word_segment=False)
    return {
        "sent_acc": accuracy_score(sents, sent_preds),
        "sent_wf1": f1_score(sents, sent_preds, average="weighted", zero_division=0),
        "sent_mf1": f1_score(sents, sent_preds, average="macro", zero_division=0),
        "topic_acc": accuracy_score(topics, topic_preds),
        "topic_wf1": f1_score(topics, topic_preds, average="weighted", zero_division=0),
        "topic_mf1": f1_score(topics, topic_preds, average="macro", zero_division=0),
    }


def train_baseline(data_dir: str = "data/uit_vsfc", save_path: str = "saved_models/baseline.joblib", **kwargs):
    """Huấn luyện mô hình Baseline và đánh giá trên validation/test."""
    print("=" * 72 + "\n HUẤN LUYỆN BASELINE CỔ ĐIỂN (TF-IDF + DUAL LOGISTIC REGRESSION)\n" + "=" * 72)

    # 1. Tải tập dữ liệu
    print(f"\n[1/3] Đang tải dữ liệu từ '{data_dir}'...")
    train_texts, train_sents, train_topics = load_vsfc_split("train", data_dir=data_dir, word_segment=True)
    val_texts, val_sents, val_topics = load_vsfc_split("validation", data_dir=data_dir, word_segment=True)
    test_texts, test_sents, test_topics = load_vsfc_split("test", data_dir=data_dir, word_segment=True)
    print(f"      Số lượng mẫu -> Train: {len(train_texts):,}, Val: {len(val_texts):,}, Test: {len(test_texts):,}")

    # 2. Khởi tạo và khớp mô hình
    print("\n[2/3] Huấn luyện TF-IDF vectorizer và 2 bộ phân loại Logistic Regression...")
    t0 = time.time()
    model = VietnameseFeedbackBaseline(
        max_features=kwargs.get("max_features", 25000),
        C=kwargs.get("C", 3.0),
        max_iter=kwargs.get("max_iter", 1000),
    )
    model.fit(train_texts, train_sents, train_topics, word_segment=False)
    print(f"      Huấn luyện hoàn tất trong {time.time() - t0:.2f} giây.")

    # 3. Đánh giá kết quả trên tập val và test
    print("\n[3/3] Kết quả đánh giá hiệu năng (Độ đo Accuracy & Macro-F1 per task):")
    header = f"  {'Tập dữ liệu':<14} {'Nhiệm vụ':<14} {'Accuracy':<12} {'Weighted-F1':<14} {'Macro-F1':<12}"
    print("-" * 72 + f"\n{header}\n" + "-" * 72)

    splits = [("Validation", val_texts, val_sents, val_topics), ("Test", test_texts, test_sents, test_topics)]
    metrics = {}
    for name, texts, sents, topics in splits:
        res = evaluate_split(model, texts, sents, topics)
        metrics[name.lower()] = res
        print(f"  {name:<14} {'Sentiment':<14} {res['sent_acc']*100:6.2f}%      {res['sent_wf1']*100:6.2f}%        {res['sent_mf1']*100:6.2f}%")
        print(f"  {'':<14} {'Topic':<14} {res['topic_acc']*100:6.2f}%      {res['topic_wf1']*100:6.2f}%        {res['topic_mf1']*100:6.2f}%\n" + "-" * 72)

    # 4. Lưu artifact
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    print(f"\n[*] Đã lưu mô hình Baseline thành công tại '{save_path}'.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline trên UIT-VSFC")
    parser.add_argument("--data_dir", type=str, default="data/uit_vsfc")
    parser.add_argument("--save_path", type=str, default="saved_models/baseline.joblib")
    parser.add_argument("--max_features", type=int, default=25000)
    parser.add_argument("--C", type=float, default=3.0)
    args = parser.parse_args()

    train_baseline(data_dir=args.data_dir, save_path=args.save_path, max_features=args.max_features, C=args.C)


if __name__ == "__main__":
    main()
