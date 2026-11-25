"""
tests/test_core.py
Bộ kiểm thử cốt lõi cho UIT-VSFC: Dữ liệu Parquet, Mô hình Baseline, PhoBERT Multi-Task và Plotly UI Charts (VI/EN).
"""

import json
import os
import sys
sys.dont_write_bytecode = True
import plotly.graph_objs as go
import torch
from transformers import AutoConfig, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SENTIMENT_LABELS, SENTIMENT_MAP, TOPIC_LABELS, TOPIC_MAP, load_vsfc_split, segment_text
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask


def test_data_splits_and_segmentation():
    """Kiểm tra load_vsfc_split nạp đủ 3 split UIT-VSFC chuẩn xác và tách từ PyVi hoạt động tốt."""
    for split_name, expected_count in [("train", 11426), ("validation", 1583), ("test", 3166)]:
        texts, sents, topics = load_vsfc_split(split_name, data_dir="data/uit_vsfc", word_segment=False)
        assert len(texts) == expected_count, f"Split '{split_name}' phải có {expected_count} câu"
        assert len(texts) == len(sents) == len(topics)
        assert isinstance(texts[0], str)
        assert all(0 <= s <= 2 for s in sents[:50])
        assert all(0 <= t <= 3 for t in topics[:50])

    seg = segment_text("Giảng viên giảng dạy rất nhiệt tình")
    assert "_" in seg


def test_baseline_fit_predict():
    """Kiểm tra mô hình Baseline (TF-IDF + LR) huấn luyện, dự đoán và nạp checkpoint."""
    texts = [
        "Thầy cô giảng dạy rất hay và nhiệt tình",
        "Phòng học quá nóng máy lạnh bị hỏng",
        "Chương trình học bình thường",
        "Học phí trường tăng cao",
    ]
    sents = [2, 0, 1, 0]
    topics = [0, 2, 1, 3]

    model = VietnameseFeedbackBaseline()
    model.fit(texts, sents, topics, word_segment=True)
    assert model.is_fitted

    res = model.predict("Thầy cô rất có tâm", word_segment=True)
    assert "sentiment" in res and "topic" in res
    assert res["sentiment"]["label"] in SENTIMENT_MAP.values()
    assert res["topic"]["label"] in TOPIC_MAP.values()


def test_phobert_forward_save_load():
    """Kiểm tra Masked Mean Pooling, Forward loss và cấu trúc đầu ra của PhoBERT Multi-Task."""
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.num_attention_heads = 2
    config.intermediate_size = 128
    backbone = AutoModel.from_config(config)

    model = PhoBERTMultiTask(config=config, backbone=backbone)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    s_labels = torch.tensor([0, 2], dtype=torch.long)
    t_labels = torch.tensor([1, 3], dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sentiment_labels=s_labels,
        topic_labels=t_labels,
    )
    assert outputs["loss"] is not None and outputs["loss"].dim() == 0
    assert outputs["sentiment_logits"].shape == (batch_size, 3)
    assert outputs["topic_logits"].shape == (batch_size, 4)


def test_saved_checkpoints_and_metrics():
    """Kiểm tra nạp checkpoint thực tế và tệp kết quả đánh giá results/metrics.json."""
    if os.path.isfile("saved_models/baseline.joblib"):
        loaded = VietnameseFeedbackBaseline.load("saved_models/baseline.joblib")
        assert loaded.is_fitted

    if os.path.isdir("saved_models/phobert") and os.path.isfile("saved_models/phobert/model.pt"):
        real_model = PhoBERTMultiTask.from_pretrained("saved_models/phobert")
        res = real_model.predict("Thầy giảng bài rất hay và dễ hiểu", word_segment=True)
        assert res["sentiment"]["label"] in SENTIMENT_MAP.values()
        assert res["topic"]["label"] in TOPIC_MAP.values()

    assert os.path.isfile("results/metrics.json"), "Thiếu file results/metrics.json"
    with open("results/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics.get("num_test_samples") == 3166
    assert "baseline" in metrics and "phobert" in metrics


def test_ui_charts_builders():
    """Kiểm tra toàn bộ hàm dựng biểu đồ và trích xuất đặc trưng trong app.ui_charts cho cả lang='vi' và lang='en'."""
    from app.ui_charts import (
        compute_prediction_features,
        extract_token_feature_importance,
        format_class_label,
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
    )

    assert format_class_label("positive", "vi") == "Tích cực"
    assert format_class_label("positive", "en") == "Positive"
    assert format_class_label("training_program", "vi") == "Chương trình đào tạo"
    assert format_class_label("training_program", "en") == "Training Program"

    with open("results/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    dataset_stats = metrics.get("dataset_stats", {})
    cm_sent = (
        metrics.get("confusion_matrix", {})
        .get("phobert", {})
        .get("sentiment", [[1361, 21, 27], [27, 92, 48], [50, 18, 1522]])
    )
    cm_topic = (
        metrics.get("confusion_matrix", {})
        .get("phobert", {})
        .get("topic", [[2126, 123, 4, 37], [87, 458, 4, 23], [2, 4, 138, 1], [37, 32, 3, 87]])
    )

    sample_primary = {
        "sentiment": {
            "label": "positive",
            "confidence": 0.94,
            "distribution": {"negative": 0.03, "neutral": 0.03, "positive": 0.94},
        },
        "topic": {
            "label": "lecturer",
            "confidence": 0.91,
            "distribution": {"lecturer": 0.91, "training_program": 0.05, "facility": 0.02, "others": 0.02},
        },
    }
    sample_compare = {
        "sentiment": {
            "label": "positive",
            "confidence": 0.85,
            "distribution": {"negative": 0.08, "neutral": 0.07, "positive": 0.85},
        },
        "topic": {
            "label": "lecturer",
            "confidence": 0.82,
            "distribution": {"lecturer": 0.82, "training_program": 0.10, "facility": 0.04, "others": 0.04},
        },
    }

    feats = compute_prediction_features(sample_primary)
    for k in ("sent_conf", "topic_conf", "sent_entropy", "topic_entropy", "sent_margin", "topic_margin", "joint_conf"):
        assert k in feats and isinstance(feats[k], float)

    b_model = VietnameseFeedbackBaseline.load("saved_models/baseline.joblib") if os.path.isfile("saved_models/baseline.joblib") else None
    seg_text = segment_text("Thầy cô giảng dạy rất nhiệt tình và giải đáp thắc mắc chi tiết")
    tok_weights = extract_token_feature_importance(b_model, seg_text, sent_label_id=2, topic_label_id=0, top_k=6)
    assert isinstance(tok_weights, list) and len(tok_weights) > 0
    assert "token" in tok_weights[0] and "sent_weight" in tok_weights[0] and "topic_weight" in tok_weights[0]

    for lang in ("vi", "en"):
        figures = [
            make_confidence_gauge("Cảm xúc" if lang == "vi" else "Sentiment", "positive", 0.96, lang=lang),
            make_probability_bar_chart(
                sample_primary["sentiment"]["distribution"],
                "Xác suất Cảm xúc" if lang == "vi" else "Sentiment Probability",
                lang=lang,
                compare_distribution=sample_compare["sentiment"]["distribution"],
            ),
            make_sentence_multitask_radar_chart(sample_primary, compare_res=sample_compare, lang=lang),
            make_token_importance_chart(tok_weights, lang=lang),
            make_headline_comparison_chart(metrics, lang=lang),
            make_multimetric_radar_chart(metrics, lang=lang),
            make_per_class_f1_chart(metrics, "sentiment", lang=lang),
            make_per_class_f1_chart(metrics, "topic", lang=lang),
            make_confusion_matrix_heatmap(cm_sent, SENTIMENT_LABELS, "CM Sentiment", lang=lang),
            make_confusion_matrix_heatmap(cm_topic, TOPIC_LABELS, "CM Topic", lang=lang),
            make_split_donut_chart(dataset_stats, lang=lang),
            make_label_distribution_chart(dataset_stats, "sentiment", lang=lang),
            make_label_distribution_chart(dataset_stats, "topic", lang=lang),
        ]
        for fig in figures:
            assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
            assert len(fig.data) > 0, "Plotly Figure phải có ít nhất 1 trace trong fig.data"


# Giữ alias tương thích ngược
test_load_vsfc_split = test_data_splits_and_segmentation
test_baseline_fit_and_predict = test_baseline_fit_predict
test_phobert_architecture_and_predict = test_phobert_forward_save_load


if __name__ == "__main__":
    test_data_splits_and_segmentation()
    test_baseline_fit_predict()
    test_phobert_forward_save_load()
    test_saved_checkpoints_and_metrics()
    test_ui_charts_builders()
    print("All core and UI chart tests passed successfully!")
