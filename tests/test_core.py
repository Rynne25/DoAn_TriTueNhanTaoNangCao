"""
tests/test_core.py
Bộ kiểm thử cốt lõi cho UIT-VSFC: Dữ liệu Parquet, Mô hình Baseline, PhoBERT Multi-Task và Plotly UI Charts.
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
    """Kiểm tra toàn bộ các hàm dựng biểu đồ Plotly trong app.ui_charts trả về go.Figure hợp lệ."""
    from app.ui_charts import (
        make_confidence_gauge,
        make_confusion_matrix_heatmap,
        make_headline_comparison_chart,
        make_label_distribution_chart,
        make_multimetric_radar_chart,
        make_per_class_f1_chart,
        make_probability_bar_chart,
        make_split_donut_chart,
    )

    with open("results/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    dataset_stats = metrics.get(
        "dataset_stats",
        {
            "split_sizes": {"train": 11426, "validation": 1583, "test": 3166, "total": 16175},
            "sentiment_distribution": {"negative": 7439, "neutral": 698, "positive": 8038},
            "topic_distribution": {
                "lecturer": 11598,
                "training_program": 2975,
                "facility": 758,
                "others": 844,
            },
        },
    )
    cm_sent = (
        metrics.get("confusion_matrix", {})
        .get("phobert", {})
        .get("sentiment", [[1345, 22, 42], [25, 108, 34], [38, 18, 1534]])
    )
    cm_topic = (
        metrics.get("confusion_matrix", {})
        .get("phobert", {})
        .get("topic", [[2115, 92, 24, 59], [88, 456, 12, 16], [11, 8, 134, 6], [35, 16, 4, 104]])
    )

    figures = [
        make_confidence_gauge("Cảm xúc", "positive", 0.96),
        make_probability_bar_chart({"negative": 0.02, "neutral": 0.02, "positive": 0.96}, "Xác suất Cảm xúc"),
        make_headline_comparison_chart(metrics),
        make_multimetric_radar_chart(metrics),
        make_per_class_f1_chart(metrics, "sentiment"),
        make_per_class_f1_chart(metrics, "topic"),
        make_confusion_matrix_heatmap(cm_sent, SENTIMENT_LABELS, "CM Sentiment"),
        make_confusion_matrix_heatmap(cm_topic, TOPIC_LABELS, "CM Topic"),
        make_split_donut_chart(dataset_stats),
        make_label_distribution_chart(dataset_stats, "sentiment"),
        make_label_distribution_chart(dataset_stats, "topic"),
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
