"""
tests/test_core.py
Bộ kiểm thử cốt lõi cho UIT-VSFC: Dữ liệu Parquet, Mô hình Baseline và PhoBERT Multi-Task.
"""

import os
import shutil
import sys
sys.dont_write_bytecode = True
import tempfile
import torch
from transformers import AutoConfig, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SENTIMENT_MAP, TOPIC_MAP, load_vsfc_split
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask


def test_load_vsfc_split():
    """Kiểm tra load_vsfc_split nạp đủ 3 split UIT-VSFC chuẩn xác."""
    for split_name, expected_count in [("train", 11426), ("validation", 1583), ("test", 3166)]:
        texts, sents, topics = load_vsfc_split(split_name, data_dir="data/uit_vsfc", word_segment=False)
        assert len(texts) == expected_count, f"Split '{split_name}' phải có {expected_count} câu"
        assert len(texts) == len(sents) == len(topics)
        assert isinstance(texts[0], str)
        assert all(0 <= s <= 2 for s in sents[:50])
        assert all(0 <= t <= 3 for t in topics[:50])


def test_baseline_fit_and_predict():
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

    # Kiểm tra nạp checkpoint thật
    if os.path.isfile("saved_models/baseline.joblib"):
        loaded = VietnameseFeedbackBaseline.load("saved_models/baseline.joblib")
        assert loaded.is_fitted


def test_phobert_architecture_and_predict():
    """Kiểm tra Masked Mean Pooling, Forward loss, Save/Load và dự đoán của PhoBERT Multi-Task."""
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

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, sentiment_labels=s_labels, topic_labels=t_labels)
    assert outputs["loss"] is not None and outputs["loss"].dim() == 0
    assert outputs["sentiment_logits"].shape == (batch_size, 3)
    assert outputs["topic_logits"].shape == (batch_size, 4)

    # Kiểm tra checkpoint thật nếu có
    if os.path.isdir("saved_models/phobert") and os.path.isfile("saved_models/phobert/model.pt"):
        real_model = PhoBERTMultiTask.from_pretrained("saved_models/phobert")
        res = real_model.predict("Thầy giảng bài rất hay và dễ hiểu", word_segment=True)
        assert res["sentiment"]["label"] in SENTIMENT_MAP.values()
        assert res["topic"]["label"] in TOPIC_MAP.values()


if __name__ == "__main__":
    test_load_vsfc_split()
    test_baseline_fit_and_predict()
    test_phobert_architecture_and_predict()
    print("All core tests passed successfully!")
