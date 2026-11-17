"""
models/hf_external_models.py
Tích hợp các mô hình Hugging Face bên ngoài (Pretrained & Zero-Shot):
1. zeroday117/phobert-vsfc-sentiment3 (PhoBERT v2 - Sentiment 3 lớp)
2. thnhan3/phobert-vietnamese-students-feedback (PhoBERT v1 - Sentiment 3 lớp)
3. BaoNhan/cafebert-UITVSFC-T (CafeBERT - Topic 4 lớp)
4. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (DeBERTa-v3 - Zero-Shot English)
"""

from typing import Dict, Any, Optional
import torch
from transformers import pipeline


class VietnamesePipelinePredictor:
    """
    Kết hợp 2 mô hình đơn nhiệm Hugging Face cho tiếng Việt:
    - Sentiment: zeroday117/phobert-vsfc-sentiment3
    - Topic: BaoNhan/cafebert-UITVSFC-T
    """

    def __init__(
        self,
        sentiment_model_id: str = "zeroday117/phobert-vsfc-sentiment3",
        topic_model_id: str = "BaoNhan/cafebert-UITVSFC-T",
        device: int = -1,
    ):
        self.device = device
        self.sentiment_model_id = sentiment_model_id
        self.topic_model_id = topic_model_id

        self.sentiment_pipe = pipeline(
            "text-classification",
            model=sentiment_model_id,
            device=device,
        )
        self.topic_pipe = pipeline(
            "text-classification",
            model=topic_model_id,
            device=device,
        )

    def predict(self, text: str) -> Dict[str, Any]:
        s_out = self.sentiment_pipe(text)[0]
        t_out = self.topic_pipe(text)[0]
        return {
            "sentiment": s_out["label"],
            "sentiment_score": float(s_out["score"]),
            "topic": t_out["label"],
            "topic_score": float(t_out["score"]),
            "model_type": "pipeline_external_hf",
        }


class EnglishZeroShotPredictor:
    """
    Sử dụng DeBERTa-v3-large NLI để phân loại Zero-Shot cho tiếng Anh
    cả Cảm xúc (3 lớp) và Chủ đề (4 lớp).
    """

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: int = -1,
    ):
        self.model_id = model_id
        self.device = device
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=device,
        )
        self.sentiment_labels = ["positive", "neutral", "negative"]
        self.topic_labels = ["lecturer", "training program", "facility", "others"]
        self.topic_map = {
            "lecturer": "lecturer",
            "training program": "training_program",
            "facility": "facility",
            "others": "others",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        s_res = self.classifier(
            text,
            candidate_labels=self.sentiment_labels,
            hypothesis_template="The student sentiment of this feedback is {}.",
        )
        t_res = self.classifier(
            text,
            candidate_labels=self.topic_labels,
            hypothesis_template="This feedback discusses {}.",
        )

        best_topic_raw = t_res["labels"][0]
        mapped_topic = self.topic_map.get(best_topic_raw, "others")

        return {
            "sentiment": s_res["labels"][0],
            "sentiment_score": float(s_res["scores"][0]),
            "topic": mapped_topic,
            "topic_score": float(t_res["scores"][0]),
            "model_type": "zero_shot_deberta_large",
        }
