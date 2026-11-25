"""
models/baseline_model.py
Mô hình Baseline cổ điển cho bài toán phân loại phản hồi sinh viên (UIT-VSFC):
Kiến trúc: TF-IDF (n-gram 1-2) + 2 bộ phân loại Logistic Regression độc lập (Cảm xúc & Chủ đề).
"""

import os
import sys
sys.dont_write_bytecode = True
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data.dataset import (
    SENTIMENT_LABELS, SENTIMENT_MAP, TOPIC_LABELS, TOPIC_MAP, segment_text,
)


def _calibrate_probs(probs: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Đảm bảo ma trận xác suất đủ n_classes cột tương ứng với các lớp nhãn."""
    if probs.shape[1] == n_classes:
        return probs
    full = np.zeros((probs.shape[0], n_classes), dtype=float)
    for idx, c in enumerate(classes):
        if 0 <= c < n_classes:
            full[:, c] = probs[:, idx]
    return full


class VietnameseFeedbackBaseline:
    """Mô hình Baseline cổ điển so sánh trong đồ án: TF-IDF + Dual Logistic Regression."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.9,
        max_features: int = 25000,
        sublinear_tf: bool = True,
        C: float = 3.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_df, max_df=max_df,
            max_features=max_features, sublinear_tf=sublinear_tf,
        )
        self.sentiment_clf = LogisticRegression(
            C=C, max_iter=max_iter, class_weight=class_weight, random_state=random_state
        )
        self.topic_clf = LogisticRegression(
            C=C, max_iter=max_iter, class_weight=class_weight, random_state=random_state
        )
        self.sentiment_labels = list(SENTIMENT_LABELS)
        self.topic_labels = list(TOPIC_LABELS)
        self.is_fitted = False

    def fit(self, texts: List[str], sentiments: List[int], topics: List[int], word_segment: bool = True):
        """Huấn luyện trích xuất đặc trưng TF-IDF và 2 bộ phân loại Logistic Regression."""
        processed = [segment_text(t) if word_segment else str(t or "") for t in texts]
        if len(texts) <= 5:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        X = self.vectorizer.fit_transform(processed)
        self.sentiment_clf.fit(X, sentiments)
        self.topic_clf.fit(X, topics)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: Union[str, List[str]], word_segment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán phân phối xác suất cho Cảm xúc (3 lớp) và Chủ đề (4 lớp)."""
        input_list = [texts] if isinstance(texts, str) else list(texts)
        if not input_list:
            return np.empty((0, 3)), np.empty((0, 4))

        processed = [segment_text(t) if word_segment else str(t or "") for t in input_list]
        X = self.vectorizer.transform(processed)
        s_probs = _calibrate_probs(self.sentiment_clf.predict_proba(X), self.sentiment_clf.classes_, 3)
        t_probs = _calibrate_probs(self.topic_clf.predict_proba(X), self.topic_clf.classes_, 4)
        return s_probs, t_probs

    def predict_labels(self, texts: Union[str, List[str]], word_segment: bool = True) -> Tuple[List[int], List[int]]:
        """Dự đoán nhãn ID cho cả 2 tác vụ."""
        sp, tp = self.predict_proba(texts, word_segment=word_segment)
        if len(sp) == 0:
            return [], []
        return np.argmax(sp, axis=1).astype(int).tolist(), np.argmax(tp, axis=1).astype(int).tolist()

    def predict(self, texts: Union[str, List[str]], word_segment: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Giao diện dự đoán chuẩn kèm nhãn, độ tin cậy và phân phối xác suất."""
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else list(texts)
        if not input_list:
            return []

        sp, tp = self.predict_proba(input_list, word_segment=word_segment)
        results = []
        for i in range(len(input_list)):
            sid, tid = int(np.argmax(sp[i])), int(np.argmax(tp[i]))
            results.append({
                "sentiment": {
                    "label": SENTIMENT_MAP[sid], "label_id": sid, "confidence": float(sp[i, sid]),
                    "distribution": {SENTIMENT_MAP[c]: float(sp[i, c]) for c in range(3)},
                },
                "topic": {
                    "label": TOPIC_MAP[tid], "label_id": tid, "confidence": float(tp[i, tid]),
                    "distribution": {TOPIC_MAP[c]: float(tp[i, c]) for c in range(4)},
                },
            })
        return results[0] if is_single else results

    def save(self, path: str = "saved_models/baseline.joblib") -> str:
        """Lưu toàn bộ mô hình và bộ trích xuất đặc trưng ra file joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str = "saved_models/baseline.joblib") -> "VietnameseFeedbackBaseline":
        """Nạp mô hình đã lưu từ file joblib."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại '{path}'.")
        return joblib.load(path)
