"""
data/dataset.py
Đọc dữ liệu Parquet UIT-VSFC, tách từ tiếng Việt qua PyVi và tạo PyTorch DataLoader.
"""

import os
import sys
sys.dont_write_bytecode = True
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 2 tác vụ theo đúng đặc tả UIT-VSFC: Cảm xúc (3 lớp) & Chủ đề (4 lớp)
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TOPIC_LABELS = ["lecturer", "training_program", "facility", "others"]

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

SENTIMENT_TO_ID = {v: k for k, v in SENTIMENT_MAP.items()}
TOPIC_TO_ID = {v: k for k, v in TOPIC_MAP.items()}


def segment_text(text: str) -> str:
    """Tách từ tiếng Việt chuẩn bằng PyVi (nối từ ghép bằng dấu gạch dưới '_')."""
    if not text:
        return ""
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(str(text).strip())


def load_vsfc_split(
    split_name: str,
    data_dir: str = "data/uit_vsfc",
    word_segment: bool = True,
) -> Tuple[List[str], List[int], List[int]]:
    """Đọc dữ liệu 1 split (train, validation, test) từ file Parquet."""
    norm = "validation" if split_name in ("val", "valid", "dev") else split_name
    parquet_path = os.path.join(data_dir, f"{norm}.parquet")

    if not os.path.isfile(parquet_path):
        from data.download_data import download_vsfc_dataset
        download_vsfc_dataset(output_dir=data_dir)

    df = pd.read_parquet(parquet_path)
    text_col = "sentence" if "sentence" in df.columns else "text"
    texts = df[text_col].astype(str).tolist()
    sentiments = df["sentiment"].astype(int).tolist()
    topics = df["topic"].astype(int).tolist()

    if word_segment:
        texts = [segment_text(t) for t in texts]
    return texts, sentiments, topics


class VSFCDataset(Dataset):
    """PyTorch Dataset đóng gói câu và nhãn nhị vụ Cảm xúc + Chủ đề."""

    def __init__(
        self,
        texts: List[str],
        sentiments: Optional[List[int]] = None,
        topics: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 64,
    ):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.sentiment_labels = (
            torch.tensor(sentiments, dtype=torch.long) if sentiments is not None else None
        )
        self.topic_labels = (
            torch.tensor(topics, dtype=torch.long) if topics is not None else None
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.sentiment_labels is not None:
            item["sentiment_labels"] = self.sentiment_labels[idx]
        if self.topic_labels is not None:
            item["topic_labels"] = self.topic_labels[idx]
        return item


def get_dataloaders(
    tokenizer,
    batch_size: int = 32,
    max_length: int = 64,
    data_dir: str = "data/uit_vsfc",
    word_segment: bool = True,
    quick_mode: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Khởi tạo DataLoader cho 3 tập train, validation, test."""
    tr_x, tr_s, tr_t = load_vsfc_split("train", data_dir=data_dir, word_segment=word_segment)
    va_x, va_s, va_t = load_vsfc_split("validation", data_dir=data_dir, word_segment=word_segment)
    te_x, te_s, te_t = load_vsfc_split("test", data_dir=data_dir, word_segment=word_segment)

    if quick_mode:
        tr_x, tr_s, tr_t = tr_x[:128], tr_s[:128], tr_t[:128]
        va_x, va_s, va_t = va_x[:64], va_s[:64], va_t[:64]
        te_x, te_s, te_t = te_x[:64], te_s[:64], te_t[:64]

    train_ds = VSFCDataset(tr_x, tr_s, tr_t, tokenizer, max_length)
    val_ds = VSFCDataset(va_x, va_s, va_t, tokenizer, max_length)
    test_ds = VSFCDataset(te_x, te_s, te_t, tokenizer, max_length)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
