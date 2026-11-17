"""
data/dataset.py
UIT-VSFC Data Loading, PyVi Vietnamese Word Segmentation, and PyTorch Dataset / DataLoader helpers.
"""

import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _clean_text(s: Optional[str]) -> str:
    """Sanitizes text by converting None to empty string and collapsing duplicate whitespace."""
    if s is None:
        return ""
    text = str(s).strip()
    return re.sub(r"\s+", " ", text)


def _resolve_language(language: Optional[str], data_dir: str) -> str:
    """Validates and resolves dataset language ('vi' or 'en')."""
    if language is not None:
        norm = str(language).strip().lower()
        if norm not in {"vi", "en"}:
            raise ValueError(f"Unsupported language '{language}'. Must be 'vi' or 'en'.")
        return norm
    base = os.path.basename(os.path.normpath(data_dir)).lower()
    if "english" in base:
        return "en"
    return "vi"


def _resolve_word_segment(
    word_segment: Optional[bool], language: Optional[str], data_dir: str
) -> Tuple[bool, str]:
    """Resolves effective (word_segment, language) honoring explicit word_segment overrides."""
    resolved_lang = _resolve_language(language, data_dir)
    if word_segment is not None:
        return bool(word_segment), resolved_lang
    return (resolved_lang == "vi"), resolved_lang


def _parse_int_label(val: Any, col_name: str, idx: int, file_path: str) -> int:
    """Parses a label cell into a strict integer with row-index and column diagnostics."""
    if val is None or isinstance(val, bool):
        raise ValueError(
            f"Invalid {col_name} label at index {idx} in {file_path}: {val!r}. Expected integer."
        )
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val) or not val.is_integer():
            raise ValueError(
                f"Invalid {col_name} label at index {idx} in {file_path}: {val!r}. Expected integer."
            )
        return int(val)
    if isinstance(val, int):
        return val
    try:
        text = str(val).strip()
        if not text:
            raise ValueError("empty")
        return int(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {col_name} label at index {idx} in {file_path}: {val!r}. Expected integer."
        ) from exc


# Official UIT-VSFC Label Mappings
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TOPIC_LABELS = ["lecturer", "training_program", "facility", "others"]

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

SENTIMENT_TO_ID = {v: k for k, v in SENTIMENT_MAP.items()}
TOPIC_TO_ID = {v: k for k, v in TOPIC_MAP.items()}

# Valid split names
VALID_SPLITS = {"train", "validation", "test"}
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "training": "train",
    "testing": "test",
}


def _read_table_file(file_path: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Reads a .parquet or .csv file using Polars, Pandas, or standard library csv.
    Returns: (sentences, sentiments, topics)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".parquet":
        # 1. Try Polars
        try:
            import polars as pl
            df = pl.read_parquet(file_path)
            cols = set(df.columns)
            text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
            if not text_col or "sentiment" not in cols or "topic" not in cols:
                raise ValueError(
                    f"Parquet file {file_path} missing required columns ('sentence', 'sentiment', 'topic'). Found: {df.columns}"
                )
            sentences = [_clean_text(s) for s in df[text_col].to_list()]
            sentiments = [_parse_int_label(s, "sentiment", idx, file_path) for idx, s in enumerate(df["sentiment"].to_list())]
            topics = [_parse_int_label(t, "topic", idx, file_path) for idx, t in enumerate(df["topic"].to_list())]
            return sentences, sentiments, topics
        except ImportError:
            pass

        # 2. Try Pandas
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            cols = set(df.columns)
            text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
            if not text_col or "sentiment" not in cols or "topic" not in cols:
                raise ValueError(
                    f"Parquet file {file_path} missing required columns ('sentence', 'sentiment', 'topic'). Found: {list(df.columns)}"
                )
            sentences = [_clean_text(s) for s in df[text_col].tolist()]
            sentiments = [_parse_int_label(s, "sentiment", idx, file_path) for idx, s in enumerate(df["sentiment"].tolist())]
            topics = [_parse_int_label(t, "topic", idx, file_path) for idx, t in enumerate(df["topic"].tolist())]
            return sentences, sentiments, topics
        except ImportError:
            raise ImportError(
                f"Cannot read parquet file '{file_path}'. Neither 'polars' nor 'pandas' with pyarrow is installed. "
                "Please install polars (python -m pip install polars) or pyarrow (python -m pip install pyarrow)."
            )

    elif ext == ".csv":
        # 1. Try Polars
        try:
            import polars as pl
            df = pl.read_csv(file_path)
            cols = set(df.columns)
            text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
            if not text_col or "sentiment" not in cols or "topic" not in cols:
                raise ValueError(
                    f"CSV file {file_path} missing required columns ('sentence', 'sentiment', 'topic'). Found: {df.columns}"
                )
            sentences = [_clean_text(s) for s in df[text_col].to_list()]
            sentiments = [_parse_int_label(s, "sentiment", idx, file_path) for idx, s in enumerate(df["sentiment"].to_list())]
            topics = [_parse_int_label(t, "topic", idx, file_path) for idx, t in enumerate(df["topic"].to_list())]
            return sentences, sentiments, topics
        except ImportError:
            pass

        # 2. Try Pandas
        try:
            import pandas as pd
            df = pd.read_csv(file_path, encoding="utf-8")
            cols = set(df.columns)
            text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
            if not text_col or "sentiment" not in cols or "topic" not in cols:
                raise ValueError(
                    f"CSV file {file_path} missing required columns ('sentence', 'sentiment', 'topic'). Found: {list(df.columns)}"
                )
            sentences = [_clean_text(s) for s in df[text_col].tolist()]
            sentiments = [_parse_int_label(s, "sentiment", idx, file_path) for idx, s in enumerate(df["sentiment"].tolist())]
            topics = [_parse_int_label(t, "topic", idx, file_path) for idx, t in enumerate(df["topic"].tolist())]
            return sentences, sentiments, topics
        except ImportError:
            pass

        # 3. Fallback to standard library csv module
        import csv
        sentences, sentiments, topics = [], [], []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            text_col = "sentence" if "sentence" in fieldnames else ("text" if "text" in fieldnames else None)
            if not text_col or "sentiment" not in fieldnames or "topic" not in fieldnames:
                raise ValueError(
                    f"CSV file {file_path} missing required columns ('sentence', 'sentiment', 'topic'). Found: {reader.fieldnames}"
                )
            for idx, row in enumerate(reader):
                sentences.append(_clean_text(row[text_col]))
                sentiments.append(_parse_int_label(row["sentiment"], "sentiment", idx, file_path))
                topics.append(_parse_int_label(row["topic"], "topic", idx, file_path))
        return sentences, sentiments, topics

    else:
        raise ValueError(f"Unsupported file format '{ext}' for file '{file_path}'. Must be .parquet or .csv")


def load_vsfc_split(
    split_name: str,
    data_dir: str = "data/uit_vsfc",
    word_segment: Optional[bool] = None,
    auto_download: bool = True,
    language: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], List[int], List[int]]:
    """
    Loads a student feedback dataset split ('train', 'validation', 'test').

    Args:
        split_name: Name of the split ('train', 'validation'/'val', 'test').
        data_dir: Directory containing the cached parquet or csv files.
                  (Alias 'cache_dir' is also accepted for API compatibility).
        word_segment: If True, applies PyVi Vietnamese word segmentation (e.g. 'giáo_viên').
                      If None, defaults to True for Vietnamese ('vi') and False for English ('en').
        auto_download: If True and Vietnamese dataset files are missing, attempts to download
                       them via data.download_data.download_vsfc_dataset.
        language: Optional language code ('vi' or 'en'). Auto-inferred from data_dir if None.
        **kwargs: Handles backwards compatibility aliases like 'cache_dir'.

    Returns:
        Tuple of:
            texts (List[str]): feedback sentences (word-segmented if resolved word_segment is True).
            sentiments (List[int]): sentiment labels in {0: negative, 1: neutral, 2: positive}.
            topics (List[int]): topic labels in {0: lecturer, 1: training_program, 2: facility, 3: others}.

    Raises:
        ValueError: If split_name/language is invalid, sentences are empty, or labels are out of bounds.
        FileNotFoundError: If the split file is not found in data_dir.
        ImportError: If word_segment=True but pyvi is not installed.
    """
    # Support cache_dir alias from PROJECT.md
    if "cache_dir" in kwargs:
        data_dir = kwargs["cache_dir"]

    resolved_ws, resolved_lang = _resolve_word_segment(word_segment, language, data_dir)

    # Optional PROJECT_ROOT fallback for default relative data directories when invoked outside repo root
    norm_rel = os.path.normpath(data_dir)
    if norm_rel in (os.path.normpath("data/uit_vsfc"), os.path.normpath("data/english_feedback")):
        if not os.path.isdir(data_dir):
            candidate_root_dir = os.path.join(PROJECT_ROOT, norm_rel)
            if os.path.isdir(candidate_root_dir):
                data_dir = candidate_root_dir

    # Normalize split name
    norm_split = split_name.strip().lower()
    norm_split = SPLIT_ALIASES.get(norm_split, norm_split)

    if norm_split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split_name: '{split_name}'. Must be one of: {sorted(VALID_SPLITS)} "
            f"(or aliases {list(SPLIT_ALIASES.keys())})"
        )

    # Candidate file paths
    parquet_path = os.path.join(data_dir, f"{norm_split}.parquet")
    csv_path = os.path.join(data_dir, f"{norm_split}.csv")

    target_file = None
    download_err: Optional[Exception] = None
    if os.path.isfile(parquet_path):
        target_file = parquet_path
    elif os.path.isfile(csv_path):
        target_file = csv_path
    elif auto_download and resolved_lang == "vi":
        # Try importing download_data and running it for Vietnamese UIT-VSFC
        try:
            from data.download_data import download_vsfc_dataset
            print(f"[dataset.py] Split '{norm_split}' not found locally. Auto-downloading to '{data_dir}'...")
            download_vsfc_dataset(output_dir=data_dir)
            if os.path.isfile(parquet_path):
                target_file = parquet_path
            elif os.path.isfile(csv_path):
                target_file = csv_path
        except Exception as err:
            download_err = err

    if target_file is None:
        msg = (
            f"Dataset file for split '{split_name}' (normalized: '{norm_split}') not found in '{data_dir}'. "
            f"Looked for '{parquet_path}' and '{csv_path}'."
        )
        if resolved_lang == "vi":
            msg += " Please run 'python data/download_data.py' first."
        else:
            msg += " Please prepare English dataset splits first."
        if download_err is not None:
            msg += f" Auto-download error ({type(download_err).__name__}): {download_err}"
            raise FileNotFoundError(msg) from download_err
        raise FileNotFoundError(msg)

    # Read data from file
    texts, sentiments, topics = _read_table_file(target_file)

    # Validate non-empty cleaned sentences
    for idx, text in enumerate(texts):
        if not text:
            raise ValueError(f"Empty or null sentence at index {idx} in {target_file}.")

    # Validate label ranges
    # Sentiments must be in [0, 2]
    # Topics must be in [0, 3]
    for idx, (s, t) in enumerate(zip(sentiments, topics)):
        if s not in (0, 1, 2):
            raise ValueError(
                f"Invalid sentiment label at index {idx} in {target_file}: {s}. "
                f"Expected integer in [0, 2] ({SENTIMENT_MAP})."
            )
        if t not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid topic label at index {idx} in {target_file}: {t}. "
                f"Expected integer in [0, 3] ({TOPIC_MAP})."
            )

    # Apply PyVi word segmentation if requested
    if resolved_ws:
        try:
            from pyvi import ViTokenizer
        except ImportError:
            raise ImportError(
                "The 'pyvi' package is required for Vietnamese word segmentation. "
                "Please install it using: python -m pip install pyvi"
            )
        texts = [ViTokenizer.tokenize(t) for t in texts]

    return texts, sentiments, topics


class VSFCDataset(Dataset):
    """
    PyTorch Dataset for UIT-VSFC Multi-Task Classification with PhoBERT.
    Holds tokenized input_ids, attention_mask, and optional sentiment & topic labels.

    Supports initialization via:
    1. Pre-tokenized tensors/lists: `VSFCDataset(input_ids, attention_mask, sentiment_labels, topic_labels)`
    2. Texts + tokenizer: `VSFCDataset(texts=texts, sentiments=sents, topics=topics, tokenizer=tokenizer, max_length=128)`
    """
    def __init__(
        self,
        input_ids: Optional[Union[torch.Tensor, List[List[int]]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
        sentiment_labels: Optional[Union[torch.Tensor, List[int]]] = None,
        topic_labels: Optional[Union[torch.Tensor, List[int]]] = None,
        texts: Optional[List[str]] = None,
        sentiments: Optional[List[int]] = None,
        topics: Optional[List[int]] = None,
        tokenizer = None,
        max_length: int = 128,
    ):
        # Handle texts + tokenizer construction if provided
        if texts is not None and tokenizer is not None:
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if sentiments is not None and sentiment_labels is None:
                sentiment_labels = sentiments
            if topics is not None and topic_labels is None:
                topic_labels = topics

        if input_ids is None or attention_mask is None:
            raise ValueError("VSFCDataset requires either (input_ids, attention_mask) or (texts, tokenizer).")

        if not isinstance(input_ids, torch.Tensor):
            self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            self.input_ids = input_ids.to(dtype=torch.long)

        if not isinstance(attention_mask, torch.Tensor):
            self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            self.attention_mask = attention_mask.to(dtype=torch.long)

        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError(
                f"Mismatched lengths: input_ids ({len(self.input_ids)}) vs "
                f"attention_mask ({len(self.attention_mask)})"
            )

        if sentiment_labels is not None:
            if not isinstance(sentiment_labels, torch.Tensor):
                self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)
            else:
                self.sentiment_labels = sentiment_labels.to(dtype=torch.long)
            if len(self.sentiment_labels) != len(self.input_ids):
                raise ValueError(
                    f"Mismatched lengths: sentiment_labels ({len(self.sentiment_labels)}) vs "
                    f"input_ids ({len(self.input_ids)})"
                )
        else:
            self.sentiment_labels = None

        if topic_labels is not None:
            if not isinstance(topic_labels, torch.Tensor):
                self.topic_labels = torch.tensor(topic_labels, dtype=torch.long)
            else:
                self.topic_labels = topic_labels.to(dtype=torch.long)
            if len(self.topic_labels) != len(self.input_ids):
                raise ValueError(
                    f"Mismatched lengths: topic_labels ({len(self.topic_labels)}) vs "
                    f"input_ids ({len(self.input_ids)})"
                )
        else:
            self.topic_labels = None

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

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        sentiments: Optional[List[int]] = None,
        topics: Optional[List[int]] = None,
        tokenizer = None,
        max_length: int = 128,
    ) -> "VSFCDataset":
        """
        Creates a VSFCDataset directly from raw/segmented text strings and a Hugging Face tokenizer.
        """
        if tokenizer is None:
            raise ValueError("A Hugging Face tokenizer must be provided to VSFCDataset.from_texts()")

        return cls(
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            tokenizer=tokenizer,
            max_length=max_length,
        )


def create_dataloader(
    texts: List[str],
    sentiments: Optional[List[int]] = None,
    topics: Optional[List[int]] = None,
    tokenizer = None,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a single PyTorch DataLoader for a given set of texts and labels.
    """
    dataset = VSFCDataset(
        texts=texts,
        sentiments=sentiments,
        topics=topics,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_dataloaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    data_dir: str = "data/uit_vsfc",
    word_segment: Optional[bool] = None,
    num_workers: int = 0,
    quick_mode: bool = False,
    language: Optional[str] = None,
    use_fast: Optional[bool] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads train, validation, and test splits and returns PyTorch DataLoaders for each.

    Args:
        tokenizer: Hugging Face tokenizer instance (or Hugging Face model identifier string,
                   e.g. 'vinai/phobert-base-v2' or 'distilroberta-base').
        batch_size: Batch size for DataLoader (default: 16).
        max_length: Maximum sequence length for Transformer tokenization (default: 128).
        data_dir: Directory containing split files (default: 'data/uit_vsfc').
        word_segment: If True, applies PyVi word segmentation before tokenization.
                      If None, defaults to True for 'vi' and False for 'en'.
        num_workers: Number of DataLoader worker threads (default: 0 for Windows compatibility).
        quick_mode: If True, only loads a small subset (e.g. 64 train, 32 val, 32 test) for fast tests.
        language: Optional language code ('vi' or 'en').
        use_fast: Optional fast-tokenizer flag when tokenizer is a string. Auto-selected if None.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Support cache_dir alias from PROJECT.md
    if "cache_dir" in kwargs:
        data_dir = kwargs["cache_dir"]

    resolved_ws, resolved_lang = _resolve_word_segment(word_segment, language, data_dir)

    # If a string identifier is passed, automatically load the tokenizer
    if isinstance(tokenizer, str):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required when passing a model name string to get_dataloaders. "
                "Install with: python -m pip install transformers"
            )
        model_id = tokenizer
        resolved_fast = ("phobert" not in model_id.lower()) if use_fast is None else bool(use_fast)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=resolved_fast)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=not resolved_fast)

    # 1. Load train split
    train_texts, train_sents, train_topics = load_vsfc_split(
        "train", data_dir=data_dir, word_segment=resolved_ws, language=resolved_lang, **kwargs
    )
    # 2. Load validation split
    val_texts, val_sents, val_topics = load_vsfc_split(
        "validation", data_dir=data_dir, word_segment=resolved_ws, language=resolved_lang, **kwargs
    )
    # 3. Load test split
    test_texts, test_sents, test_topics = load_vsfc_split(
        "test", data_dir=data_dir, word_segment=resolved_ws, language=resolved_lang, **kwargs
    )

    if quick_mode:
        subset_train = min(len(train_texts), max(batch_size * 2, 32))
        subset_val = min(len(val_texts), max(batch_size * 2, 32))
        subset_test = min(len(test_texts), max(batch_size * 2, 32))

        rng = random.Random(42)

        def _deterministic_subset(
            t_list: List[str], s_list: List[int], top_list: List[int], limit: int
        ) -> Tuple[List[str], List[int], List[int]]:
            if len(t_list) <= limit:
                return t_list, s_list, top_list
            indices = list(range(len(t_list)))
            rng.shuffle(indices)
            chosen = indices[:limit]
            return [t_list[i] for i in chosen], [s_list[i] for i in chosen], [top_list[i] for i in chosen]

        train_texts, train_sents, train_topics = _deterministic_subset(
            train_texts, train_sents, train_topics, subset_train
        )
        val_texts, val_sents, val_topics = _deterministic_subset(
            val_texts, val_sents, val_topics, subset_val
        )
        test_texts, test_sents, test_topics = _deterministic_subset(
            test_texts, test_sents, test_topics, subset_test
        )

    train_loader = create_dataloader(
        texts=train_texts,
        sentiments=train_sents,
        topics=train_topics,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        texts=val_texts,
        sentiments=val_sents,
        topics=val_topics,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        texts=test_texts,
        sentiments=test_sents,
        topics=test_topics,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
