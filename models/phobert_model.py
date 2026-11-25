"""
models/phobert_model.py
Mô hình Deep Learning chính của đồ án: Fine-tune PhoBERT trên UIT-VSFC (Sentiment + Topic).
Kiến trúc: vinai/phobert-base-v2 + Masked Mean Pooling + 2 nhánh phân loại MLP đa nhiệm.
"""

import json
import os
import sys
sys.dont_write_bytecode = True
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data.dataset import SENTIMENT_MAP, TOPIC_MAP, segment_text

PHOBERT_WEIGHTS_RELEASE_URL = "https://github.com/Tr-Hieu205/DoAn_TriTueNhanTaoNangCao/releases/tag/v1.0.0"
PHOBERT_WEIGHTS_DOWNLOAD_URL = (
    "https://github.com/Tr-Hieu205/DoAn_TriTueNhanTaoNangCao/releases/download/v1.0.0/model.pt"
)


def download_phobert_weights(weights_file: str, url: str = PHOBERT_WEIGHTS_DOWNLOAD_URL) -> str:
    """Tải trọng số model.pt từ GitHub Releases khi chạy lần đầu nếu chưa có trên máy."""
    target_dir = os.path.dirname(os.path.abspath(weights_file))
    os.makedirs(target_dir, exist_ok=True)
    tmp_file = weights_file + ".tmp"
    print(f"[*] Đang tải trọng số PhoBERT từ GitHub Releases ({url})...")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp_file, "wb") as f:
            first_chunk = resp.read(1024 * 1024)
            if not first_chunk.startswith(b"PK\x03\x04"):
                raise RuntimeError(
                    f"File tải về không đúng định dạng checkpoint PyTorch. Vui lòng tải thủ công từ: {PHOBERT_WEIGHTS_RELEASE_URL}"
                )
            f.write(first_chunk)
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp_file, weights_file)
        print(f"[+] Đã tải xong trọng số PhoBERT: {weights_file}")
        return weights_file
    except Exception as e:
        if os.path.isfile(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        raise RuntimeError(
            f"Không thể tự động tải trọng số PhoBERT ({e}). Vui lòng tải thủ công 'model.pt' từ {PHOBERT_WEIGHTS_RELEASE_URL} và đặt vào '{weights_file}'."
        ) from e


def _make_mlp_head(hidden_size: int, num_classes: int, dropout: float = 0.2) -> nn.Sequential:
    """Tạo nhánh MLP 2 tầng chuẩn kèm LayerNorm và kích hoạt phi tuyến GELU."""
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )


class PhoBERTMultiTask(nn.Module):
    """Mô hình phân loại đa nhiệm dựa trên PhoBERT cho phản hồi sinh viên tiếng Việt."""

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base-v2",
        num_sentiment: int = 3,
        num_topic: int = 4,
        dropout: float = 0.2,
        config: Optional[AutoConfig] = None,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_sentiment = num_sentiment
        self.num_topic = num_topic
        self.dropout_rate = dropout

        self.config = config or AutoConfig.from_pretrained(model_name_or_path)
        self.phobert = backbone or AutoModel.from_pretrained(model_name_or_path, config=self.config)

        hidden_size = getattr(self.config, "hidden_size", 768)
        self.sentiment_head = _make_mlp_head(hidden_size, num_sentiment, dropout)
        self.topic_head = _make_mlp_head(hidden_size, num_topic, dropout)

    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Thực hiện Masked Mean Pooling trên toàn chuỗi token có chú ý."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        topic_labels: Optional[torch.Tensor] = None,
        sentiment_weights: Optional[torch.Tensor] = None,
        topic_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Lan truyền tiến tính loss và logit đồng thời cho cả 2 tác vụ."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)

        s_logits = self.sentiment_head(pooled)
        t_logits = self.topic_head(pooled)

        loss = None
        if sentiment_labels is not None and topic_labels is not None:
            fn_s = nn.CrossEntropyLoss(weight=sentiment_weights, label_smoothing=0.03)
            fn_t = nn.CrossEntropyLoss(weight=topic_weights, label_smoothing=0.03)
            loss = fn_s(s_logits, sentiment_labels.long()) + fn_t(t_logits, topic_labels.long())

        return {"loss": loss, "sentiment_logits": s_logits, "topic_logits": t_logits, "pooled": pooled}

    def predict_proba(
        self,
        texts: Union[str, List[str]],
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 64,
        batch_size: int = 64,
        word_segment: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán phân phối xác suất softmax."""
        input_list = [texts] if isinstance(texts, str) else list(texts)
        if not input_list:
            return np.empty((0, self.num_sentiment)), np.empty((0, self.num_topic))

        tok_src = getattr(self, "checkpoint_dir", self.model_name_or_path)
        tok = tokenizer or AutoTokenizer.from_pretrained(tok_src, use_fast=False)
        dev = device or next(self.parameters()).device

        was_training = self.training
        self.eval()
        s_prob_list, t_prob_list = [], []

        with torch.no_grad():
            for i in range(0, len(input_list), batch_size):
                batch = input_list[i : i + batch_size]
                proc = [segment_text(t) if word_segment else str(t or "") for t in batch]
                enc = tok(proc, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = self.forward(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
                s_prob_list.append(torch.softmax(out["sentiment_logits"], dim=-1).cpu().numpy())
                t_prob_list.append(torch.softmax(out["topic_logits"], dim=-1).cpu().numpy())

        if was_training:
            self.train()
        return np.vstack(s_prob_list), np.vstack(t_prob_list)

    def predict_labels(
        self, texts: Union[str, List[str]], tokenizer: Any = None, device: Optional[Union[str, torch.device]] = None,
        max_length: int = 64, batch_size: int = 64, word_segment: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """Dự đoán nhãn ID cho cả 2 tác vụ."""
        sp, tp = self.predict_proba(texts, tokenizer, device, max_length, batch_size, word_segment)
        return (
            [] if len(sp) == 0 else np.argmax(sp, axis=1).astype(int).tolist(),
            [] if len(tp) == 0 else np.argmax(tp, axis=1).astype(int).tolist(),
        )

    def predict(
        self, texts: Union[str, List[str]], tokenizer: Any = None, device: Optional[Union[str, torch.device]] = None,
        max_length: int = 64, batch_size: int = 64, word_segment: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Giao diện dự đoán chuẩn trả về nhãn, độ tin cậy và phân phối xác suất."""
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else list(texts)
        if not input_list:
            return []

        sp, tp = self.predict_proba(input_list, tokenizer, device, max_length, batch_size, word_segment)
        results = []
        for i in range(len(input_list)):
            sid, tid = int(np.argmax(sp[i])), int(np.argmax(tp[i]))
            results.append({
                "sentiment": {
                    "label": SENTIMENT_MAP[sid], "label_id": sid, "confidence": float(sp[i, sid]),
                    "distribution": {SENTIMENT_MAP[c]: float(sp[i, c]) for c in range(self.num_sentiment)},
                },
                "topic": {
                    "label": TOPIC_MAP[tid], "label_id": tid, "confidence": float(tp[i, tid]),
                    "distribution": {TOPIC_MAP[c]: float(tp[i, c]) for c in range(self.num_topic)},
                },
            })
        return results[0] if is_single else results

    def save_pretrained(self, save_directory: str) -> str:
        """Lưu trọng số model.pt và cấu hình vào thư mục."""
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(save_directory)
        meta = {
            "model_type": "PhoBERTMultiTask", "base_model_name_or_path": self.model_name_or_path,
            "num_sentiment": self.num_sentiment, "num_topic": self.num_topic, "dropout": self.dropout_rate,
        }
        with open(os.path.join(save_directory, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        return save_directory

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        device: Optional[Union[str, torch.device]] = None,
        auto_download: bool = True,
    ) -> "PhoBERTMultiTask":
        """Nạp mô hình đã lưu từ thư mục checkpoint (tự động tải từ GitHub Releases nếu thiếu)."""
        save_dir = str(save_directory)
        weights_file = os.path.join(save_dir, "model.pt")
        if not os.path.isfile(weights_file):
            alt = os.path.join(save_dir, "pytorch_model.bin")
            weights_file = alt if os.path.isfile(alt) else weights_file
            if not os.path.isfile(weights_file):
                if auto_download and os.path.isfile(os.path.join(save_dir, "model_meta.json")):
                    download_phobert_weights(weights_file)
                else:
                    raise FileNotFoundError(f"Không tìm thấy file trọng số trong '{save_dir}'.")

        meta = {}
        meta_file = os.path.join(save_dir, "model_meta.json")
        if os.path.isfile(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        base_model = meta.get("base_model_name_or_path", "vinai/phobert-base-v2")
        config = AutoConfig.from_pretrained(save_dir if os.path.isfile(os.path.join(save_dir, "config.json")) else base_model)
        try:
            backbone = AutoModel.from_config(config)
        except Exception:
            backbone = AutoModel.from_pretrained(base_model, config=config)

        model = cls(
            model_name_or_path=base_model,
            num_sentiment=meta.get("num_sentiment", 3),
            num_topic=meta.get("num_topic", 4),
            dropout=meta.get("dropout", 0.2),
            config=config, backbone=backbone,
        )
        map_location = device or "cpu"
        state_dict = torch.load(weights_file, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        model.checkpoint_dir = save_dir
        if device is not None:
            model.to(device)
        return model
