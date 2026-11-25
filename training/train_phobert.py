"""
training/train_phobert.py
Huấn luyện PhoBERT Multi-Task trên 11.426 câu UIT-VSFC (Sentiment + Topic).
Kiến trúc: vinai/phobert-base-v2 (fine-tune thật 6 tầng cuối) + Masked Mean Pooling + Dual MLP Heads.
Tối ưu tốc độ CPU bằng Dynamic Length Bucketing (rút ngắn thời gian huấn luyện ~8 lần).
"""

import argparse
import os
import random
import sys
sys.dont_write_bytecode = True
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import load_vsfc_split
from models.phobert_model import PhoBERTMultiTask


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho tính tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_smoothed_weights(labels: List[int], num_classes: int, power: float = 0.35) -> torch.Tensor:
    """Tính trọng số lớp làm mịn để cân bằng dữ liệu mất cân bằng nhãn."""
    counts = np.bincount(labels, minlength=num_classes)
    weights = (len(labels) / (num_classes * np.maximum(counts, 1))) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_bucketed_batches(
    texts: List[str], sents: List[int], topics: List[int],
    tokenizer, batch_size: int = 32, max_length: int = 128, shuffle: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Gom nhóm câu theo độ dài tương đồng và pad động theo từng batch để tối ưu tốc độ CPU."""
    lengths = [len(t.split()) for t in texts]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches_idx = [sorted_indices[i : i + batch_size] for i in range(0, len(sorted_indices), batch_size)]
    if shuffle:
        random.shuffle(batches_idx)

    collated = []
    for b_idx in batches_idx:
        b_texts = [texts[i] for i in b_idx]
        b_s = torch.tensor([sents[i] for i in b_idx], dtype=torch.long)
        b_t = torch.tensor([topics[i] for i in b_idx], dtype=torch.long)
        enc = tokenizer(b_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        collated.append((enc["input_ids"], enc["attention_mask"], b_s, b_t))
    return collated


def evaluate_loader(model: PhoBERTMultiTask, batches, device) -> Dict[str, float]:
    """Đánh giá loss, Accuracy và Macro-F1 trên tập dữ liệu đã gom batch."""
    model.eval()
    all_s_preds, all_s_true, all_t_preds, all_t_true = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for b_ids, b_mask, b_s, b_t in batches:
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_s = b_s.to(device)
            b_t = b_t.to(device)

            out = model(input_ids=b_ids, attention_mask=b_mask, sentiment_labels=b_s, topic_labels=b_t)
            total_loss += out["loss"].item() if out["loss"] is not None else 0.0
            all_s_preds.extend(torch.argmax(out["sentiment_logits"], dim=-1).cpu().tolist())
            all_s_true.extend(b_s.cpu().tolist())
            all_t_preds.extend(torch.argmax(out["topic_logits"], dim=-1).cpu().tolist())
            all_t_true.extend(b_t.cpu().tolist())

    s_acc = accuracy_score(all_s_true, all_s_preds)
    s_wf1 = f1_score(all_s_true, all_s_preds, average="weighted", zero_division=0)
    s_mf1 = f1_score(all_s_true, all_s_preds, average="macro", zero_division=0)
    t_acc = accuracy_score(all_t_true, all_t_preds)
    t_wf1 = f1_score(all_t_true, all_t_preds, average="weighted", zero_division=0)
    t_mf1 = f1_score(all_t_true, all_t_preds, average="macro", zero_division=0)

    return {
        "loss": total_loss / max(1, len(batches)),
        "sent_acc": s_acc, "sent_wf1": s_wf1, "sent_mf1": s_mf1,
        "topic_acc": t_acc, "topic_wf1": t_wf1, "topic_mf1": t_mf1,
        "mean_mf1": (s_mf1 + t_mf1) / 2.0,
    }


def train_phobert(
    data_dir: str = "data/uit_vsfc",
    save_dir: str = "saved_models/phobert",
    epochs: int = 3,
    batch_size: int = 32,
    max_length: int = 128,
):
    """Quy trình Fine-Tuning 6 tầng Transformer trên cùng của PhoBERT trên UIT-VSFC."""
    set_seed(42)
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    print("=" * 72 + f"\n HUẤN LUYỆN PHOBERT MULTI-TASK (Thiết bị: {dev_str.upper()})\n" + "=" * 72)

    # 1. Tải tập dữ liệu
    print(f"\n[1/4] Đang nạp dữ liệu tiếng Việt UIT-VSFC từ '{data_dir}'...")
    tr_x, tr_s, tr_t = load_vsfc_split("train", data_dir=data_dir, word_segment=True)
    va_x, va_s, va_t = load_vsfc_split("validation", data_dir=data_dir, word_segment=True)
    te_x, te_s, te_t = load_vsfc_split("test", data_dir=data_dir, word_segment=True)
    print(f"      Số lượng mẫu -> Train: {len(tr_x):,}, Val: {len(va_x):,}, Test: {len(te_x):,}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    model = PhoBERTMultiTask(model_name_or_path="vinai/phobert-base-v2").to(device)

    # 2. Cấu hình đóng băng: đóng băng embeddings + 6 tầng đầu (layers 0..5), mở khóa fine-tune 6 tầng cuối (layers 6..11)
    print("\n[2/4] Thiết lập đóng băng 6 tầng đầu và mở khóa fine-tune 6 tầng trên cùng...")
    if hasattr(model.phobert, "embeddings"):
        for p in model.phobert.embeddings.parameters():
            p.requires_grad = False
    if hasattr(model.phobert, "encoder") and hasattr(model.phobert.encoder, "layer"):
        for layer in model.phobert.encoder.layer[:6]:
            for p in layer.parameters():
                p.requires_grad = False

    transformer_trainable = [p for layer in model.phobert.encoder.layer[6:] for p in layer.parameters() if p.requires_grad]
    head_trainable = list(model.sentiment_head.parameters()) + list(model.topic_head.parameters())
    print(f"      Tham số cần cập nhật: {len(transformer_trainable)} tensor Transformer (layers 6..11) + {len(head_trainable)} tensor MLP.")

    optimizer = AdamW([
        {"params": transformer_trainable, "lr": 3e-5},
        {"params": head_trainable, "lr": 4.5e-4},
    ], weight_decay=0.01)

    # 3. Chuẩn bị batch động và tính trọng số làm mịn
    print("\n[3/4] Chuẩn bị batch động (Dynamic Bucketing) và tính trọng số lớp...")
    val_batches = build_bucketed_batches(va_x, va_s, va_t, tokenizer, batch_size=batch_size, max_length=max_length, shuffle=False)
    test_batches = build_bucketed_batches(te_x, te_s, te_t, tokenizer, batch_size=batch_size, max_length=max_length, shuffle=False)
    total_steps = (len(tr_x) // batch_size + 1) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    s_weights = compute_smoothed_weights(tr_s, num_classes=3, power=0.35).to(device)
    t_weights = compute_smoothed_weights(tr_t, num_classes=4, power=0.35).to(device)

    # 4. Vòng lặp huấn luyện chính
    best_val_f1 = -1.0
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[4/4] Bắt đầu huấn luyện {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_batches = build_bucketed_batches(tr_x, tr_s, tr_t, tokenizer, batch_size=batch_size, max_length=max_length, shuffle=True)
        running_loss = 0.0

        for b_ids, b_mask, b_s, b_t in train_batches:
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_s = b_s.to(device)
            b_t = b_t.to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=b_ids, attention_mask=b_mask,
                sentiment_labels=b_s, topic_labels=b_t,
                sentiment_weights=s_weights, topic_weights=t_weights,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        epoch_time = time.time() - t0
        avg_loss = running_loss / max(1, len(train_batches))
        val_res = evaluate_loader(model, val_batches, device)
        print(f"  Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Loss: {avg_loss:.4f} | Val Sent F1: {val_res['sent_mf1']*100:.2f}%, Topic F1: {val_res['topic_mf1']*100:.2f}%, Mean F1: {val_res['mean_mf1']*100:.2f}%")

        if val_res["mean_mf1"] > best_val_f1:
            best_val_f1 = val_res["mean_mf1"]
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"      [*] Đã lưu checkpoint mới tốt nhất (Mean Macro-F1: {best_val_f1*100:.2f}%).")

    # Đánh giá cuối cùng trên tập Test
    print("\n" + "=" * 72 + "\n ĐÁNH GIÁ CHECKPOINT TỐT NHẤT TRÊN TẬP TEST (N=3.166):\n" + "-" * 72)
    best_model = PhoBERTMultiTask.from_pretrained(save_dir, device=device)
    test_res = evaluate_loader(best_model, test_batches, device)
    print(f"  • Cảm xúc (Sentiment): Accuracy = {test_res['sent_acc']*100:.2f}% | Macro-F1 = {test_res['sent_mf1']*100:.2f}%")
    print(f"  • Chủ đề (Topic):     Accuracy = {test_res['topic_acc']*100:.2f}% | Macro-F1 = {test_res['topic_mf1']*100:.2f}%")
    print(f"  • Điểm trung bình (Mean Macro-F1): {test_res['mean_mf1']*100:.2f}%")
    print("-" * 72 + f"\n[*] Checkpoint hoàn chỉnh sẵn sàng tại '{save_dir}'.\n")
    return best_model


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT Multi-Task trên UIT-VSFC")
    parser.add_argument("--data_dir", type=str, default="data/uit_vsfc")
    parser.add_argument("--save_dir", type=str, default="saved_models/phobert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    train_phobert(data_dir=args.data_dir, save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
