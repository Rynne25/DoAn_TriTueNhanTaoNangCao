"""
tests/test_bilingual_core.py
Fast (<2s), offline, deterministic unit tests for Milestone M1 features (F2-F5):
- Data pipeline R1 fixes & English loading
- Language detection, model metadata, checkpoint_dir tokenizer lookup, bilingual routing
- Training pipeline R1 fixes & EN CLI/metadata
- Evaluation R1 fixes, dynamic distributions, majority-class baseline, and EN CLI defaults
"""

import csv
import json
import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import joblib
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from data.dataset import get_dataloaders, load_vsfc_split
from data.download_data import _is_valid_cached_parquet, download_vsfc_dataset
from evaluation import evaluate_majority_baseline
from evaluation.evaluate import (
    format_terminal_table,
    generate_json_report,
    generate_markdown_report,
    get_file_or_dir_size_mb,
    parse_args,
)
from models.baseline_model import VietnameseFeedbackBaseline
from models.bilingual import BilingualBaselinePredictor, BilingualTransformerPredictor
from models.language import detect_language
from models.phobert_model import PhoBERTMultiTask
from training.train_baseline import train_baseline
from training.train_phobert import _extract_frozen_layer_cache, train_phobert


class _DeterministicTinyTokenizer:
    """Fast offline deterministic tokenizer stub for unit tests."""

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        seq_len = min(max_length, 8)
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        for i, t in enumerate(texts):
            tokens = str(t or "").split()[:seq_len]
            for j, tok in enumerate(tokens):
                input_ids[i, j] = (sum(ord(c) for c in tok) % 95) + 3
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "RobertaTokenizerFast"}, f)


def _write_csv_split(directory, split_name, rows):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{split_name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "sentiment", "topic"])
        for row in rows:
            writer.writerow(row)
    return path


def _make_tiny_roberta():
    cfg = RobertaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=34,
        type_vocab_size=1,
    )
    backbone = RobertaModel(cfg)
    return cfg, backbone


class TestDataPipelineBilingual(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="bilingual_data_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parquet_cache_validation_requires_par1_magic(self):
        bad_file = os.path.join(self.temp_dir, "train.parquet")
        with open(bad_file, "wb") as f:
            f.write(b"HTML" * 400)
        self.assertFalse(_is_valid_cached_parquet(bad_file))

        good_file = os.path.join(self.temp_dir, "valid.parquet")
        with open(good_file, "wb") as f:
            f.write(b"PAR1" + b"\x00" * 1200)
        self.assertTrue(_is_valid_cached_parquet(good_file))

        def fake_download(url, dest_path):
            import polars as pl
            df = pl.DataFrame({
                "sentence": ["Feedback sample " + ("x" * 60) for _ in range(30)],
                "sentiment": [i % 3 for i in range(30)],
                "topic": [i % 4 for i in range(30)],
            })
            df.write_parquet(dest_path)
            return os.path.getsize(dest_path)

        with patch("data.download_data._download_stream_to_file", side_effect=fake_download) as mock_dl:
            res = download_vsfc_dataset(output_dir=self.temp_dir, force=False, splits=["train"])
            self.assertEqual(mock_dl.call_count, 1)
            self.assertTrue(_is_valid_cached_parquet(res["train"]))

    def test_load_english_split_skips_pyvi_and_respects_override(self):
        _write_csv_split(
            self.temp_dir,
            "test",
            [
                ("The lecturer explains concepts very clearly.", 2, 0),
                ("The classroom projector is broken.", 0, 2),
            ],
        )
        texts, sents, topics = load_vsfc_split("test", data_dir=self.temp_dir, language="en")
        self.assertEqual(texts[0], "The lecturer explains concepts very clearly.")
        self.assertNotIn("_", texts[0])
        self.assertEqual(sents, [2, 0])
        self.assertEqual(topics, [0, 2])

        with self.assertRaises(ValueError):
            load_vsfc_split("test", data_dir=self.temp_dir, language="fr")

    def test_malformed_labels_and_empty_sentences_diagnostics(self):
        _write_csv_split(
            self.temp_dir,
            "train",
            [
                ("Valid English sentence", "1.5", 0),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            load_vsfc_split("train", data_dir=self.temp_dir, language="en")
        self.assertIn("index 0", str(ctx.exception).lower())
        self.assertIn("sentiment", str(ctx.exception).lower())

        _write_csv_split(
            self.temp_dir,
            "validation",
            [
                ("   ", 2, 0),
            ],
        )
        with self.assertRaises(ValueError) as ctx_empty:
            load_vsfc_split("validation", data_dir=self.temp_dir, language="en")
        self.assertIn("empty", str(ctx_empty.exception).lower())

    def test_auto_download_not_called_for_en_and_preserves_vi_error(self):
        empty_en = os.path.join(self.temp_dir, "english_feedback")
        os.makedirs(empty_en, exist_ok=True)
        with patch("data.download_data.download_vsfc_dataset") as mock_dl:
            with self.assertRaises(FileNotFoundError):
                load_vsfc_split("test", data_dir=empty_en, auto_download=True, language="en")
            mock_dl.assert_not_called()

        empty_vi = os.path.join(self.temp_dir, "uit_missing")
        os.makedirs(empty_vi, exist_ok=True)
        with patch("data.download_data.download_vsfc_dataset", side_effect=RuntimeError("network down")):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_vsfc_split("test", data_dir=empty_vi, auto_download=True, language="vi")
            self.assertIn("network down", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_get_dataloaders_use_fast_fallback_and_quick_mode_shuffle(self):
        rows = [(f"English sample number {i} for class {i % 3}", i % 3, i % 4) for i in range(40)]
        for sp in ("train", "validation", "test"):
            _write_csv_split(self.temp_dir, sp, rows)

        tok = _DeterministicTinyTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.side_effect = [ValueError("fast unavailable"), tok]
            tr, va, te = get_dataloaders(
                tokenizer="distilroberta-base",
                batch_size=8,
                max_length=16,
                data_dir=self.temp_dir,
                language="en",
                quick_mode=True,
            )
            self.assertEqual(mock_from_pretrained.call_args_list[0].kwargs["use_fast"], True)
            self.assertEqual(mock_from_pretrained.call_args_list[1].kwargs["use_fast"], False)
            self.assertEqual(len(tr.dataset), 32)
            self.assertEqual(len(va.dataset), 32)
            self.assertEqual(len(te.dataset), 32)


class TestLanguageAndModelsBilingual(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="bilingual_models_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_language_rules_and_edge_cases(self):
        self.assertEqual(detect_language("Thầy giảng bài rất nhiệt tình và dễ hiểu."), "vi")
        self.assertEqual(detect_language("Tha\u0302\u0300y gia\u0309ng ra\u0302\u0301t hay"), "vi")  # NFD
        self.assertEqual(detect_language("Professor dạy rất good và nhiệt tình"), "vi")
        self.assertEqual(detect_language("thay giang bai rat nhiet tinh"), "vi")
        self.assertEqual(detect_language("The lecturer explains concepts very clearly."), "en")
        self.assertEqual(detect_language("The campus café is noisy during exams."), "en")
        self.assertEqual(detect_language("The phở at the student cafe is good."), "en")
        self.assertEqual(detect_language(""), "en")
        self.assertEqual(detect_language(None), "en")

    def test_baseline_legacy_unpickle_and_default_word_segment(self):
        en_texts = ["great lecturer clear slides", "good teaching helpful lecturer", "broken wifi noisy room", "bad projector terrible room"]
        en_s = [2, 2, 0, 0]
        en_t = [0, 0, 2, 2]
        model = VietnameseFeedbackBaseline(min_df=1, max_features=200, default_word_segment=False, language="en")
        model.fit(en_texts, en_s, en_t)
        pred = model.predict("great lecturer")
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)

        # Simulate legacy pickled instance missing default_word_segment & language
        legacy_path = os.path.join(self.temp_dir, "legacy_baseline.joblib")
        del model.default_word_segment
        del model.language
        joblib.dump(model, legacy_path)
        self.assertTrue(model._resolve_word_segment(None))

        loaded = VietnameseFeedbackBaseline.load(legacy_path)
        self.assertTrue(loaded.default_word_segment)
        self.assertEqual(loaded.language, "vi")

    def test_phobert_meta_and_local_checkpoint_dir_tokenizer(self):
        cfg, backbone = _make_tiny_roberta()
        model = PhoBERTMultiTask(
            model_name_or_path="distilroberta-base",
            config=cfg,
            backbone=backbone,
            default_word_segment=False,
            language="en",
        )
        ckpt_dir = os.path.join(self.temp_dir, "en_ckpt")
        model.save_pretrained(ckpt_dir)
        _DeterministicTinyTokenizer().save_pretrained(ckpt_dir)

        with open(os.path.join(ckpt_dir, "phobert_multitask_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertFalse(meta["default_word_segment"])
        self.assertEqual(meta["language"], "en")

        reloaded = PhoBERTMultiTask.from_pretrained(ckpt_dir, device="cpu")
        self.assertEqual(reloaded.checkpoint_dir, os.path.abspath(ckpt_dir))
        self.assertFalse(reloaded.default_word_segment)
        self.assertEqual(reloaded.language, "en")

        tok = _DeterministicTinyTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok) as mock_tok:
            out = reloaded.predict(["The lecturer is helpful."])
            self.assertEqual(len(out), 1)
            self.assertEqual(mock_tok.call_args.args[0], os.path.abspath(ckpt_dir))
            self.assertTrue(mock_tok.call_args.kwargs["local_files_only"])

        with self.assertRaises(ValueError):
            reloaded.predict_proba(["test"], tokenizer=tok, batch_size=0)

    def test_bilingual_predictors_order_and_fallback(self):
        vi_m = VietnameseFeedbackBaseline(min_df=1, max_features=200, default_word_segment=True, language="vi")
        vi_m.fit(["Thầy dạy rất hay", "Phòng học quá nóng", "Chương trình ổn", "Học phí cao"], [2, 0, 1, 0], [0, 2, 1, 3])

        en_m = VietnameseFeedbackBaseline(min_df=1, max_features=200, default_word_segment=False, language="en")
        en_m.fit(["Great lecturer", "Hot broken classroom", "Syllabus is okay", "Tuition fee high"], [2, 0, 1, 0], [0, 2, 1, 3])

        vi_p = os.path.join(self.temp_dir, "b_vi.joblib")
        en_p = os.path.join(self.temp_dir, "b_en.joblib")
        vi_m.save(vi_p)
        en_m.save(en_p)

        bp = BilingualBaselinePredictor(vi_path=vi_p, en_path=en_p)
        batch_out = bp.predict(["The lecturer is great", "Thầy giảng rất tốt", None])
        self.assertEqual([x["language"] for x in batch_out], ["en", "vi", "en"])
        self.assertIn("sentiment", batch_out[0])
        self.assertIn("topic", batch_out[1])

        # Fallback when only EN exists
        bp_en_only = BilingualBaselinePredictor(vi_path=os.path.join(self.temp_dir, "missing.joblib"), en_path=en_p)
        fb = bp_en_only.predict("Thầy giảng rất tốt")
        self.assertEqual(fb["language"], "vi")

        # BilingualTransformerPredictor with tiny models
        cfg, bb1 = _make_tiny_roberta()
        _, bb2 = _make_tiny_roberta()
        tok = _DeterministicTinyTokenizer()
        t_vi = PhoBERTMultiTask(model_name_or_path="tiny-vi", config=cfg, backbone=bb1, default_word_segment=True, language="vi")
        t_en = PhoBERTMultiTask(model_name_or_path="tiny-en", config=cfg, backbone=bb2, default_word_segment=False, language="en")
        tp = BilingualTransformerPredictor(vi_dir=None, en_dir=None, vi_model=t_vi, vi_tokenizer=tok, en_model=t_en, en_tokenizer=tok)
        t_out = tp.predict(["Great course", "Phòng học mát mẻ"])
        self.assertEqual([x["language"] for x in t_out], ["en", "vi"])


class TestTrainingAndEvaluationBilingual(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="bilingual_train_eval_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_baseline_en_saves_meta_json(self):
        data_dir = os.path.join(self.temp_dir, "en_data")
        rows = [
            ("Great lecturer very clear", 2, 0),
            ("Terrible broken projector", 0, 2),
            ("Syllabus is fine overall", 1, 1),
            ("Tuition administration slow", 0, 3),
        ] * 4
        for sp in ("train", "validation", "test"):
            _write_csv_split(data_dir, sp, rows)

        save_path = os.path.join(self.temp_dir, "baseline_en.joblib")
        model = train_baseline(
            data_dir=data_dir,
            save_path=save_path,
            max_features=200,
            language="en",
            word_segment=False,
        )
        self.assertFalse(model.default_word_segment)
        self.assertEqual(model.language, "en")
        meta_path = os.path.join(self.temp_dir, "baseline_en_meta.json")
        self.assertTrue(os.path.isfile(meta_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["language"], "en")
        self.assertFalse(meta["default_word_segment"])
        self.assertIsNotNone(meta["test_metrics"])

    def test_train_phobert_epochs_zero_and_best_model_return(self):
        data_dir = os.path.join(self.temp_dir, "en_data")
        rows = [
            ("Great lecturer very clear", 2, 0),
            ("Terrible broken projector", 0, 2),
            ("Syllabus is fine overall", 1, 1),
            ("Tuition administration slow", 0, 3),
        ] * 4
        for sp in ("train", "validation", "test"):
            _write_csv_split(data_dir, sp, rows)

        cfg, backbone = _make_tiny_roberta()
        tok = _DeterministicTinyTokenizer()
        save_dir_0 = os.path.join(self.temp_dir, "ckpt_ep0")
        with patch("training.train_phobert._load_training_tokenizer", return_value=tok), \
             patch("training.train_phobert.PhoBERTMultiTask", return_value=PhoBERTMultiTask(
                 model_name_or_path="distilroberta-base", config=cfg, backbone=backbone, default_word_segment=False, language="en"
             )):
            _, summary = train_phobert(
                epochs=0,
                batch_size=8,
                max_length=16,
                model_name="distilroberta-base",
                save_dir=save_dir_0,
                data_dir=data_dir,
                quick_run=True,
                eval_test=False,
                language="en",
                word_segment=False,
            )
        self.assertTrue(os.path.isfile(os.path.join(save_dir_0, "history.json")))
        self.assertEqual(summary["language"], "en")
        self.assertFalse(summary["word_segment"])

        # Tuple-returning layer guard check in _extract_frozen_layer_cache
        class _EmbWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(128, 16)

            def forward(self, input_ids):
                return self.emb(input_ids)

        class _TupleLayer(nn.Module):
            def forward(self, x, attention_mask=None):
                return (x,)

        class _TupleBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = _EmbWrapper()
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList([_TupleLayer()])

            def _create_attention_masks(self, attention_mask, **kwargs):
                return attention_mask, None

        tiny_model = PhoBERTMultiTask(
            model_name_or_path="tuple-bb",
            config=RobertaConfig(vocab_size=128, hidden_size=16, num_hidden_layers=1, num_attention_heads=2),
            backbone=_TupleBackbone(),
        )
        loader = torch.utils.data.DataLoader(
            [{"input_ids": torch.ones(4, dtype=torch.long), "attention_mask": torch.ones(4, dtype=torch.long), "sentiment_labels": torch.tensor(0), "topic_labels": torch.tensor(0)}],
            batch_size=1,
        )
        cached = _extract_frozen_layer_cache(tiny_model, loader, torch.device("cpu"), freeze_layers=1)
        self.assertEqual(len(cached), 1)

    def test_evaluation_majority_dynamic_json_md_and_cli_defaults(self):
        maj = evaluate_majority_baseline(
            sentiments=[0, 0, 2, 1, 0],
            topics=[0, 0, 0, 2, 3],
            train_sentiments=[2, 2, 0],
            train_topics=[0, 0, 1],
        )
        self.assertEqual(maj["majority_sentiment_label"], "positive")
        self.assertEqual(maj["majority_topic_label"], "lecturer")
        self.assertAlmostEqual(maj["sentiment_accuracy"], 0.2, places=4)
        self.assertAlmostEqual(maj["topic_accuracy"], 0.6, places=4)

        b_metrics = {
            "sentiment_accuracy": 0.8, "sentiment_macro_f1": 0.7,
            "topic_accuracy": 0.75, "topic_macro_f1": 0.65,
            "mean_macro_f1": 0.675, "storage_footprint_mb": 0.5,
            "per_class": {
                "sentiment": {
                    "negative": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 10},
                    "neutral": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 5},
                    "positive": {"precision": 0.85, "recall": 0.85, "f1-score": 0.85, "support": 15},
                },
                "topic": {
                    "lecturer": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 12},
                    "training_program": {"precision": 0.6, "recall": 0.6, "f1-score": 0.6, "support": 8},
                    "facility": {"precision": 0.7, "recall": 0.7, "f1-score": 0.7, "support": 6},
                    "others": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 4},
                },
            },
        }
        p_metrics = dict(b_metrics)
        json_out = os.path.join(self.temp_dir, "metrics_en.json")
        generate_json_report(
            b_metrics, p_metrics, {},
            split_name="test", num_samples=30, device_name="CPU",
            output_path=json_out, language="en", majority_baseline=maj,
        )
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["metadata"]["dataset"]["name"], "English Student Feedback")
        self.assertEqual(data["metadata"]["dataset"]["sentiment_distribution"], {"negative": 10, "neutral": 5, "positive": 15})
        self.assertEqual(data["metadata"]["dataset"]["topic_distribution"], {"lecturer": 12, "training_program": 8, "facility": 6, "others": 4})
        self.assertIn("majority_baseline", data)

        md_out = os.path.join(self.temp_dir, "summary_en.md")
        generate_markdown_report(
            b_metrics, p_metrics, {},
            split_name="test", num_samples=30, device_name="CPU",
            output_path=md_out, language="en", majority_baseline=maj,
        )
        with open(md_out, "r", encoding="utf-8") as f:
            md_text = f.read()
        self.assertIn("English Student Feedback", md_text)
        self.assertNotIn("UIT-VSFC", md_text)
        self.assertIn("Majority-Class Reference", md_text)

        table = format_terminal_table(b_metrics, p_metrics, {}, num_samples=30, language="en", majority_baseline=maj)
        self.assertIn("ENGLISH STUDENT FEEDBACK", table)
        self.assertIn("Majority Baseline", table)

        # CLI defaults preserved vs EN resolved
        args_vi = parse_args([])
        self.assertEqual(args_vi.baseline_path, "saved_models/baseline.joblib")
        self.assertEqual(args_vi.phobert_path, "saved_models/phobert_multitask")
        args_en = parse_args(["--language", "en"])
        self.assertEqual(args_en.data_dir, "data/english_feedback")
        self.assertEqual(args_en.baseline_path, "saved_models/baseline_en.joblib")
        self.assertEqual(args_en.phobert_path, "saved_models/english_multitask")
        self.assertTrue(args_en.metrics_out.endswith("metrics_en.json"))
        self.assertTrue(args_en.markdown_out.endswith("comparison_summary_en.md"))

        # Multi-file dir size check (even under mocked os.path.isfile=True)
        size_dir = os.path.join(self.temp_dir, "size_dir")
        os.makedirs(size_dir, exist_ok=True)
        with open(os.path.join(size_dir, "model.pt"), "wb") as f:
            f.write(b"a" * (1024 * 1024))
        with open(os.path.join(size_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"b" * (1024 * 1024))
        with patch("os.path.isfile", return_value=True):
            self.assertAlmostEqual(get_file_or_dir_size_mb(size_dir), 2.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
