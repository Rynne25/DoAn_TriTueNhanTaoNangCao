"""Tier 1 E2E Tests: Feature Coverage across F1-F11 (>= 5 tests per feature)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from typing import Any, Dict, List

import torch

from e2e_tests.e2e_helpers import (
    APPTEST_TIMEOUT,
    APP_PATH,
    EXPECTED_VI_PRESETS_SNAPSHOT,
    ORIGINAL_TWELVE_TEST_FILES,
    PROJECT_ROOT,
    SENTIMENT_MAP,
    TOPIC_MAP,
    assert_prediction_schema,
    compute_majority_baseline,
    create_tiny_local_hf_checkpoint,
    extract_majority_baseline_metrics,
    find_en_dataset_split_path,
    find_main_button,
    get_cached_baseline,
    get_cached_phobert,
    get_e2e_scope,
    has_vi_diacritics,
    load_local_tokenizer,
    require_en_deliverables,
    resolve_bilingual_router,
    resolve_detect_language,
    run_unit_test_suite_cached,
    select_model_tier,
    write_synthetic_vsfc_dir,
)


class TestF1CodebaseReviewReport(unittest.TestCase):
    """F1: Written Codebase Review Report (`CODE_REVIEW.md`)."""

    def _read_review(self) -> str:
        require_en_deliverables(self)
        path = os.path.join(PROJECT_ROOT, "CODE_REVIEW.md")
        self.assertTrue(
            os.path.isfile(path),
            "M1 deliverable missing: CODE_REVIEW.md not found at repository root",
        )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_f1_code_review_file_exists_and_nonempty(self) -> None:
        content = self._read_review()
        self.assertGreater(len(content.encode("utf-8")), 1000)

    def test_f1_covers_all_six_core_directories(self) -> None:
        content = self._read_review().lower()
        for d in ("models/", "data/", "training/", "evaluation/", "app/", "tests/"):
            self.assertIn(d, content, f"CODE_REVIEW.md must cover directory '{d}'")

    def test_f1_cites_concrete_file_paths(self) -> None:
        content = self._read_review()
        expected_files = (
            "models/baseline_model.py",
            "models/phobert_model.py",
            "data/dataset.py",
            "data/download_data.py",
            "training/train_baseline.py",
            "training/train_phobert.py",
            "evaluation/evaluate.py",
            "app/app.py",
        )
        for rel in expected_files:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, rel)),
                f"Expected repo file missing: {rel}",
            )
            self.assertTrue(
                rel in content or os.path.basename(rel) in content,
                f"CODE_REVIEW.md should reference concrete module '{rel}'",
            )

    def test_f1_documents_concrete_issues_and_fixes(self) -> None:
        content = self._read_review()
        headings = re.findall(r"^#{1,4}\s+.+$", content, flags=re.MULTILINE)
        self.assertGreaterEqual(len(headings), 4, "Expected structured markdown headings")
        low = content.lower()
        self.assertTrue(
            any(tok in low for tok in ("use_fast", "word_segment", "checkpoint_dir")),
            "CODE_REVIEW.md should document concrete R1 model/data findings",
        )
        self.assertTrue(
            any(tok in low for tok in ("hardcod", "distribution", "majority", "epochs=0", "best_model")),
            "CODE_REVIEW.md should document concrete R1 training/evaluation findings",
        )
        self.assertTrue(
            any(tok in low for tok in ("text_area", "session_state", "cache", "oom", "device")),
            "CODE_REVIEW.md should document concrete R1 Streamlit app findings",
        )

    def test_f1_explicitly_addresses_english_adaptation(self) -> None:
        low = self._read_review().lower()
        self.assertIn("english", low)
        self.assertIn("vietnamese", low)
        self.assertTrue(
            any(tok in low for tok in ("sample library", "sample comment", "preset", "get_english_presets")),
            "CODE_REVIEW.md must address R3 dual sample libraries",
        )

    def test_f1_review_does_not_leak_secrets_or_broken_placeholders(self) -> None:
        content = self._read_review()
        for bad in ("TODO", "TBD", "FIXME", "XXX", "<placeholder>"):
            self.assertNotIn(bad, content, f"CODE_REVIEW.md contains unfinished marker '{bad}'")


class TestF2DataPipeline(unittest.TestCase):
    """F2: Data Pipeline R1 Fixes & EN Support (`data/dataset.py`, `data/download_data.py`)."""

    def test_f2_vi_splits_preserve_exact_counts_and_labels(self) -> None:
        from data.dataset import load_vsfc_split

        vi_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        expected_counts = {"train": 11426, "validation": 1583, "test": 3166}
        for split_name, expected_n in expected_counts.items():
            texts, sents, tops = load_vsfc_split(
                split_name, data_dir=vi_dir, word_segment=False, auto_download=False
            )
            self.assertEqual(len(texts), expected_n)
            self.assertEqual(len(sents), expected_n)
            self.assertEqual(len(tops), expected_n)
            self.assertTrue(all(isinstance(t, str) and len(t.strip()) > 0 for t in texts))
            self.assertEqual(set(sents), {0, 1, 2})
            self.assertEqual(set(tops), {0, 1, 2, 3})

    def test_f2_label_constants_contract(self) -> None:
        from data.dataset import SENTIMENT_LABELS, SENTIMENT_MAP, TOPIC_LABELS, TOPIC_MAP

        self.assertEqual(SENTIMENT_LABELS, ["negative", "neutral", "positive"])
        self.assertEqual(TOPIC_LABELS, ["lecturer", "training_program", "facility", "others"])
        self.assertEqual(SENTIMENT_MAP, {0: "negative", 1: "neutral", 2: "positive"})
        self.assertEqual(TOPIC_MAP, {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"})

    def test_f2_language_en_disables_pyvi_segmentation(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "validation.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("sentence,sentiment,topic\n")
                f.write('"The lecturer explains concepts very clearly",2,0\n')
                f.write('"Campus Wi-Fi and air conditioning are broken",0,2\n')
            texts, sents, tops = load_vsfc_split(
                "validation", data_dir=tmp, language="en", auto_download=False
            )
            self.assertEqual(len(texts), 2)
            self.assertEqual(sents, [2, 0])
            self.assertEqual(tops, [0, 2])
            self.assertNotIn("The_lecturer", texts[0])
            self.assertIn("The lecturer explains concepts very clearly", texts[0])

    def test_f2_language_vi_enables_pyvi_segmentation(self) -> None:
        from data.dataset import load_vsfc_split

        vi_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        seg_texts, _, _ = load_vsfc_split(
            "validation", data_dir=vi_dir, word_segment=True, auto_download=False
        )
        raw_texts, _, _ = load_vsfc_split(
            "validation", data_dir=vi_dir, word_segment=False, auto_download=False
        )
        sample_seg = seg_texts[:100]
        underscored = sum(1 for t in sample_seg if "_" in t)
        self.assertGreater(underscored, 80, "Expected PyVi word segmentation underscores in VI text")
        self.assertEqual(len(seg_texts), len(raw_texts))

    def test_f2_vsfc_dataset_and_get_dataloaders_contract(self) -> None:
        from data.dataset import get_dataloaders

        vi_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            tok = load_local_tokenizer(ckpt_dir)
            train_dl, val_dl, test_dl = get_dataloaders(
                tokenizer=tok,
                batch_size=8,
                max_length=32,
                data_dir=vi_dir,
                word_segment=False,
                quick_mode=True,
            )
            for dl in (train_dl, val_dl, test_dl):
                batch = next(iter(dl))
                for key in ("input_ids", "attention_mask", "sentiment_labels", "topic_labels"):
                    self.assertIn(key, batch)
                    self.assertEqual(batch[key].dtype, torch.long)

    def test_f2_get_dataloaders_accepts_language_and_fast_tokenizer(self) -> None:
        require_en_deliverables(self)
        from data.dataset import get_dataloaders

        with tempfile.TemporaryDirectory() as tmp:
            en_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            ckpt_dir = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            train_dl, val_dl, test_dl = get_dataloaders(
                tokenizer=ckpt_dir,
                batch_size=4,
                max_length=32,
                data_dir=en_dir,
                language="en",
            )
            batch = next(iter(train_dl))
            self.assertEqual(batch["input_ids"].ndim, 2)
            self.assertEqual(len(val_dl.dataset), 12)
            self.assertEqual(len(test_dl.dataset), 12)


class TestF3ModelFixesAndRouting(unittest.TestCase):
    """F3: Model R1 Fixes & Language Routing (`models/`)."""

    def test_f3_01_detect_language_vi_and_en(self) -> None:
        require_en_deliverables(self)
        detect_language = resolve_detect_language()
        vi_samples = [
            "Thầy dạy rất nhiệt tình và dễ hiểu.",
            "Phòng máy tính hư hỏng nhiều, máy lạnh rò rỉ nước.",
            "Chương trình đào tạo nặng lý thuyết.",
            "Học phí tăng cao nhưng thủ tục hành chính chậm.",
            "Bài giảng sinh động, giảng viên tận tâm.",
        ]
        en_samples = [
            "The lecturer explains concepts clearly and patiently.",
            "The classroom air conditioner is broken and Wi-Fi drops.",
            "Course assignments are practical and well organized.",
            "Tuition keeps rising while administrative service is slow.",
            "The syllabus follows the schedule closely.",
        ]
        for txt in vi_samples:
            self.assertEqual(detect_language(txt), "vi", f"Expected 'vi' for: {txt}")
        for txt in en_samples:
            self.assertEqual(detect_language(txt), "en", f"Expected 'en' for: {txt}")

    def test_f3_02_baseline_default_word_segment_metadata(self) -> None:
        vi_model = get_cached_baseline(
            os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        )
        self.assertTrue(getattr(vi_model, "default_word_segment", True))
        require_en_deliverables(self)
        en_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        self.assertTrue(os.path.isfile(en_path), "Missing saved_models/baseline_en.joblib")
        en_model = get_cached_baseline(en_path)
        self.assertIs(getattr(en_model, "default_word_segment", None), False)
        self.assertEqual(getattr(en_model, "language", None), "en")

    def test_f3_03_transformer_metadata_and_tokenizer_none_offline(self) -> None:
        require_en_deliverables(self)
        old_hf = os.environ.get("HF_HUB_OFFLINE")
        old_tr = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            vi_ckpt = os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask")
            en_ckpt = os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
            self.assertTrue(os.path.isdir(en_ckpt), "Missing saved_models/english_multitask")

            with open(
                os.path.join(en_ckpt, "phobert_multitask_meta.json"), "r", encoding="utf-8"
            ) as f:
                en_meta = json.load(f)
            self.assertIs(en_meta.get("default_word_segment"), False)
            self.assertEqual(en_meta.get("language"), "en")

            vi_model, _ = get_cached_phobert(vi_ckpt)
            en_model, _ = get_cached_phobert(en_ckpt)
            vi_pred = vi_model.predict("Thầy dạy rất hay", tokenizer=None)
            en_pred = en_model.predict("The lecturer explains clearly.", tokenizer=None)
            assert_prediction_schema(self, vi_pred)
            assert_prediction_schema(self, en_pred)
        finally:
            if old_hf is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf
            if old_tr is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_tr

    def test_f3_04_bilingual_routing_both_tiers(self) -> None:
        require_en_deliverables(self)
        router_info = resolve_bilingual_router()
        detect_language = router_info["detect_language"]

        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))

        samples = [
            ("Thầy giảng bài rất tận tâm.", "vi"),
            ("The instructor is very supportive and clear.", "en"),
        ]
        for text, expected_lang in samples:
            lang = detect_language(text)
            self.assertEqual(lang, expected_lang)
            t1_pred = vi_base.predict(text) if lang == "vi" else en_base.predict(text)
            t2_pred = (
                vi_tr.predict(text, tokenizer=vi_tok)
                if lang == "vi"
                else en_tr.predict(text, tokenizer=en_tok)
            )
            assert_prediction_schema(self, t1_pred)
            assert_prediction_schema(self, t2_pred)

    def test_f3_05_schema_native_types_and_distribution_sums(self) -> None:
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))

        vi_texts = [
            "Thầy dạy rất hay và tận tình.",
            "Phòng học quá nóng, máy lạnh hư.",
        ]
        for model, kwargs in ((vi_base, {}), (vi_tr, {"tokenizer": vi_tok})):
            single = model.predict(vi_texts[0], **kwargs)
            assert_prediction_schema(self, single)
            batch = model.predict(vi_texts, **kwargs)
            self.assertIsInstance(batch, list)
            self.assertEqual(len(batch), 2)
            for item in batch:
                assert_prediction_schema(self, item)


class TestF4TrainingPipeline(unittest.TestCase):
    """F4: Training Pipeline R1 Fixes & EN CLI (`training/train_baseline.py`, `train_phobert.py`)."""

    def test_f4_train_baseline_cli_help_includes_bilingual_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "training", "train_baseline.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0)
        for flag in ("--data_dir", "--save_path", "--max_features"):
            self.assertIn(flag, proc.stdout)
        require_en_deliverables(self)
        self.assertIn("--language", proc.stdout)
        self.assertIn("--no_word_segment", proc.stdout)

    def test_f4_train_phobert_cli_help_includes_bilingual_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "training", "train_phobert.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0)
        for flag in ("--data_dir", "--save_dir", "--model_name", "--quick_run"):
            self.assertIn(flag, proc.stdout)
        require_en_deliverables(self)
        self.assertIn("--language", proc.stdout)
        self.assertIn("--no_word_segment", proc.stdout)

    def test_f4_train_baseline_vi_smoke_to_temp_dir(self) -> None:
        from models.baseline_model import VietnameseFeedbackBaseline
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            vi_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "vi_data"), lang="vi")
            out_path = os.path.join(tmp, "baseline_vi.joblib")
            model = train_baseline(
                data_dir=vi_dir,
                save_path=out_path,
                max_features=200,
                eval_val=True,
                eval_test=True,
                show_report=False,
            )
            self.assertTrue(os.path.isfile(out_path))
            loaded = VietnameseFeedbackBaseline.load(out_path)
            pred = loaded.predict("Thầy giảng bài rất dễ hiểu")
            assert_prediction_schema(self, pred)
            self.assertTrue(model.is_fitted)

    def test_f4_train_baseline_en_smoke_to_temp_dir(self) -> None:
        require_en_deliverables(self)
        from models.baseline_model import VietnameseFeedbackBaseline

        with tempfile.TemporaryDirectory() as tmp:
            en_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            out_path = os.path.join(tmp, "baseline_en.joblib")
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_baseline.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    en_dir,
                    "--save_path",
                    out_path,
                    "--max_features",
                    "500",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isfile(out_path))
            loaded = VietnameseFeedbackBaseline.load(out_path)
            self.assertIs(getattr(loaded, "default_word_segment", None), False)
            self.assertEqual(getattr(loaded, "language", None), "en")
            pred = loaded.predict("The lecturer explains clearly")
            assert_prediction_schema(self, pred)

    def test_f4_train_phobert_returns_best_model_and_saves_meta(self) -> None:
        require_en_deliverables(self)
        from training.train_phobert import train_phobert

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            hf_dir = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            save_dir = os.path.join(tmp, "ckpt_out")
            best_model, summary = train_phobert(
                epochs=1,
                batch_size=8,
                max_length=32,
                model_name=hf_dir,
                save_dir=save_dir,
                data_dir=data_dir,
                quick_run=True,
                no_amp=True,
                eval_test=True,
                language="en",
            )
            self.assertIsInstance(summary, dict)
            self.assertTrue(
                os.path.isfile(os.path.join(save_dir, "model.pt"))
                or os.path.isfile(os.path.join(save_dir, "pytorch_model.bin"))
            )
            self.assertTrue(os.path.isfile(os.path.join(save_dir, "phobert_multitask_meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(save_dir, "training_meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(save_dir, "history.json")))
            pred = best_model.predict("The lecturer is good", tokenizer=load_local_tokenizer(save_dir))
            assert_prediction_schema(self, pred)

    def test_f4_train_phobert_en_defaults_do_not_require_pyvi(self) -> None:
        require_en_deliverables(self)
        from training.train_phobert import train_phobert

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            hf_dir = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            save_dir = os.path.join(tmp, "ckpt_en")
            train_phobert(
                epochs=1,
                batch_size=8,
                max_length=32,
                model_name=hf_dir,
                save_dir=save_dir,
                data_dir=data_dir,
                quick_run=True,
                no_amp=True,
                eval_test=False,
                language="en",
            )
            with open(os.path.join(save_dir, "phobert_multitask_meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertIs(meta.get("default_word_segment"), False)
            self.assertEqual(meta.get("language"), "en")


class TestF5EvaluationFixes(unittest.TestCase):
    """F5: Evaluation R1 Fixes & Majority Baseline (`evaluation/evaluate.py`)."""

    def test_f5_evaluate_cli_help_exposes_language_flag(self) -> None:
        proc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0)
        for flag in ("--data_dir", "--output_dir", "--quick_eval", "--no_save"):
            self.assertIn(flag, proc.stdout)
        require_en_deliverables(self)
        self.assertIn("--language", proc.stdout)

    def test_f5_dynamic_dataset_distribution_in_json(self) -> None:
        require_en_deliverables(self)
        from evaluation.evaluate import run_evaluation
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en", n_test=12)
            base_path = os.path.join(tmp, "base.joblib")
            train_baseline(
                data_dir=data_dir,
                save_path=base_path,
                max_features=300,
                eval_val=False,
                eval_test=False,
                language="en",
            )
            out_dir = os.path.join(tmp, "out")
            run_evaluation(
                data_dir=data_dir,
                split="test",
                baseline_path=base_path,
                phobert_path=os.path.join(tmp, "no_phobert"),
                output_dir=out_dir,
                language="en",
                device="cpu",
            )
            json_path = os.path.join(out_dir, "metrics_en.json")
            self.assertTrue(os.path.isfile(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ds = data["metadata"]["dataset"]
            self.assertEqual(ds["sample_count"], 12)
            self.assertEqual(ds["sentiment_distribution"], {"negative": 4, "neutral": 4, "positive": 4})
            self.assertEqual(
                ds["topic_distribution"],
                {"lecturer": 3, "training_program": 3, "facility": 3, "others": 3},
            )

    def test_f5_majority_class_baseline_accuracy_and_macro_f1(self) -> None:
        require_en_deliverables(self)
        from evaluation.evaluate import run_evaluation
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en", n_test=12)
            # Rewrite test.csv with imbalanced labels: 8 positive (2), 2 negative (0), 2 neutral (1)
            # and topics: 9 lecturer (0), 1 training_program (1), 1 facility (2), 1 others (3)
            with open(os.path.join(data_dir, "test.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("sentence,sentiment,topic\n")
                for i in range(12):
                    s = 2 if i < 8 else (0 if i < 10 else 1)
                    t = 0 if i < 9 else (i - 8)
                    f.write(f'"Sample feedback sentence number {i}",{s},{t}\n')
            base_path = os.path.join(tmp, "base.joblib")
            train_baseline(
                data_dir=data_dir,
                save_path=base_path,
                max_features=300,
                eval_val=False,
                eval_test=False,
                language="en",
            )
            out_dir = os.path.join(tmp, "out")
            run_evaluation(
                data_dir=data_dir,
                split="test",
                baseline_path=base_path,
                phobert_path=os.path.join(tmp, "no_phobert"),
                output_dir=out_dir,
                language="en",
                device="cpu",
            )
            with open(os.path.join(out_dir, "metrics_en.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            maj = extract_majority_baseline_metrics(data)
            self.assertIsNotNone(maj, "Expected majority_baseline metrics in metrics_en.json")
            exp_s_acc, exp_s_f1 = compute_majority_baseline([2] * 8 + [0] * 2 + [1] * 2, 3)
            exp_t_acc, exp_t_f1 = compute_majority_baseline([0] * 9 + [1, 2, 3], 4)
            self.assertAlmostEqual(maj["sentiment_accuracy"], exp_s_acc, places=4)
            self.assertAlmostEqual(maj["sentiment_macro_f1"], exp_s_f1, places=4)
            self.assertAlmostEqual(maj["topic_accuracy"], exp_t_acc, places=4)
            self.assertAlmostEqual(maj["topic_macro_f1"], exp_t_f1, places=4)

    def test_f5_markdown_report_contains_required_sections(self) -> None:
        from evaluation.evaluate import compute_comparisons, generate_markdown_report

        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), "r", encoding="utf-8") as f:
            ref = json.load(f)
        b_m = ref["baseline"]
        p_m = ref["phobert"]
        comp = compute_comparisons(b_m, p_m)
        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "summary.md")
            generate_markdown_report(
                baseline_metrics=b_m,
                phobert_metrics=p_m,
                comparisons=comp,
                split_name="test",
                num_samples=3166,
                device_name="CPU",
                output_path=md_path,
            )
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("Sentiment Accuracy", text)
            self.assertIn("Topic Accuracy", text)
            self.assertIn("Sentiment Classification", text)
            self.assertIn("Topic Classification", text)

    def test_f5_language_en_outputs_separate_files(self) -> None:
        require_en_deliverables(self)
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            base_path = os.path.join(tmp, "base_en.joblib")
            train_baseline(
                data_dir=data_dir,
                save_path=base_path,
                max_features=300,
                eval_val=False,
                eval_test=False,
                language="en",
            )
            out_dir = os.path.join(tmp, "results_tmp")
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    data_dir,
                    "--baseline_path",
                    base_path,
                    "--phobert_path",
                    os.path.join(tmp, "no_phobert"),
                    "--output_dir",
                    out_dir,
                    "--device",
                    "cpu",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "metrics_en.json")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "comparison_summary_en.md")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "metrics.json")))

    def test_f5_evaluation_quick_eval_no_save_vi(self) -> None:
        vi_metrics_path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
        before_mtime = os.path.getmtime(vi_metrics_path)
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                "--quick_eval",
                "--no_save",
                "--device",
                "cpu",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        after_mtime = os.path.getmtime(vi_metrics_path)
        self.assertEqual(before_mtime, after_mtime)


class TestF6EnglishDataset(unittest.TestCase):
    """F6: English Dataset Preparation (`data/english_feedback/`)."""

    def test_f6_english_feedback_files_exist(self) -> None:
        require_en_deliverables(self)
        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        self.assertTrue(os.path.isdir(en_dir), "Missing data/english_feedback/")
        self.assertTrue(
            os.path.isfile(os.path.join(PROJECT_ROOT, "data", "prepare_english_data.py")),
            "Missing data/prepare_english_data.py",
        )
        for split in ("train", "validation", "test"):
            self.assertIsNotNone(
                find_en_dataset_split_path(en_dir, split),
                f"Missing {split}.parquet/.csv in data/english_feedback/",
            )

    def test_f6_english_splits_schema_and_nonempty_text(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        min_sizes = {"train": 1000, "validation": 100, "test": 500}
        for split, min_n in min_sizes.items():
            texts, sents, tops = load_vsfc_split(
                split, data_dir=en_dir, language="en", auto_download=False
            )
            self.assertGreaterEqual(len(texts), min_n)
            self.assertEqual(len(texts), len(sents))
            self.assertEqual(len(texts), len(tops))
            self.assertTrue(all(isinstance(t, str) and len(t.strip()) > 0 for t in texts))

    def test_f6_english_label_bounds_and_all_classes_present(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            _, sents, tops = load_vsfc_split(
                split, data_dir=en_dir, language="en", auto_download=False
            )
            self.assertEqual(set(sents), {0, 1, 2}, f"Missing sentiment class in {split}")
            self.assertEqual(set(tops), {0, 1, 2, 3}, f"Missing topic class in {split}")

    def test_f6_zero_cross_split_leakage_english(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        split_sets: Dict[str, set] = {}
        for split in ("train", "validation", "test"):
            texts, _, _ = load_vsfc_split(
                split, data_dir=en_dir, language="en", auto_download=False
            )
            split_sets[split] = {t.strip().lower() for t in texts}
        self.assertEqual(len(split_sets["train"] & split_sets["validation"]), 0)
        self.assertEqual(len(split_sets["train"] & split_sets["test"]), 0)
        self.assertEqual(len(split_sets["validation"] & split_sets["test"]), 0)

    def test_f6_english_text_quality_no_translation_placeholders(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            texts, _, _ = load_vsfc_split(
                split, data_dir=en_dir, language="en", auto_download=False
            )
            bad = [t for t in texts if "wzjwz" in t.lower() or "&#" in t or "&amp;" in t]
            self.assertEqual(len(bad), 0, f"Found raw artifacts in EN {split}: {bad[:3]}")

    def test_f6_curated_native_english_eval_slice_valid(self) -> None:
        require_en_deliverables(self)
        import pandas as pd
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        curated_path = os.path.join(en_dir, "test_curated.csv")
        train_texts, _, _ = load_vsfc_split(
            "train", data_dir=en_dir, language="en", auto_download=False
        )
        train_norm = {t.strip().lower() for t in train_texts}

        if os.path.isfile(curated_path):
            df = pd.read_csv(curated_path)
            self.assertTrue({"sentence", "sentiment", "topic"}.issubset(set(df.columns)))
            self.assertTrue(set(df["sentiment"].astype(int)).issubset({0, 1, 2}))
            self.assertTrue(set(df["topic"].astype(int)).issubset({0, 1, 2, 3}))
            cur_norm = {str(s).strip().lower() for s in df["sentence"].tolist()}
            self.assertEqual(len(cur_norm & train_norm), 0)
        else:
            test_texts, test_sents, test_tops = load_vsfc_split(
                "test", data_dir=en_dir, language="en", auto_download=False
            )
            slice_texts = test_texts[:100]
            self.assertEqual(len(slice_texts), min(100, len(test_texts)))
            self.assertTrue(all("wzjwz" not in t.lower() for t in slice_texts))
            self.assertTrue(all(not has_vi_diacritics(t) for t in slice_texts))
            self.assertTrue(set(test_sents[:100]).issubset({0, 1, 2}))
            self.assertTrue(set(test_tops[:100]).issubset({0, 1, 2, 3}))


class TestF7EnglishTier1Model(unittest.TestCase):
    """F7: English Tier 1 Training (`saved_models/baseline_en.joblib`)."""

    def test_f7_01_artifact_exists_and_loads(self) -> None:
        require_en_deliverables(self)
        path = os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        self.assertTrue(os.path.isfile(path), "Missing saved_models/baseline_en.joblib")
        self.assertGreater(os.path.getsize(path), 50 * 1024)
        model = get_cached_baseline(path)
        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.sentiment_clf.classes_), 3)
        self.assertEqual(len(model.topic_clf.classes_), 4)

    def test_f7_02_all_sentiments_and_topics_realistic_en(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(
            os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        )
        canonical = [
            ("The professor explains concepts clearly and is always supportive.", "positive", "lecturer"),
            ("The classroom air conditioner is broken and Wi-Fi keeps dropping.", "negative", "facility"),
            ("The curriculum schedule is generally moderate and acceptable.", None, None),
            ("Tuition increased while administrative paperwork remains slow.", None, None),
        ]
        for text, exp_s, exp_t in canonical:
            pred = model.predict(text)
            assert_prediction_schema(self, pred)
            if exp_s is not None:
                self.assertEqual(pred["sentiment"]["label"], exp_s, f"Text: {text}")
            if exp_t is not None:
                self.assertEqual(pred["topic"]["label"], exp_t, f"Text: {text}")

    def test_f7_03_predict_proba_and_predict_labels_parity(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(
            os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        )
        batch = [
            "The lecturer is very patient and explains clearly.",
            "The projector in room B102 is broken.",
            "The syllabus is okay overall.",
        ]
        s_probs, t_probs = model.predict_proba(batch)
        self.assertEqual(s_probs.shape, (3, 3))
        self.assertEqual(t_probs.shape, (3, 4))
        s_ids, t_ids = model.predict_labels(batch)
        preds = model.predict(batch)
        for i in range(3):
            self.assertAlmostEqual(float(s_probs[i].sum()), 1.0, places=5)
            self.assertAlmostEqual(float(t_probs[i].sum()), 1.0, places=5)
            self.assertEqual(s_ids[i], int(s_probs[i].argmax()))
            self.assertEqual(t_ids[i], int(t_probs[i].argmax()))
            self.assertEqual(preds[i]["sentiment"]["label_id"], s_ids[i])
            self.assertEqual(preds[i]["topic"]["label_id"], t_ids[i])

    def test_f7_04_vi_baseline_no_regression(self) -> None:
        model = get_cached_baseline(
            os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        )
        p1 = model.predict("Thầy cô giảng dạy rất tận tình và chu đáo")
        p2 = model.predict("Cơ sở vật chất phòng máy quá cũ kỹ, máy lạnh thường xuyên hư")
        assert_prediction_schema(self, p1)
        assert_prediction_schema(self, p2)
        self.assertEqual(p1["sentiment"]["label"], "positive")
        self.assertEqual(p1["topic"]["label"], "lecturer")
        self.assertEqual(p2["sentiment"]["label"], "negative")
        self.assertEqual(p2["topic"]["label"], "facility")

    def test_f7_05_en_baseline_cpu_latency(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(
            os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        )
        batch = ["The lecturer explains the course material very clearly."] * 100
        t0 = time.perf_counter()
        preds = model.predict(batch)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(preds), 100)
        self.assertLess(elapsed, 1.0)


class TestF8EnglishTier2Model(unittest.TestCase):
    """F8: English Tier 2 Training (`saved_models/english_multitask/`)."""

    def test_f8_01_checkpoint_files_present(self) -> None:
        require_en_deliverables(self)
        ckpt = os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
        self.assertTrue(os.path.isdir(ckpt), "Missing saved_models/english_multitask/")
        self.assertTrue(
            os.path.isfile(os.path.join(ckpt, "model.pt"))
            or os.path.isfile(os.path.join(ckpt, "pytorch_model.bin"))
        )
        for req in ("config.json", "phobert_multitask_meta.json", "training_meta.json"):
            self.assertTrue(os.path.isfile(os.path.join(ckpt, req)), f"Missing {req}")

    def test_f8_02_offline_load_and_infer(self) -> None:
        require_en_deliverables(self)
        ckpt = os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
        model, tok = get_cached_phobert(ckpt)
        pred = model.predict("The professor explains concepts very clearly.", tokenizer=tok)
        assert_prediction_schema(self, pred)

    def test_f8_03_all_sentiments_and_topics_realistic_en(self) -> None:
        require_en_deliverables(self)
        model, tok = get_cached_phobert(
            os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
        )
        cases = [
            ("The professor explains concepts clearly and is always supportive.", "positive", "lecturer"),
            ("The classroom air conditioner is broken and Wi-Fi keeps dropping.", "negative", "facility"),
            ("The course curriculum is practical and strengthens engineering skills.", "positive", "training_program"),
            ("Tuition fees are rising and admin paperwork takes too long.", None, None),
        ]
        for text, exp_s, exp_t in cases:
            pred = model.predict(text, tokenizer=tok)
            assert_prediction_schema(self, pred)
            if exp_s is not None:
                self.assertEqual(pred["sentiment"]["label"], exp_s, f"Text: {text}")
            if exp_t is not None:
                self.assertEqual(pred["topic"]["label"], exp_t, f"Text: {text}")

    def test_f8_04_predict_proba_and_predict_labels_parity(self) -> None:
        require_en_deliverables(self)
        model, tok = get_cached_phobert(
            os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
        )
        batch = [
            "The lecturer is very patient and explains clearly.",
            "The projector in room B102 is broken.",
            "The syllabus is okay overall.",
        ]
        s_probs, t_probs = model.predict_proba(batch, tokenizer=tok)
        self.assertEqual(s_probs.shape, (3, 3))
        self.assertEqual(t_probs.shape, (3, 4))
        s_ids, t_ids = model.predict_labels(batch, tokenizer=tok)
        preds = model.predict(batch, tokenizer=tok)
        for i in range(3):
            self.assertAlmostEqual(float(s_probs[i].sum()), 1.0, places=4)
            self.assertAlmostEqual(float(t_probs[i].sum()), 1.0, places=4)
            self.assertEqual(s_ids[i], int(s_probs[i].argmax()))
            self.assertEqual(t_ids[i], int(t_probs[i].argmax()))
            self.assertEqual(preds[i]["sentiment"]["label_id"], s_ids[i])
            self.assertEqual(preds[i]["topic"]["label_id"], t_ids[i])

    def test_f8_05_vi_phobert_no_regression_offline(self) -> None:
        model, tok = get_cached_phobert(
            os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask")
        )
        p1 = model.predict("Thầy dạy rất nhiệt tình và dễ hiểu", tokenizer=tok, word_segment=True)
        p2 = model.predict(
            "Phòng máy tính hư hỏng nhiều, máy lạnh bị rò rỉ nước",
            tokenizer=tok,
            word_segment=True,
        )
        assert_prediction_schema(self, p1)
        assert_prediction_schema(self, p2)
        self.assertEqual(p1["sentiment"]["label"], "positive")
        self.assertEqual(p1["topic"]["label"], "lecturer")
        self.assertEqual(p2["sentiment"]["label"], "negative")
        self.assertEqual(p2["topic"]["label"], "facility")


class TestF9DualEvaluationReports(unittest.TestCase):
    """F9: Dual VI + EN Evaluation (`results/metrics*.json`, `comparison_summary*.md`)."""

    def test_f9_vi_metrics_json_schema_and_no_regression(self) -> None:
        path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["num_samples"], 3166)
        self.assertEqual(data["split"], "test")
        b = data["models"]["baseline"]
        p = data["models"]["phobert"]
        tol = 1e-4
        self.assertGreaterEqual(b["sentiment_accuracy"], 0.86734049 - tol)
        self.assertGreaterEqual(b["sentiment_macro_f1"], 0.71897139 - tol)
        self.assertGreaterEqual(b["topic_accuracy"], 0.80985471 - tol)
        self.assertGreaterEqual(b["topic_macro_f1"], 0.72716562 - tol)

        self.assertGreaterEqual(p["sentiment_accuracy"], 0.93493367 - tol)
        self.assertGreaterEqual(p["sentiment_macro_f1"], 0.82888945 - tol)
        self.assertGreaterEqual(p["topic_accuracy"], 0.88408086 - tol)
        self.assertGreaterEqual(p["topic_macro_f1"], 0.80103109 - tol)

    def test_f9_vi_markdown_matches_metrics_json(self) -> None:
        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        with open(os.path.join(PROJECT_ROOT, "results", "comparison_summary.md"), "r", encoding="utf-8") as f:
            md = f.read()
        b = m["models"]["baseline"]
        p = m["models"]["phobert"]
        for val in (
            b["sentiment_accuracy"],
            b["sentiment_macro_f1"],
            b["topic_accuracy"],
            b["topic_macro_f1"],
            b["mean_macro_f1"],
            p["sentiment_accuracy"],
            p["sentiment_macro_f1"],
            p["topic_accuracy"],
            p["topic_macro_f1"],
            p["mean_macro_f1"],
        ):
            self.assertIn(f"{val * 100:.2f}%", md)

    def test_f9_en_metrics_json_exists_and_valid_schema(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_json = os.path.join(PROJECT_ROOT, "results", "metrics_en.json")
        self.assertTrue(os.path.isfile(en_json), "Missing results/metrics_en.json")
        with open(en_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts, _, _ = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        self.assertEqual(data["num_samples"], len(texts))
        self.assertIn("baseline", data["models"])
        self.assertIn("phobert", data["models"])
        self.assertIsNotNone(extract_majority_baseline_metrics(data))

    def test_f9_en_models_beat_majority_baseline_macro_f1(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        _, sents, tops = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        _, maj_s_f1 = compute_majority_baseline(sents, 3)
        _, maj_t_f1 = compute_majority_baseline(tops, 4)

        with open(os.path.join(PROJECT_ROOT, "results", "metrics_en.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for tier_key in ("baseline", "phobert"):
            m = data["models"][tier_key]
            self.assertGreater(
                m["sentiment_macro_f1"],
                maj_s_f1 + 0.05,
                f"EN {tier_key} sentiment Macro-F1 failed to beat majority baseline {maj_s_f1:.4f}",
            )
            self.assertGreater(
                m["topic_macro_f1"],
                maj_t_f1 + 0.05,
                f"EN {tier_key} topic Macro-F1 failed to beat majority baseline {maj_t_f1:.4f}",
            )

    def test_f9_en_markdown_matches_metrics_en_json(self) -> None:
        require_en_deliverables(self)
        en_json = os.path.join(PROJECT_ROOT, "results", "metrics_en.json")
        en_md = os.path.join(PROJECT_ROOT, "results", "comparison_summary_en.md")
        self.assertTrue(os.path.isfile(en_md), "Missing results/comparison_summary_en.md")
        with open(en_json, "r", encoding="utf-8") as f:
            m = json.load(f)
        with open(en_md, "r", encoding="utf-8") as f:
            md = f.read()
        b = m["models"]["baseline"]
        p = m["models"]["phobert"]
        for val in (
            b["sentiment_accuracy"],
            b["sentiment_macro_f1"],
            b["topic_accuracy"],
            b["topic_macro_f1"],
            p["sentiment_accuracy"],
            p["sentiment_macro_f1"],
            p["topic_accuracy"],
            p["topic_macro_f1"],
        ):
            self.assertIn(f"{val * 100:.2f}%", md)

    def test_f9_live_reproducible_eval_matches_saved_reports(self) -> None:
        from data.dataset import load_vsfc_split
        from evaluation.evaluate import evaluate_baseline, evaluate_phobert

        texts, sents, tops = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc"),
            word_segment=True,
            auto_download=False,
        )
        sub_t, sub_s, sub_top = texts[:30], sents[:30], tops[:30]
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        r1 = evaluate_baseline(vi_base, sub_t, sub_s, sub_top, batch_size=30)
        r2 = evaluate_baseline(vi_base, sub_t, sub_s, sub_top, batch_size=30)
        self.assertEqual(r1["sentiment_accuracy"], r2["sentiment_accuracy"])
        p1 = evaluate_phobert(vi_tr, vi_tok, sub_t, sub_s, sub_top, device=torch.device("cpu"), batch_size=15)
        self.assertGreater(p1["sentiment_accuracy"], 0.7)


class TestF10DualSampleLibrariesAppTest(unittest.TestCase):
    """F10: Dual Sample Libraries & UI Fixes in `app/app.py` (unmocked AppTest)."""

    def test_f10_vi_presets_preserved_verbatim(self) -> None:
        import app.app as demo_app

        presets = demo_app.get_presets()
        self.assertEqual(len(presets), 10)
        actual = tuple(
            (p["id"], p["text"], p["expected_sentiment"], p["expected_topic"])
            for p in presets
        )
        self.assertEqual(actual, EXPECTED_VI_PRESETS_SNAPSHOT)

    def test_f10_en_presets_diverse_coverage(self) -> None:
        require_en_deliverables(self)
        import app.app as demo_app

        fn = getattr(demo_app, "get_english_presets", None)
        self.assertTrue(callable(fn), "M3 deliverable missing: app/app.py:get_english_presets()")
        en_presets = fn()
        self.assertGreaterEqual(len(en_presets), 10)
        seen_ids = set()
        seen_texts = set()
        sentiments = set()
        topics = set()
        for item in en_presets:
            for key in ("id", "title", "text", "expected_sentiment", "expected_topic", "note"):
                self.assertIn(key, item)
                self.assertTrue(str(item[key]).strip())
            self.assertFalse(has_vi_diacritics(item["text"]), f"EN preset contains VI diacritics: {item['text']}")
            seen_ids.add(item["id"])
            seen_texts.add(item["text"].strip())
            sentiments.add(item["expected_sentiment"])
            topics.add(item["expected_topic"])
        self.assertEqual(len(seen_ids), len(en_presets))
        self.assertEqual(len(seen_texts), len(en_presets))
        self.assertEqual(sentiments, {"negative", "neutral", "positive"})
        self.assertEqual(topics, {"lecturer", "training_program", "facility", "others"})

    def test_f10_sidebar_has_two_distinct_libraries(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        self.assertFalse(at.exception, f"AppTest exception: {at.exception}")
        self.assertGreaterEqual(len(at.sidebar.selectbox), 2)
        self.assertGreaterEqual(len(at.sidebar.button), 2)
        vi_opts = list(at.sidebar.selectbox[0].options)
        en_opts = list(at.sidebar.selectbox[1].options)
        self.assertEqual(len(vi_opts), 10)
        self.assertGreaterEqual(len(en_opts), 10)
        self.assertEqual(len(set(vi_opts) & set(en_opts)), 0)

    def test_f10_load_vi_preset_populates_user_text(self) -> None:
        from streamlit.testing.v1 import AppTest
        import app.app as demo_app

        vi_presets = demo_app.get_presets()
        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        self.assertFalse(at.exception)
        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[1]).run()
        at.sidebar.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_area[0].value, vi_presets[1]["text"])
        self.assertEqual(at.session_state["user_text"], vi_presets[1]["text"])

    def test_f10_load_en_preset_populates_user_text(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest
        import app.app as demo_app

        en_presets = demo_app.get_english_presets()
        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        self.assertFalse(at.exception)
        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[0]).run()
        at.sidebar.button[1].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_area[0].value, en_presets[0]["text"])
        self.assertEqual(at.session_state["user_text"], en_presets[0]["text"])

    def test_f10_vi_inference_all_three_modes(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[0]).run()
        at.sidebar.button[0].click().run()
        for tier, expected_bars in (("baseline", 7), ("transformer", 7), ("both", 14)):
            with self.subTest(tier=tier):
                at = select_model_tier(at, tier)
                analyze_btn = find_main_button(at, ("Phân tích", "Analyze"))
                at = analyze_btn.click().run()
                self.assertFalse(at.exception, f"Exception in tier={tier}: {at.exception}")
                self.assertEqual(len(at.progress), expected_bars)
                for p in at.progress:
                    self.assertGreaterEqual(p.value, 0)
                    self.assertLessEqual(p.value, 100)

    def test_f10_en_inference_all_three_modes(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[0]).run()
        at.sidebar.button[1].click().run()
        for tier, expected_bars in (("baseline", 7), ("transformer", 7), ("both", 14)):
            with self.subTest(tier=tier):
                at = select_model_tier(at, tier)
                analyze_btn = find_main_button(at, ("Phân tích", "Analyze"))
                at = analyze_btn.click().run()
                self.assertFalse(at.exception, f"Exception in EN tier={tier}: {at.exception}")
                self.assertEqual(len(at.progress), expected_bars)
                for p in at.progress:
                    self.assertGreaterEqual(p.value, 0)
                    self.assertLessEqual(p.value, 100)


class TestF11UnitSuiteIntegrity(unittest.TestCase):
    """F11: Existing & Bilingual Unit/Integration Suite (`tests/`)."""

    def test_f11_existing_unit_test_count_preserved(self) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
            self.assertGreaterEqual(suite.countTestCases(), 169)
        finally:
            os.chdir(old_cwd)

    def test_f11_original_twelve_test_files_intact(self) -> None:
        tests_dir = os.path.join(PROJECT_ROOT, "tests")
        for fname in ORIGINAL_TWELVE_TEST_FILES:
            fpath = os.path.join(tests_dir, fname)
            self.assertTrue(os.path.isfile(fpath), f"Missing test file: {fname}")
            self.assertGreater(os.path.getsize(fpath), 100)

    def test_f11_bilingual_tests_added_in_tests_dir(self) -> None:
        require_en_deliverables(self)
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
            self.assertGreater(
                suite.countTestCases(),
                169,
                "Expected additional bilingual tests in tests/ (> 169)",
            )
        finally:
            os.chdir(old_cwd)

    def test_f11_tests_directory_passes_cleanly(self) -> None:
        require_en_deliverables(self)
        res = run_unit_test_suite_cached()
        self.assertEqual(
            res["returncode"],
            0,
            f"tests/ suite failed (rc={res['returncode']}):\n{res['stderr'][-2000:]}",
        )

    def test_f11_e2e_tests_isolated_from_unit_test_discovery(self) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
            flattened = repr(suite)
            self.assertNotIn("e2e_tests", flattened)
        finally:
            os.chdir(old_cwd)
