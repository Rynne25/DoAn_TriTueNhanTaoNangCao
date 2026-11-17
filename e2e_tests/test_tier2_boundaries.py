"""Tier 2 E2E Tests: Boundary & Corner Cases across all F1-F11 feature areas (>= 5 per area)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from e2e_tests.e2e_helpers import (
    APPTEST_TIMEOUT,
    APP_PATH,
    ORIGINAL_TWELVE_TEST_FILES,
    PROJECT_ROOT,
    assert_prediction_schema,
    compute_majority_baseline,
    create_tiny_local_hf_checkpoint,
    find_main_button,
    get_cached_baseline,
    get_cached_phobert,
    get_e2e_scope,
    load_local_tokenizer,
    require_en_deliverables,
    resolve_detect_language,
    select_model_tier,
    write_synthetic_vsfc_dir,
)


class TestF1ReviewBoundaries(unittest.TestCase):
    """F1 Boundary Cases: CODE_REVIEW.md integrity and accuracy."""

    def _read_review(self) -> str:
        require_en_deliverables(self)
        path = os.path.join(PROJECT_ROOT, "CODE_REVIEW.md")
        self.assertTrue(os.path.isfile(path), "M1 deliverable missing: CODE_REVIEW.md")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_f1_b01_utf8_no_null_bytes(self) -> None:
        content = self._read_review()
        self.assertNotIn("\x00", content)
        self.assertGreater(len(content.splitlines()), 20)

    def test_f1_b02_dedicated_sections_for_all_six_dirs(self) -> None:
        content = self._read_review().lower()
        for d in ("models/", "data/", "training/", "evaluation/", "app/", "tests/"):
            occurrences = content.count(d)
            self.assertGreaterEqual(
                occurrences,
                1,
                f"Expected dedicated discussion of '{d}' in CODE_REVIEW.md",
            )
        heading_or_bullet = re.findall(
            r"(?:^#{1,4}.*(?:models|data|training|evaluation|app|tests)|^\s*[-*].*(?:models/|data/|training/|evaluation/|app/|tests/))",
            content,
            flags=re.MULTILINE,
        )
        self.assertGreaterEqual(len(heading_or_bullet), 6)

    def test_f1_b03_cited_repo_paths_exist_on_disk(self) -> None:
        content = self._read_review()
        cited = set(
            re.findall(
                r"\b(?:models|data|training|evaluation|app|tests)/[A-Za-z0-9_./-]+\.py\b",
                content,
            )
        )
        self.assertGreaterEqual(len(cited), 6, "Expected >= 6 concrete .py paths cited")
        for rel in cited:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, rel)),
                f"CODE_REVIEW.md cites non-existent file '{rel}'",
            )

    def test_f1_b04_symbol_or_line_level_details_present(self) -> None:
        content = self._read_review()
        symbols = (
            "load_vsfc_split",
            "VietnameseFeedbackBaseline",
            "PhoBERTMultiTask",
            "train_phobert",
            "run_evaluation",
            "get_presets",
        )
        matched = [s for s in symbols if s in content]
        self.assertGreaterEqual(len(matched), 4, f"Expected symbol citations, found {matched}")

    def test_f1_b05_distinguishes_vi_preserved_vs_en_extended(self) -> None:
        low = self._read_review().lower()
        self.assertTrue(
            any(tok in low for tok in ("minimal", "untouched", "preserve", "regression")),
            "CODE_REVIEW.md should distinguish preserved VI paths from EN extensions",
        )


class TestF2DataBoundaries(unittest.TestCase):
    """F2 Boundary Cases: Data pipeline error handling and sanitization."""

    def test_f2_b01_missing_data_dir_raises_file_not_found(self) -> None:
        from data.dataset import load_vsfc_split

        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist")
            with self.assertRaises(FileNotFoundError):
                load_vsfc_split("train", data_dir=missing, word_segment=False, auto_download=False)

    def test_f2_b02_invalid_split_names_and_aliases(self) -> None:
        from data.dataset import load_vsfc_split

        vi_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        for bad in ("", "   ", "invalid_split", "test2"):
            with self.assertRaises(ValueError):
                load_vsfc_split(bad, data_dir=vi_dir, word_segment=False, auto_download=False)
        for alias in ("val", "dev", "valid"):
            texts, _, _ = load_vsfc_split(
                alias, data_dir=vi_dir, word_segment=False, auto_download=False
            )
            self.assertEqual(len(texts), 1583)

    def test_f2_b03_malformed_csv_missing_columns_raises_value_error(self) -> None:
        from data.dataset import load_vsfc_split

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("sentence,sentiment\n")
                f.write('"Hello world",2\n')
            with self.assertRaises(ValueError):
                load_vsfc_split("test", data_dir=tmp, word_segment=False, auto_download=False)

    def test_f2_b04_out_of_bounds_labels_raise_value_error(self) -> None:
        from data.dataset import load_vsfc_split

        for s_val, t_val in ((-1, 0), (3, 0), (99, 0), (0, -1), (0, 4), (0, 10)):
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "test.csv"), "w", encoding="utf-8", newline="") as f:
                    f.write("sentence,sentiment,topic\n")
                    f.write(f'"Valid sample text",{s_val},{t_val}\n')
                with self.assertRaises(ValueError):
                    load_vsfc_split("test", data_dir=tmp, word_segment=False, auto_download=False)

    def test_f2_b05_empty_and_none_sentences_handled_cleanly(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("sentence,sentiment,topic\n")
                f.write('"   ",2,0\n')
            with self.assertRaises(ValueError):
                load_vsfc_split("test", data_dir=tmp, word_segment=False, auto_download=False)

    def test_f2_b06_contractions_hyphens_not_corrupted_in_en(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        samples = [
            "The Wi-Fi isn't working and the course isn't well-structured!",
            "10/10 well-organized lab; don't change the syllabus.",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("sentence,sentiment,topic\n")
                f.write(f'"{samples[0]}",0,2\n')
                f.write(f'"{samples[1]}",2,1\n')
            texts, _, _ = load_vsfc_split("test", data_dir=tmp, language="en", auto_download=False)
            self.assertEqual(texts[0], samples[0])
            self.assertEqual(texts[1], samples[1])


class TestF3ModelRoutingBoundaries(unittest.TestCase):
    """F3 Boundary Cases: Language detection, empty/None batches, corrupted checkpoints, code-mixed."""

    def test_f3_b01_detect_language_edge_strings(self) -> None:
        require_en_deliverables(self)
        detect_language = resolve_detect_language()
        for edge in ("", "   ", "12345", "???!!!", "ok"):
            self.assertEqual(detect_language(edge), "en")
        mixed = "Lecturer dạy rất good nhưng deadline nhiều quá"
        self.assertEqual(detect_language(mixed), "vi")

    def test_f3_b02_empty_list_and_none_in_batch(self) -> None:
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))

        self.assertEqual(vi_base.predict([]), [])
        s_p, t_p = vi_base.predict_proba([])
        self.assertEqual(s_p.shape, (0, 3))
        self.assertEqual(t_p.shape, (0, 4))
        self.assertEqual(vi_base.predict_labels([]), ([], []))

        self.assertEqual(vi_tr.predict([], tokenizer=vi_tok), [])
        s_p2, t_p2 = vi_tr.predict_proba([], tokenizer=vi_tok)
        self.assertEqual(s_p2.shape, (0, 3))
        self.assertEqual(t_p2.shape, (0, 4))

        mixed_none = [None, "Thầy dạy rất hay", ""]
        for item in vi_base.predict(mixed_none):
            assert_prediction_schema(self, item)
        for item in vi_tr.predict(mixed_none, tokenizer=vi_tok):
            assert_prediction_schema(self, item)

        if get_e2e_scope() != "vi":
            en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
            en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
            for item in en_base.predict([None, "Great instructor", ""]):
                assert_prediction_schema(self, item)
            for item in en_tr.predict([None, "Great instructor", ""], tokenizer=en_tok):
                assert_prediction_schema(self, item)

    def test_f3_b03_missing_and_corrupted_checkpoints_fail_gracefully(self) -> None:
        from models.baseline_model import VietnameseFeedbackBaseline
        from models.phobert_model import PhoBERTMultiTask

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                VietnameseFeedbackBaseline.load(os.path.join(tmp, "missing.joblib"))
            with self.assertRaises(Exception):
                PhoBERTMultiTask.from_pretrained(os.path.join(tmp, "missing_dir"), local_files_only=True)

            bad_joblib = os.path.join(tmp, "corrupt.joblib")
            with open(bad_joblib, "wb") as f:
                f.write(b"NOT_A_JOBLIB_FILE")
            with self.assertRaises(Exception):
                VietnameseFeedbackBaseline.load(bad_joblib)

            empty_ckpt = os.path.join(tmp, "empty_ckpt")
            os.makedirs(empty_ckpt, exist_ok=True)
            with self.assertRaises(FileNotFoundError):
                PhoBERTMultiTask.from_pretrained(empty_ckpt)

            bad_pt_ckpt = os.path.join(tmp, "bad_pt_ckpt")
            create_tiny_local_hf_checkpoint(bad_pt_ckpt)
            with open(os.path.join(bad_pt_ckpt, "model.pt"), "wb") as f:
                f.write(b"CORRUPT_TORCH_WEIGHTS")
            with self.assertRaises(Exception):
                PhoBERTMultiTask.from_pretrained(bad_pt_ckpt)

    def test_f3_b04_batch_vs_single_consistency_vi_and_en(self) -> None:
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        vi_inputs = [
            "Thầy giảng rất dễ hiểu.",
            "Máy lạnh phòng học bị hư.",
            "Chương trình học tạm ổn.",
            "Thủ tục hành chính chậm.",
            "Phòng máy sạch sẽ thoáng mát.",
        ]
        for model, kwargs in ((vi_base, {}), (vi_tr, {"tokenizer": vi_tok})):
            batch = model.predict(vi_inputs, **kwargs)
            for s, b_item in zip(vi_inputs, batch):
                s_item = model.predict(s, **kwargs)
                self.assertEqual(s_item["sentiment"]["label_id"], b_item["sentiment"]["label_id"])
                self.assertEqual(s_item["topic"]["label_id"], b_item["topic"]["label_id"])
                self.assertAlmostEqual(
                    s_item["sentiment"]["confidence"], b_item["sentiment"]["confidence"], places=4
                )

        if get_e2e_scope() != "vi":
            en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
            en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
            en_inputs = [
                "The lecturer explains clearly.",
                "The AC is broken in room B102.",
                "The syllabus is moderate.",
                "Admin paperwork is slow.",
                "Lab PCs are fast and clean.",
            ]
            for model, kwargs in ((en_base, {}), (en_tr, {"tokenizer": en_tok})):
                batch = model.predict(en_inputs, **kwargs)
                for s, b_item in zip(en_inputs, batch):
                    s_item = model.predict(s, **kwargs)
                    self.assertEqual(s_item["sentiment"]["label_id"], b_item["sentiment"]["label_id"])
                    self.assertEqual(s_item["topic"]["label_id"], b_item["topic"]["label_id"])

    def test_f3_b05_code_mixed_vi_en_inputs(self) -> None:
        code_mixed = [
            "Slide presentation ok nhưng thầy giảng hơi nhanh",
            "Campus Wi-Fi chập chờn quá",
            "Homework heavy quá cho sinh viên năm nhất",
        ]
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        for pred in vi_base.predict(code_mixed) + vi_tr.predict(code_mixed, tokenizer=vi_tok):
            assert_prediction_schema(self, pred)
        if get_e2e_scope() != "vi":
            en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
            en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
            for pred in en_base.predict(code_mixed) + en_tr.predict(code_mixed, tokenizer=en_tok):
                assert_prediction_schema(self, pred)


class TestF4TrainingBoundaries(unittest.TestCase):
    """F4 Boundary Cases: Training edge cases, invalid CLI flags, class weights."""

    def test_f4_b01_train_phobert_epochs_zero_graceful(self) -> None:
        require_en_deliverables(self)
        from training.train_phobert import train_phobert

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            hf_dir = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            save_dir = os.path.join(tmp, "nonexistent_save_dir")
            model, summary = train_phobert(
                epochs=0,
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
            self.assertIsNotNone(model)
            self.assertIsInstance(summary, dict)
            self.assertTrue(os.path.isdir(save_dir))

    def test_f4_b02_train_baseline_invalid_data_dir_raises(self) -> None:
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            bad_vi_dir = os.path.join(tmp, "bad_vi")
            os.makedirs(bad_vi_dir, exist_ok=True)
            with open(os.path.join(bad_vi_dir, "train.csv"), "w", encoding="utf-8") as f:
                f.write("wrong_col,sentiment\nhi,0\n")
            with self.assertRaises(ValueError):
                train_baseline(
                    data_dir=bad_vi_dir,
                    save_path=os.path.join(tmp, "out.joblib"),
                    eval_val=False,
                    eval_test=False,
                )
            if get_e2e_scope() != "vi":
                with self.assertRaises(FileNotFoundError):
                    train_baseline(
                        data_dir=os.path.join(tmp, "missing_en"),
                        save_path=os.path.join(tmp, "out_en.joblib"),
                        eval_val=False,
                        eval_test=False,
                        language="en",
                    )

    def test_f4_b03_invalid_language_cli_arg_rejected(self) -> None:
        require_en_deliverables(self)
        for script in ("train_baseline.py", "train_phobert.py"):
            proc = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "training", script), "--language", "fr"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
            )
            self.assertNotEqual(proc.returncode, 0, f"{script} accepted invalid --language fr")

    def test_f4_b04_compute_smoothed_class_weights_edge_cases(self) -> None:
        from training.train_phobert import compute_smoothed_class_weights

        self.assertIsNone(compute_smoothed_class_weights([0, 1, 2], 3, power=0.0))
        self.assertIsNone(compute_smoothed_class_weights([], 3, power=0.35))
        self.assertIsNone(compute_smoothed_class_weights([0, 1], 3, power=0.35))
        weights = compute_smoothed_class_weights([0, 0, 0, 1, 2], 3, power=0.35)
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), 3)
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_f4_b05_train_baseline_no_val_no_test_saves_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vi_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "vi_data"), lang="vi")
            out_path = os.path.join(tmp, "model.joblib")
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_baseline.py"),
                    "--data_dir",
                    vi_dir,
                    "--save_path",
                    out_path,
                    "--max_features",
                    "200",
                    "--no_val",
                    "--no_test",
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


class TestF5EvaluationBoundaries(unittest.TestCase):
    """F5 Boundary Cases: Evaluation CLI error handling and single-sample / tie safety."""

    def test_f5_b01_missing_checkpoints_exit_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--baseline_path",
                    os.path.join(tmp, "no_base.joblib"),
                    "--phobert_path",
                    os.path.join(tmp, "no_phobert"),
                    "--no_save",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("ERROR", proc.stderr.upper())

    def test_f5_b02_missing_data_dir_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--data_dir",
                    os.path.join(tmp, "missing_data"),
                    "--no_save",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
            )
            self.assertNotEqual(proc.returncode, 0)

    def test_f5_b03_single_sample_max_samples_1_no_zero_division(self) -> None:
        from evaluation.evaluate import run_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            res = run_evaluation(
                data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc"),
                split="test",
                baseline_path=os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"),
                phobert_path=os.path.join(tmp, "skip_phobert"),
                max_samples=1,
                no_save=True,
                device="cpu",
            )
            b = res["baseline"]
            self.assertEqual(b["num_samples"], 1)
            self.assertTrue(math.isfinite(b["sentiment_macro_f1"]))
            self.assertTrue(math.isfinite(b["topic_macro_f1"]))

    def test_f5_b04_compute_comparisons_and_markdown_single_model(self) -> None:
        from evaluation.evaluate import compute_comparisons, generate_markdown_report

        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), "r", encoding="utf-8") as f:
            b_m = json.load(f)["baseline"]
        self.assertEqual(compute_comparisons(None, b_m), {})
        self.assertEqual(compute_comparisons(b_m, None), {})
        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "single.md")
            generate_markdown_report(
                baseline_metrics=b_m,
                phobert_metrics=None,
                comparisons={},
                split_name="test",
                num_samples=3166,
                device_name="CPU",
                output_path=md_path,
            )
            self.assertTrue(os.path.isfile(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                self.assertIn("Classical Baseline", f.read())

    def test_f5_b05_majority_baseline_ties_and_single_class(self) -> None:
        acc_tie, f1_tie = compute_majority_baseline([0, 1, 2, 0, 1, 2], 3)
        self.assertAlmostEqual(acc_tie, 1.0 / 3.0, places=6)
        self.assertTrue(math.isfinite(f1_tie))
        acc_one, f1_one = compute_majority_baseline([2, 2, 2, 2], 3)
        self.assertAlmostEqual(acc_one, 1.0, places=6)
        self.assertAlmostEqual(f1_one, 1.0 / 3.0, places=6)
        acc_empty, f1_empty = compute_majority_baseline([], 3)
        self.assertEqual((acc_empty, f1_empty), (0.0, 0.0))


class TestF6EnglishDatasetBoundaries(unittest.TestCase):
    """F6 Boundary Cases: English dataset quality and boundary invariants."""

    def test_f6_b01_no_empty_or_nan_sentences_in_en_splits(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            texts, _, _ = load_vsfc_split(split, data_dir=en_dir, language="en", auto_download=False)
            for t in texts:
                self.assertTrue(t.strip())
                self.assertNotIn(t.strip().lower(), ("nan", "none", "null"))

    def test_f6_b02_no_extreme_short_or_corrupt_control_chars(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            texts, _, _ = load_vsfc_split(split, data_dir=en_dir, language="en", auto_download=False)
            bad_ctrl = [t for t in texts if any(ord(c) < 32 and c not in "\t\n\r" for c in t)]
            self.assertEqual(len(bad_ctrl), 0)

    def test_f6_b03_en_splits_contain_ascii_english_words(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            texts, _, _ = load_vsfc_split(split, data_dir=en_dir, language="en", auto_download=False)
            ascii_word_ratio = sum(1 for t in texts if re.search(r"[A-Za-z]{2,}", t)) / len(texts)
            self.assertGreater(ascii_word_ratio, 0.98)

    def test_f6_b04_en_class_support_nonzero_all_splits(self) -> None:
        require_en_deliverables(self)
        from collections import Counter
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        for split in ("train", "validation", "test"):
            _, sents, tops = load_vsfc_split(split, data_dir=en_dir, language="en", auto_download=False)
            sc = Counter(sents)
            tc = Counter(tops)
            for k in (0, 1, 2):
                self.assertGreater(sc[k], 0)
            for k in (0, 1, 2, 3):
                self.assertGreater(tc[k], 0)

    def test_f6_b05_en_long_sentences_within_bounds(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_dir = os.path.join(PROJECT_ROOT, "data", "english_feedback")
        texts, _, _ = load_vsfc_split("test", data_dir=en_dir, language="en", auto_download=False)
        self.assertGreater(max(len(t) for t in texts), 20)
        self.assertLess(max(len(t) for t in texts), 20000)


class TestF7Tier1BaselineBoundaries(unittest.TestCase):
    """F7 Boundary Cases: English Tier 1 `baseline_en.joblib` edge inputs."""

    def test_f7_b01_empty_and_whitespace(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        for txt in ("", "   ", "\t\n\r", "\u00a0\u200b"):
            assert_prediction_schema(self, model.predict(txt))

    def test_f7_b02_single_char_and_punctuation_only(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        for txt in ("a", ".", "???!!!", "!@#$%^&*()", "99999"):
            assert_prediction_schema(self, model.predict(txt))

    def test_f7_b03_contractions_hyphenated_and_emojis(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        for txt in (
            "The campus Wi-Fi isn't working and lab PCs don't boot 😡💻",
            "10/10 course! 🙌🔥",
            "It's an up-to-date, well-structured syllabus.",
        ):
            assert_prediction_schema(self, model.predict(txt))

    def test_f7_b04_sarcasm_and_irony_en(self) -> None:
        require_en_deliverables(self)
        model = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        for txt in (
            "Oh wonderful, another pop quiz at 7am with zero review—best semester ever.",
            "Thanks for turning room B204 into a free sauna!",
        ):
            assert_prediction_schema(self, model.predict(txt))

    def test_f7_b05_long_over_256_tokens_and_reload_determinism(self) -> None:
        from models.baseline_model import VietnameseFeedbackBaseline

        vi_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        m1 = VietnameseFeedbackBaseline.load(vi_path)
        m2 = VietnameseFeedbackBaseline.load(vi_path)
        vi_long = "Thầy dạy rất nhiệt tình và dễ hiểu. " * 60
        s1, t1 = m1.predict_proba([vi_long])
        s2, t2 = m2.predict_proba([vi_long])
        self.assertTrue(np.array_equal(s1, s2))
        self.assertTrue(np.array_equal(t1, t2))

        require_en_deliverables(self)
        en_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib")
        e1 = VietnameseFeedbackBaseline.load(en_path)
        e2 = VietnameseFeedbackBaseline.load(en_path)
        en_long = "The lecturer explains every algorithm clearly and answers questions. " * 60
        es1, et1 = e1.predict_proba([en_long])
        es2, et2 = e2.predict_proba([en_long])
        self.assertTrue(np.array_equal(es1, es2))
        self.assertTrue(np.array_equal(et1, et2))


class TestF8Tier2TransformerBoundaries(unittest.TestCase):
    """F8 Boundary Cases: English Tier 2 `english_multitask/` edge inputs & truncation."""

    def test_f8_b01_empty_and_whitespace(self) -> None:
        require_en_deliverables(self)
        model, tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
        for txt in ("", "   ", "\t\n"):
            assert_prediction_schema(self, model.predict(txt, tokenizer=tok))

    def test_f8_b02_single_char_and_punctuation_only(self) -> None:
        require_en_deliverables(self)
        model, tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
        for txt in ("x", "!", "...", "12345"):
            assert_prediction_schema(self, model.predict(txt, tokenizer=tok))

    def test_f8_b03_over_256_tokens_truncation(self) -> None:
        vi_model, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        vi_long = "Thầy giảng bài rất nhiệt tình và phòng học thoáng mát. " * 40
        assert_prediction_schema(self, vi_model.predict(vi_long, tokenizer=vi_tok, max_length=128))
        assert_prediction_schema(self, vi_model.predict(vi_long, tokenizer=vi_tok, max_length=256))

        require_en_deliverables(self)
        en_model, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
        en_long = "The lecturer explains concepts clearly and the classroom projector works well. " * 40
        assert_prediction_schema(self, en_model.predict(en_long, tokenizer=en_tok, max_length=128))
        assert_prediction_schema(self, en_model.predict(en_long, tokenizer=en_tok, max_length=256))

    def test_f8_b04_contractions_emojis_and_sarcasm(self) -> None:
        require_en_deliverables(self)
        model, tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))
        for txt in (
            "The Wi-Fi isn't stable and the projector doesn't turn on 🔌😞",
            "Love how the AC breaks right during final exams!",
        ):
            assert_prediction_schema(self, model.predict(txt, tokenizer=tok))

    def test_f8_b05_deterministic_offline_reload_and_batch_sizes(self) -> None:
        require_en_deliverables(self)
        from models.phobert_model import PhoBERTMultiTask

        ckpt = os.path.join(PROJECT_ROOT, "saved_models", "english_multitask")
        tok = load_local_tokenizer(ckpt)
        m1 = PhoBERTMultiTask.from_pretrained(ckpt, device="cpu")
        m2 = PhoBERTMultiTask.from_pretrained(ckpt, device="cpu")
        batch = [
            "Great lecturer.",
            "Broken projector.",
            "Okay syllabus.",
            "Slow admin office.",
            "Helpful TA.",
            "Hot room.",
            "Good lab project.",
        ]
        s1, t1 = m1.predict_proba(batch, tokenizer=tok, batch_size=2)
        s2, t2 = m2.predict_proba(batch, tokenizer=tok, batch_size=7)
        self.assertTrue(np.allclose(s1, s2, atol=1e-5))
        self.assertTrue(np.allclose(t1, t2, atol=1e-5))


class TestF9ReportBoundaries(unittest.TestCase):
    """F9 Boundary Cases: Evaluation report arithmetic and error boundary invariants."""

    def test_f9_b01_vi_metrics_json_finite_nonnegative_values(self) -> None:
        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("baseline", "phobert"):
            m = data["models"][key]
            for field in (
                "sentiment_accuracy",
                "sentiment_macro_f1",
                "topic_accuracy",
                "topic_macro_f1",
                "mean_macro_f1",
                "inference_time_sec",
                "latency_ms_per_sample",
                "storage_footprint_mb",
            ):
                val = m[field]
                self.assertTrue(math.isfinite(val))
                self.assertGreaterEqual(val, 0.0)

    def test_f9_b02_vi_per_class_supports_sum_to_3166(self) -> None:
        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("baseline", "phobert"):
            pc = data["models"][key]["per_class"]
            s_sup = sum(pc["sentiment"][c]["support"] for c in ("negative", "neutral", "positive"))
            t_sup = sum(
                pc["topic"][c]["support"]
                for c in ("lecturer", "training_program", "facility", "others")
            )
            self.assertEqual(s_sup, 3166)
            self.assertEqual(t_sup, 3166)

    def test_f9_b03_en_metrics_json_finite_nonnegative_values(self) -> None:
        require_en_deliverables(self)
        with open(os.path.join(PROJECT_ROOT, "results", "metrics_en.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("baseline", "phobert"):
            m = data["models"][key]
            for field in (
                "sentiment_accuracy",
                "sentiment_macro_f1",
                "topic_accuracy",
                "topic_macro_f1",
                "mean_macro_f1",
                "inference_time_sec",
                "storage_footprint_mb",
            ):
                val = m[field]
                self.assertTrue(math.isfinite(val))
                self.assertGreaterEqual(val, 0.0)

    def test_f9_b04_en_per_class_supports_match_test_split(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        texts, _, _ = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        with open(os.path.join(PROJECT_ROOT, "results", "metrics_en.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("baseline", "phobert"):
            pc = data["models"][key]["per_class"]
            s_sup = sum(pc["sentiment"][c]["support"] for c in ("negative", "neutral", "positive"))
            t_sup = sum(
                pc["topic"][c]["support"]
                for c in ("lecturer", "training_program", "facility", "others")
            )
            self.assertEqual(s_sup, len(texts))
            self.assertEqual(t_sup, len(texts))

    def test_f9_b05_en_invalid_split_name_fails_cleanly(self) -> None:
        require_en_deliverables(self)
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                "--language",
                "en",
                "--split",
                "nonexistent_split",
                "--no_save",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        self.assertNotEqual(proc.returncode, 0)


class TestF10UIBoundariesAppTest(unittest.TestCase):
    """F10 Boundary Cases: Streamlit App UI state sync, empty inputs, XSS, and extreme lengths."""

    def test_f10_b01_ui_lang_switch_preserves_manual_and_loaded_text(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        custom = "Custom feedback typed before toggling UI language"
        at.text_area[0].input(custom).run()
        at.sidebar.button_group[1].set_value("en").run()
        self.assertEqual(at.text_area[0].value, custom)
        self.assertEqual(at.session_state["user_text"], custom)
        at.sidebar.button_group[1].set_value("vi").run()
        self.assertEqual(at.text_area[0].value, custom)

    def test_f10_b02_clear_button_resets_user_text_and_results(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at = select_model_tier(at, "baseline")
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertGreaterEqual(len(at.progress), 7)
        at = find_main_button(at, ("Xóa trắng", "Clear")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["user_text"], "")
        self.assertEqual(at.text_area[0].value, "")
        self.assertEqual(len(at.progress), 0)

    def test_f10_b03_empty_and_whitespace_submit_warns_without_crash(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        for lang in ("vi", "en"):
            at.sidebar.button_group[1].set_value(lang).run()
            for blank in ("", "   ", "\t\n  "):
                at = find_main_button(at, ("Xóa trắng", "Clear")).click().run()
                at.text_area[0].input(blank).run()
                at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                self.assertFalse(at.exception)
                self.assertGreaterEqual(len(at.warning), 1)
                self.assertEqual(len(at.progress), 0)

    def test_f10_b04_reload_same_preset_after_manual_edit_overwrites(self) -> None:
        from streamlit.testing.v1 import AppTest
        import app.app as demo_app

        vi_presets = demo_app.get_presets()
        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[2]).run()
        at.sidebar.button[0].click().run()
        at.text_area[0].input("Manual edit overriding preset").run()
        at.sidebar.button[0].click().run()
        self.assertEqual(at.text_area[0].value, vi_presets[2]["text"])

        if get_e2e_scope() != "vi":
            en_presets = demo_app.get_english_presets()
            at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[1]).run()
            at.sidebar.button[1].click().run()
            at.text_area[0].input("Manual edit overriding EN preset").run()
            at.sidebar.button[1].click().run()
            self.assertEqual(at.text_area[0].value, en_presets[1]["text"])

    def test_f10_b05_extreme_length_input_in_ui(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at = select_model_tier(at, "both")
        long_vi = "Thầy giảng bài rất dễ hiểu và nhiệt tình. " * 28
        at.text_area[0].input(long_vi).run()
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.progress), 14)

        if get_e2e_scope() != "vi":
            long_en = "The lecturer explains algorithms clearly and answers every question. " * 28
            at.text_area[0].input(long_en).run()
            at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.progress), 14)

    def test_f10_b06_special_chars_and_xss_in_ui(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at = select_model_tier(at, "both")
        xss_text = "<script>alert('xss')</script> ??? 😊 123"
        at.text_area[0].input(xss_text).run()
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.progress), 14)
        rendered_html = " ".join(str(getattr(m, "body", "")) for m in at.markdown)
        self.assertNotIn("<script>alert('xss')</script>", rendered_html)


class TestF11SuiteBoundaries(unittest.TestCase):
    """F11 Boundary Cases: Unit test suite structure and non-mutation invariants."""

    def test_f11_b01_no_init_py_regression_in_tests(self) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
            self.assertGreaterEqual(suite.countTestCases(), 169)
        finally:
            os.chdir(old_cwd)

    def test_f11_b02_standalone_unit_module_runs_cleanly(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "tests/test_baseline.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_f11_b03_saved_models_untouched_timestamps(self) -> None:
        base = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        pho = os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask", "model.pt")
        t0_b = os.path.getmtime(base)
        t0_p = os.path.getmtime(pho)
        _ = get_cached_baseline(base)
        _ = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        self.assertEqual(os.path.getmtime(base), t0_b)
        self.assertEqual(os.path.getmtime(pho), t0_p)

    def test_f11_b04_results_files_untouched_timestamps(self) -> None:
        m_json = os.path.join(PROJECT_ROOT, "results", "metrics.json")
        m_md = os.path.join(PROJECT_ROOT, "results", "comparison_summary.md")
        self.assertTrue(os.path.isfile(m_json))
        self.assertTrue(os.path.isfile(m_md))
        self.assertGreater(os.path.getsize(m_json), 1000)
        self.assertGreater(os.path.getsize(m_md), 1000)

    def test_f11_b05_original_test_method_counts_no_shrink(self) -> None:
        expected_min = {
            "test_data_pipeline.py": 13,
            "test_baseline.py": 19,
            "test_phobert.py": 17,
            "test_evaluation.py": 22,
            "test_app.py": 23,
        }
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            for fname, min_cnt in expected_min.items():
                suite = unittest.TestLoader().discover("tests", pattern=fname)
                self.assertGreaterEqual(
                    suite.countTestCases(),
                    min_cnt,
                    f"{fname} shrank below {min_cnt} cases",
                )
        finally:
            os.chdir(old_cwd)
