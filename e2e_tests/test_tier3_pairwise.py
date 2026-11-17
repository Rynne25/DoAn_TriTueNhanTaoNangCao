"""Tier 3 E2E Tests: Cross-Feature Pairwise Combinations (>= 15 test methods)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter

from e2e_tests.e2e_helpers import (
    APPTEST_TIMEOUT,
    APP_PATH,
    PROJECT_ROOT,
    assert_prediction_schema,
    compute_majority_baseline,
    extract_majority_baseline_metrics,
    find_main_button,
    get_cached_baseline,
    get_cached_phobert,
    get_e2e_scope,
    has_vi_diacritics,
    require_en_deliverables,
    resolve_detect_language,
    select_model_tier,
    write_synthetic_vsfc_dir,
)


class TestTier3PairwiseCombinations(unittest.TestCase):
    """Cross-feature pairwise combinatorial interactions across F1-F11."""

    def test_pair_f10_vi_preset_x_all_tiers(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        for idx in (0, 1, 9):
            at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[idx]).run()
            at.sidebar.button[0].click().run()
            for tier, bars in (("baseline", 7), ("transformer", 7), ("both", 14)):
                with self.subTest(vi_preset_idx=idx, tier=tier):
                    at = select_model_tier(at, tier)
                    at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(len(at.progress), bars)

    def test_pair_f10_en_preset_x_all_tiers(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        for idx in (0, 1, 2):
            at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[idx]).run()
            at.sidebar.button[1].click().run()
            for tier, bars in (("baseline", 7), ("transformer", 7), ("both", 14)):
                with self.subTest(en_preset_idx=idx, tier=tier):
                    at = select_model_tier(at, tier)
                    at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(len(at.progress), bars)

    def test_pair_f10_ui_lang_x_sample_language(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        sample_langs = ("vi",) if get_e2e_scope() == "vi" else ("vi", "en")
        for ui_lang in ("vi", "en"):
            for sample_lang in sample_langs:
                with self.subTest(ui_lang=ui_lang, sample_lang=sample_lang):
                    at.sidebar.button_group[1].set_value(ui_lang).run()
                    if sample_lang == "vi":
                        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[0]).run()
                        at.sidebar.button[0].click().run()
                    else:
                        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[0]).run()
                        at.sidebar.button[1].click().run()
                    at = select_model_tier(at, "both")
                    at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(len(at.progress), 14)

    def test_pair_f10_consecutive_en_then_vi_same_session(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest
        import app.app as demo_app

        vi_presets = demo_app.get_presets()
        en_presets = demo_app.get_english_presets()
        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at = select_model_tier(at, "both")

        at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[0]).run()
        at.sidebar.button[1].click().run()
        self.assertEqual(at.text_area[0].value, en_presets[0]["text"])
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.progress), 14)

        at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[1]).run()
        at.sidebar.button[0].click().run()
        self.assertEqual(at.text_area[0].value, vi_presets[1]["text"])
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.progress), 14)

    def test_pair_f10_theme_x_ui_lang(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        for theme in ("light", "dark"):
            for ui_lang in ("vi", "en"):
                with self.subTest(theme=theme, ui_lang=ui_lang):
                    at.sidebar.button_group[0].set_value(theme).run()
                    at.sidebar.button_group[1].set_value(ui_lang).run()
                    at = select_model_tier(at, "both")
                    at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(len(at.progress), 14)

    def test_pair_f2_f6_vi_en_data_isolation(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        vi_texts, _, _ = load_vsfc_split(
            "validation",
            data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc"),
            language="vi",
            auto_download=False,
        )
        en_texts, _, _ = load_vsfc_split(
            "validation",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        self.assertGreater(sum(1 for t in vi_texts[:100] if "_" in t), 80)
        self.assertEqual(sum(1 for t in en_texts[:100] if has_vi_diacritics(t)), 0)

    def test_pair_f2_f4_en_data_to_train_baseline_cli(self) -> None:
        require_en_deliverables(self)
        from models.baseline_model import VietnameseFeedbackBaseline

        with tempfile.TemporaryDirectory() as tmp:
            en_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            out_path = os.path.join(tmp, "baseline_en_tmp.joblib")
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
                    "400",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            model = VietnameseFeedbackBaseline.load(out_path)
            self.assertIs(getattr(model, "default_word_segment", None), False)
            assert_prediction_schema(self, model.predict("The lecturer is helpful."))

    def test_pair_f4_f5_temp_train_to_evaluate_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vi_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "vi_data"), lang="vi", n_test=12)
            ckpt = os.path.join(tmp, "tmp_base.joblib")
            out_dir = os.path.join(tmp, "eval_out")

            p_train = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_baseline.py"),
                    "--data_dir",
                    vi_dir,
                    "--save_path",
                    ckpt,
                    "--max_features",
                    "300",
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
            self.assertEqual(p_train.returncode, 0, p_train.stderr)

            p_eval = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--data_dir",
                    vi_dir,
                    "--split",
                    "test",
                    "--baseline_path",
                    ckpt,
                    "--phobert_path",
                    os.path.join(tmp, "missing_phobert"),
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
            self.assertEqual(p_eval.returncode, 0, p_eval.stderr)
            with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["num_samples"], 12)
            self.assertIn("baseline", data["models"])

    def test_pair_f6_f9_en_test_data_vs_metrics_en_json(self) -> None:
        require_en_deliverables(self)
        from data.dataset import SENTIMENT_MAP, TOPIC_MAP, load_vsfc_split

        texts, sents, tops = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        with open(os.path.join(PROJECT_ROOT, "results", "metrics_en.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = data["metadata"]["dataset"]
        self.assertEqual(ds["sample_count"], len(texts))
        exp_s_dist = {SENTIMENT_MAP[k]: v for k, v in Counter(sents).items()}
        exp_t_dist = {TOPIC_MAP[k]: v for k, v in Counter(tops).items()}
        self.assertEqual(ds["sentiment_distribution"], exp_s_dist)
        self.assertEqual(ds["topic_distribution"], exp_t_dist)

        maj = extract_majority_baseline_metrics(data)
        self.assertIsNotNone(maj)
        maj_s_acc, maj_s_f1 = compute_majority_baseline(sents, 3)
        maj_t_acc, maj_t_f1 = compute_majority_baseline(tops, 4)
        self.assertAlmostEqual(maj["sentiment_accuracy"], maj_s_acc, places=4)
        self.assertAlmostEqual(maj["sentiment_macro_f1"], maj_s_f1, places=4)
        self.assertAlmostEqual(maj["topic_accuracy"], maj_t_acc, places=4)
        self.assertAlmostEqual(maj["topic_macro_f1"], maj_t_f1, places=4)

    def test_pair_f5_f9_evaluate_en_non_interference_with_metrics_json(self) -> None:
        require_en_deliverables(self)
        from training.train_baseline import train_baseline

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            ckpt = os.path.join(tmp, "b_en.joblib")
            train_baseline(
                data_dir=data_dir,
                save_path=ckpt,
                max_features=300,
                eval_val=False,
                eval_test=False,
                language="en",
            )
            out_dir = os.path.join(tmp, "res")
            os.makedirs(out_dir, exist_ok=True)
            sentinel_path = os.path.join(out_dir, "metrics.json")
            sentinel_payload = '{"sentinel": true}'
            with open(sentinel_path, "w", encoding="utf-8") as f:
                f.write(sentinel_payload)

            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    data_dir,
                    "--baseline_path",
                    ckpt,
                    "--phobert_path",
                    os.path.join(tmp, "none"),
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
            with open(sentinel_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), sentinel_payload)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "metrics_en.json")))

    def test_pair_f3_f7_language_routed_tier1(self) -> None:
        require_en_deliverables(self)
        detect_language = resolve_detect_language()
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))

        inputs = [
            ("Thầy giảng bài rất nhiệt tình.", "vi", "positive", "lecturer"),
            ("The classroom air conditioner is broken.", "en", "negative", "facility"),
        ]
        for text, exp_lang, exp_s, exp_t in inputs:
            lang = detect_language(text)
            self.assertEqual(lang, exp_lang)
            pred = vi_base.predict(text) if lang == "vi" else en_base.predict(text)
            assert_prediction_schema(self, pred)
            self.assertEqual(pred["sentiment"]["label"], exp_s)
            self.assertEqual(pred["topic"]["label"], exp_t)

    def test_pair_f3_f8_language_routed_tier2(self) -> None:
        require_en_deliverables(self)
        detect_language = resolve_detect_language()
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))
        en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))

        inputs = [
            ("Thầy giảng bài rất nhiệt tình.", "vi", "positive", "lecturer"),
            ("The classroom air conditioner is broken.", "en", "negative", "facility"),
        ]
        for text, exp_lang, exp_s, exp_t in inputs:
            lang = detect_language(text)
            self.assertEqual(lang, exp_lang)
            pred = (
                vi_tr.predict(text, tokenizer=vi_tok)
                if lang == "vi"
                else en_tr.predict(text, tokenizer=en_tok)
            )
            assert_prediction_schema(self, pred)
            self.assertEqual(pred["sentiment"]["label"], exp_s)
            self.assertEqual(pred["topic"]["label"], exp_t)

    def test_pair_f7_f8_en_tier1_vs_tier2_parity_and_latency(self) -> None:
        require_en_deliverables(self)
        en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))

        texts = [
            "The professor explains concepts clearly and is always supportive.",
            "The classroom air conditioner is broken and Wi-Fi keeps dropping.",
        ] * 5
        t0 = time.perf_counter()
        p1 = en_base.predict(texts)
        d1 = time.perf_counter() - t0
        t1 = time.perf_counter()
        p2 = en_tr.predict(texts, tokenizer=en_tok)
        d2 = time.perf_counter() - t1

        self.assertEqual(len(p1), 10)
        self.assertEqual(len(p2), 10)
        self.assertEqual(p1[0]["sentiment"]["label"], p2[0]["sentiment"]["label"])
        self.assertEqual(p1[1]["topic"]["label"], p2[1]["topic"]["label"])
        self.assertLess(d1, d2)

    def test_pair_f6_f7_f8_en_test_slice_to_both_models(self) -> None:
        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        texts, _, _ = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        slice_32 = texts[:32]
        en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))

        b_preds = en_base.predict(slice_32)
        t_preds = en_tr.predict(slice_32, tokenizer=en_tok)
        self.assertEqual(len(b_preds), 32)
        self.assertEqual(len(t_preds), 32)
        for p in b_preds + t_preds:
            assert_prediction_schema(self, p)

    def test_pair_f3_f10_app_bilingual_routing(self) -> None:
        require_en_deliverables(self)
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT).run()
        at = select_model_tier(at, "both")
        for sample in (
            "Thầy giảng bài rất hay và tận tình với sinh viên.",
            "The professor explains every concept clearly and is very helpful.",
        ):
            at.text_area[0].input(sample).run()
            at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.progress), 14)

    def test_pair_f1_f11_code_review_vs_tests_accuracy(self) -> None:
        require_en_deliverables(self)
        with open(os.path.join(PROJECT_ROOT, "CODE_REVIEW.md"), "r", encoding="utf-8") as f:
            review = f.read()
        cited_tests = set(re.findall(r"\btests/[A-Za-z0-9_./-]+\.py\b", review))
        self.assertGreaterEqual(len(cited_tests), 1)
        for rel in cited_tests:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, rel)),
                f"CODE_REVIEW.md cites missing test file '{rel}'",
            )
