"""Tier 4 E2E Tests: Real-World Application Scenarios (6 comprehensive end-to-end workloads)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

from e2e_tests.e2e_helpers import (
    APPTEST_TIMEOUT_TIER4,
    APP_PATH,
    PROJECT_ROOT,
    assert_prediction_schema,
    compute_majority_baseline,
    create_tiny_local_hf_checkpoint,
    find_main_button,
    get_cached_baseline,
    get_cached_phobert,
    get_e2e_scope,
    require_en_deliverables,
    resolve_detect_language,
    run_unit_test_suite_cached,
    select_model_tier,
    write_synthetic_vsfc_dir,
)


class TestTier4RealWorldWorkloads(unittest.TestCase):
    """End-to-end realistic academic showcase and engineering pipeline scenarios."""

    def test_t4_s1_bilingual_evaluation_audit(self) -> None:
        """Scenario 1: Bilingual Evaluation Audit (VI no-regression + EN > majority baseline)."""
        vi_json_path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
        vi_md_path = os.path.join(PROJECT_ROOT, "results", "comparison_summary.md")
        with open(vi_json_path, "r", encoding="utf-8") as f:
            vi_m = json.load(f)
        with open(vi_md_path, "r", encoding="utf-8") as f:
            vi_md = f.read()

        self.assertEqual(vi_m["num_samples"], 3166)
        b_vi = vi_m["models"]["baseline"]
        p_vi = vi_m["models"]["phobert"]
        tol = 1e-4
        self.assertGreaterEqual(b_vi["sentiment_accuracy"], 0.86734049 - tol)
        self.assertGreaterEqual(b_vi["sentiment_macro_f1"], 0.71897139 - tol)
        self.assertGreaterEqual(b_vi["topic_accuracy"], 0.80985471 - tol)
        self.assertGreaterEqual(b_vi["topic_macro_f1"], 0.72716562 - tol)
        self.assertGreaterEqual(p_vi["sentiment_accuracy"], 0.93493367 - tol)
        self.assertGreaterEqual(p_vi["sentiment_macro_f1"], 0.82888945 - tol)
        self.assertGreaterEqual(p_vi["topic_accuracy"], 0.88408086 - tol)
        self.assertGreaterEqual(p_vi["topic_macro_f1"], 0.80103109 - tol)
        self.assertIn(f"{p_vi['sentiment_macro_f1'] * 100:.2f}%", vi_md)
        self.assertIn(f"{p_vi['topic_macro_f1'] * 100:.2f}%", vi_md)

        require_en_deliverables(self)
        from data.dataset import load_vsfc_split

        en_json_path = os.path.join(PROJECT_ROOT, "results", "metrics_en.json")
        en_md_path = os.path.join(PROJECT_ROOT, "results", "comparison_summary_en.md")
        self.assertTrue(os.path.isfile(en_json_path))
        self.assertTrue(os.path.isfile(en_md_path))
        with open(en_json_path, "r", encoding="utf-8") as f:
            en_m = json.load(f)
        with open(en_md_path, "r", encoding="utf-8") as f:
            en_md = f.read()

        _, en_sents, en_tops = load_vsfc_split(
            "test",
            data_dir=os.path.join(PROJECT_ROOT, "data", "english_feedback"),
            language="en",
            auto_download=False,
        )
        self.assertEqual(en_m["num_samples"], len(en_sents))
        _, maj_s_f1 = compute_majority_baseline(en_sents, 3)
        _, maj_t_f1 = compute_majority_baseline(en_tops, 4)

        for tier_key in ("baseline", "phobert"):
            m = en_m["models"][tier_key]
            self.assertGreater(m["sentiment_macro_f1"], maj_s_f1 + 0.05)
            self.assertGreater(m["topic_macro_f1"], maj_t_f1 + 0.05)
            self.assertIn(f"{m['sentiment_macro_f1'] * 100:.2f}%", en_md)
            self.assertIn(f"{m['topic_macro_f1'] * 100:.2f}%", en_md)

    def test_t4_s2_full_streamlit_academic_showcase_journey(self) -> None:
        """Scenario 2: Full Streamlit Academic Showcase Journey (VI + UI toggles + EN + recovery)."""
        from streamlit.testing.v1 import AppTest
        import app.app as demo_app

        vi_presets = demo_app.get_presets()
        at = AppTest.from_file(APP_PATH, default_timeout=APPTEST_TIMEOUT_TIER4).run()
        self.assertFalse(at.exception)

        # Browse VI library: positive lecturer and sarcasm/facility
        at = select_model_tier(at, "both")
        for idx in (0, 9):
            at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[idx]).run()
            at.sidebar.button[0].click().run()
            self.assertEqual(at.text_area[0].value, vi_presets[idx]["text"])
            at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.progress), 14)

        # Toggle UI language to 'en' and theme to 'dark', preserving loaded text
        at.sidebar.button_group[1].set_value("en").run()
        at.sidebar.button_group[0].set_value("dark").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_area[0].value, vi_presets[9]["text"])

        if get_e2e_scope() != "vi":
            en_presets = demo_app.get_english_presets()
            self.assertGreaterEqual(len(at.sidebar.selectbox), 2)
            for idx in (1, 6):
                idx_clamped = min(idx, len(en_presets) - 1)
                at.sidebar.selectbox[1].set_value(
                    at.sidebar.selectbox[1].options[idx_clamped]
                ).run()
                at.sidebar.button[1].click().run()
                self.assertEqual(at.text_area[0].value, en_presets[idx_clamped]["text"])
                at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
                self.assertFalse(at.exception)
                self.assertEqual(len(at.progress), 14)

        # Clear -> empty submit warning -> clean recovery
        at = find_main_button(at, ("Xóa trắng", "Clear")).click().run()
        self.assertEqual(at.text_area[0].value, "")
        self.assertEqual(len(at.progress), 0)
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertGreaterEqual(len(at.warning), 1)
        self.assertEqual(len(at.progress), 0)

        at.text_area[0].input("Thầy dạy rất nhiệt tình và dễ hiểu.").run()
        at = find_main_button(at, ("Phân tích", "Analyze")).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.progress), 14)

    def test_t4_s3_bilingual_sentiment_topic_matrix(self) -> None:
        """Scenario 3: 24-input Matrix (3 sentiments x 4 topics x VI & EN) across both tiers."""
        vi_matrix = [
            # lecturer (neg, neu, pos)
            ("Giảng viên giảng bài quá nhanh và khó hiểu.", "negative", "lecturer"),
            ("Thầy dạy đúng theo slide và đúng giờ quy định.", "neutral", "lecturer"),
            ("Cô giảng bài rất nhiệt tình, tận tâm và dễ hiểu.", "positive", "lecturer"),
            # training_program
            ("Chương trình đào tạo nặng lý thuyết, lịch thi quá dày đặc.", "negative", "training_program"),
            ("Nội dung môn học ở mức độ vừa phải, bình thường.", "neutral", "training_program"),
            ("Môn học có tính thực tiễn cao, đồ án rất hữu ích.", "positive", "training_program"),
            # facility
            ("Phòng máy tính hư hỏng nhiều, máy lạnh bị rò rỉ nước.", "negative", "facility"),
            ("Cơ sở vật chất phòng học tạm ổn.", "neutral", "facility"),
            ("Phòng học thoáng mát, máy chiếu và wifi hoạt động rất tốt.", "positive", "facility"),
            # others
            ("Học phí tăng cao nhưng thủ tục hành chính xử lý rất chậm.", "negative", "others"),
            ("Thông tin đăng ký học phần được thông báo bình thường.", "neutral", "others"),
            ("Phòng công tác sinh viên hỗ trợ thủ tục rất nhanh chóng.", "positive", "others"),
        ]
        vi_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib"))
        vi_tr, vi_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask"))

        vi_texts = [x[0] for x in vi_matrix]
        vi_b_preds = vi_base.predict(vi_texts)
        vi_t_preds = vi_tr.predict(vi_texts, tokenizer=vi_tok)
        self.assertEqual(len(vi_b_preds), 12)
        self.assertEqual(len(vi_t_preds), 12)
        for p in vi_b_preds + vi_t_preds:
            assert_prediction_schema(self, p)
        self.assertGreaterEqual(len({p["topic"]["label"] for p in vi_t_preds}), 3)
        self.assertGreaterEqual(len({p["sentiment"]["label"] for p in vi_t_preds}), 2)

        require_en_deliverables(self)
        detect_language = resolve_detect_language()
        en_matrix = [
            ("The instructor speaks too fast and gives no practical examples.", "negative", "lecturer"),
            ("The lecturer follows the syllabus slides on schedule.", "neutral", "lecturer"),
            ("The professor explains every concept clearly and patiently.", "positive", "lecturer"),
            ("The exam schedule is way too packed and stressful.", "negative", "training_program"),
            ("The curriculum difficulty is moderate overall.", "neutral", "training_program"),
            ("The course projects are very practical for industry jobs.", "positive", "training_program"),
            ("The classroom air conditioner is broken and Wi-Fi drops constantly.", "negative", "facility"),
            ("The lab equipment condition is average.", "neutral", "facility"),
            ("The lecture hall is clean, cool, and the projector works great.", "positive", "facility"),
            ("Tuition fees keep rising while admin paperwork is very slow.", "negative", "others"),
            ("Registration announcements were posted as usual.", "neutral", "others"),
            ("Student affairs staff handled my documents quickly and politely.", "positive", "others"),
        ]
        en_base = get_cached_baseline(os.path.join(PROJECT_ROOT, "saved_models", "baseline_en.joblib"))
        en_tr, en_tok = get_cached_phobert(os.path.join(PROJECT_ROOT, "saved_models", "english_multitask"))

        for text, _, _ in vi_matrix:
            self.assertEqual(detect_language(text), "vi")
        for text, _, _ in en_matrix:
            self.assertEqual(detect_language(text), "en")

        en_texts = [x[0] for x in en_matrix]
        en_b_preds = en_base.predict(en_texts)
        en_t_preds = en_tr.predict(en_texts, tokenizer=en_tok)
        self.assertEqual(len(en_b_preds), 12)
        self.assertEqual(len(en_t_preds), 12)
        for p in en_b_preds + en_t_preds:
            assert_prediction_schema(self, p)
        self.assertGreaterEqual(len({p["topic"]["label"] for p in en_t_preds}), 3)
        self.assertGreaterEqual(len({p["sentiment"]["label"] for p in en_t_preds}), 2)

    def test_t4_s4_offline_subprocess_reproducible_reload(self) -> None:
        """Scenario 4: Offline Reproducible Subprocess Checkpoint Reload (HF_HUB_OFFLINE=1)."""
        scope = get_e2e_scope()
        script = f"""
import json, os, sys
sys.path.insert(0, {PROJECT_ROOT!r})
from transformers import AutoTokenizer
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask

vi_b1 = VietnameseFeedbackBaseline.load(os.path.join({PROJECT_ROOT!r}, "saved_models", "baseline.joblib"))
vi_b2 = VietnameseFeedbackBaseline.load(os.path.join({PROJECT_ROOT!r}, "saved_models", "baseline.joblib"))
p1 = vi_b1.predict("Thầy dạy rất nhiệt tình")
p2 = vi_b2.predict("Thầy dạy rất nhiệt tình")
assert p1 == p2, (p1, p2)

vi_ckpt = os.path.join({PROJECT_ROOT!r}, "saved_models", "phobert_multitask")
vi_tok = AutoTokenizer.from_pretrained(vi_ckpt, use_fast=False, local_files_only=True)
vi_t1 = PhoBERTMultiTask.from_pretrained(vi_ckpt, device="cpu")
vi_t2 = PhoBERTMultiTask.from_pretrained(vi_ckpt, device="cpu")
q1 = vi_t1.predict("Thầy dạy rất nhiệt tình", tokenizer=vi_tok)
q2 = vi_t2.predict("Thầy dạy rất nhiệt tình", tokenizer=vi_tok)
assert q1["sentiment"]["label_id"] == q2["sentiment"]["label_id"]
assert q1["topic"]["label_id"] == q2["topic"]["label_id"]

if {scope!r} != "vi":
    q3 = vi_t1.predict("Thầy dạy rất nhiệt tình", tokenizer=None)
    assert q3["sentiment"]["label_id"] == q1["sentiment"]["label_id"]
    en_b = VietnameseFeedbackBaseline.load(os.path.join({PROJECT_ROOT!r}, "saved_models", "baseline_en.joblib"))
    en_p = en_b.predict("The professor explains clearly.")
    en_ckpt = os.path.join({PROJECT_ROOT!r}, "saved_models", "english_multitask")
    en_t = PhoBERTMultiTask.from_pretrained(en_ckpt, device="cpu")
    en_q1 = en_t.predict("The professor explains clearly.", tokenizer=None)
    en_q2 = en_t.predict("The professor explains clearly.", tokenizer=None)
    assert en_q1["sentiment"]["label_id"] == en_q2["sentiment"]["label_id"]
    assert en_p["sentiment"]["label"] in ("negative", "neutral", "positive")
print("OFFLINE_RELOAD_OK")
"""
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OFFLINE_RELOAD_OK", proc.stdout)

    def test_t4_s5_bilingual_cli_training_to_evaluation_smoke(self) -> None:
        """Scenario 5: Bilingual CLI & Training-to-Evaluation Smoke Pipeline in temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            vi_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "vi_data"), lang="vi")
            vi_base = os.path.join(tmp, "vi_base.joblib")
            vi_res = os.path.join(tmp, "vi_res")

            p1 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_baseline.py"),
                    "--data_dir",
                    vi_dir,
                    "--save_path",
                    vi_base,
                    "--max_features",
                    "300",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(p1.returncode, 0, p1.stderr)

            p2 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--data_dir",
                    vi_dir,
                    "--baseline_path",
                    vi_base,
                    "--phobert_path",
                    os.path.join(tmp, "skip"),
                    "--output_dir",
                    vi_res,
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
            self.assertEqual(p2.returncode, 0, p2.stderr)
            self.assertTrue(os.path.isfile(os.path.join(vi_res, "metrics.json")))

            require_en_deliverables(self)
            en_dir = write_synthetic_vsfc_dir(os.path.join(tmp, "en_data"), lang="en")
            tiny_hf = create_tiny_local_hf_checkpoint(os.path.join(tmp, "tiny_hf"))
            en_base = os.path.join(tmp, "en_base.joblib")
            en_tr = os.path.join(tmp, "en_tr")
            en_res = os.path.join(tmp, "en_res")

            p3 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_baseline.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    en_dir,
                    "--save_path",
                    en_base,
                    "--max_features",
                    "300",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(p3.returncode, 0, p3.stderr)

            p4 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "training", "train_phobert.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    en_dir,
                    "--model_name",
                    tiny_hf,
                    "--save_dir",
                    en_tr,
                    "--epochs",
                    "1",
                    "--batch_size",
                    "8",
                    "--max_length",
                    "32",
                    "--quick_run",
                    "--no_amp",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(p4.returncode, 0, p4.stderr)

            p5 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py"),
                    "--language",
                    "en",
                    "--data_dir",
                    en_dir,
                    "--baseline_path",
                    en_base,
                    "--phobert_path",
                    en_tr,
                    "--output_dir",
                    en_res,
                    "--max_length",
                    "32",
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
            self.assertEqual(p5.returncode, 0, p5.stderr)
            self.assertTrue(os.path.isfile(os.path.join(en_res, "metrics_en.json")))
            self.assertTrue(os.path.isfile(os.path.join(en_res, "comparison_summary_en.md")))

    def test_t4_s6_code_review_and_regression_shield_audit(self) -> None:
        """Scenario 6: Code Review & Regression Shield Audit (`CODE_REVIEW.md` + `tests/`)."""
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            discovered = unittest.TestLoader().discover("tests", pattern="test_*.py")
            self.assertGreaterEqual(discovered.countTestCases(), 169)
        finally:
            os.chdir(old_cwd)

        require_en_deliverables(self)
        review_path = os.path.join(PROJECT_ROOT, "CODE_REVIEW.md")
        self.assertTrue(os.path.isfile(review_path))
        with open(review_path, "r", encoding="utf-8") as f:
            review = f.read()
        for d in ("models/", "data/", "training/", "evaluation/", "app/", "tests/"):
            self.assertIn(d, review.lower())
        cited = set(
            re.findall(
                r"\b(?:models|data|training|evaluation|app|tests)/[A-Za-z0-9_./-]+\.py\b",
                review,
            )
        )
        self.assertGreaterEqual(len(cited), 6)
        for rel in cited:
            self.assertTrue(os.path.isfile(os.path.join(PROJECT_ROOT, rel)), f"Missing {rel}")

        res = run_unit_test_suite_cached()
        self.assertEqual(
            res["returncode"],
            0,
            f"tests/ regression shield failed (rc={res['returncode']}):\n{res['stderr'][-2000:]}",
        )
