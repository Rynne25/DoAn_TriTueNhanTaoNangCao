"""Shared opaque-box helpers for the E2E test suite (`e2e_tests/`)."""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TOPIC_LABELS = ["lecturer", "training_program", "facility", "others"]
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

APPTEST_TIMEOUT = 90
APPTEST_TIMEOUT_TIER4 = 120


def _discover_project_root() -> str:
    """Discover repository root containing PROJECT.md / app / models / data."""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isfile(os.path.join(cur, "app", "app.py")) and os.path.isdir(
            os.path.join(cur, "models")
        ):
            return cur
        if os.path.isfile(os.path.join(cur, "PROJECT.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = _discover_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")

VI_DIACRITIC_CHARS = set(
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹ"
    "đ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    "ÈÉẸẺẼÊỀẾỆỂỄ"
    "ÌÍỊỈĨ"
    "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮ"
    "ỲÝỴỶỸ"
    "Đ"
)

EXPECTED_VI_PRESETS_SNAPSHOT: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "pos_lec",
        "Thầy dạy rất nhiệt tình, giảng giải cặn kẽ và luôn tạo không khí học tập thoải mái cho sinh viên.",
        "positive",
        "lecturer",
    ),
    (
        "neg_fac",
        "Phòng máy tính hư hỏng nhiều, máy lạnh bị rò rỉ nước và mạng wifi trường quá chập chờn.",
        "negative",
        "facility",
    ),
    (
        "neu_prog",
        "Chương trình đào tạo nhìn chung tạm ổn, nội dung các môn cơ sở ngành ở mức độ vừa phải.",
        "neutral",
        "training_program",
    ),
    (
        "neg_lec",
        "Giáo viên giảng bài khô khan, nói quá nhanh và không hướng dẫn sinh viên làm bài tập thực hành.",
        "negative",
        "lecturer",
    ),
    (
        "pos_fac",
        "Phòng học thoáng mát, máy lạnh chạy êm và trang thiết bị máy chiếu phục vụ thuyết trình rất tốt.",
        "positive",
        "facility",
    ),
    (
        "neg_prog",
        "Lịch học và lịch thi bố trí quá dày đặc, sinh viên không kịp tiếp thu kiến thức và làm đồ án.",
        "negative",
        "training_program",
    ),
    (
        "pos_prog",
        "Môn học có tính ứng dụng thực tế rất cao, hệ thống bài tập lớn giúp sinh viên nâng cao kỹ năng nghề nghiệp.",
        "positive",
        "training_program",
    ),
    (
        "neg_oth",
        "Học phí tăng hàng năm nhưng thủ tục hành chính tại phòng công tác sinh viên giải quyết rất chậm.",
        "negative",
        "others",
    ),
    (
        "neu_lec",
        "Thầy dạy theo đúng giáo trình trong slide, thời gian lên lớp đúng giờ quy định.",
        "neutral",
        "lecturer",
    ),
    (
        "sarcasm",
        "Cảm ơn nhà trường đã tạo cơ hội cho chúng em trải nghiệm phòng học xông hơi giữa trưa nắng.",
        "negative",
        "facility",
    ),
)

ORIGINAL_TWELVE_TEST_FILES: Tuple[str, ...] = (
    "test_adversarial_baseline.py",
    "test_adversarial_m5.py",
    "test_app.py",
    "test_baseline.py",
    "test_challenger_m4.py",
    "test_challenger_m5.py",
    "test_challenger_m5_it2_empirical.py",
    "test_data_pipeline.py",
    "test_empirical_challenger.py",
    "test_evaluation.py",
    "test_phobert.py",
    "test_phobert_stress.py",
)


def get_e2e_scope() -> str:
    """Return active E2E diagnostic scope: 'all' (default) or 'vi'."""
    scope = os.environ.get("E2E_SCOPE", "all").strip().lower()
    return "vi" if scope == "vi" else "all"


def require_en_deliverables(tc: Any) -> None:
    """
    Skip EN/M1-M3 dependent test ONLY when running under diagnostic `--scope vi`.
    In default `--scope all` mode this is a strict NO-OP so missing M1-M3
    deliverables fail loudly.
    """
    if get_e2e_scope() == "vi":
        tc.skipTest("Filtered by --scope vi")


def has_vi_diacritics(text: str) -> bool:
    """Return True if text contains Vietnamese diacritic characters."""
    return any(ch in VI_DIACRITIC_CHARS for ch in (text or ""))


def resolve_detect_language() -> Callable[[str], str]:
    """
    Probe candidate modules (`models.language`, `models.bilingual`,
    `models.baseline_model`, `models.phobert_model`) for `detect_language`.
    """
    candidates = (
        "models.language",
        "models.bilingual",
        "models.baseline_model",
        "models.phobert_model",
    )
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "detect_language", None)
        if callable(fn):
            return fn
    raise AssertionError(
        "M1 deliverable missing: `detect_language(text)` was not found in any of "
        f"{candidates}."
    )


def resolve_bilingual_router() -> Dict[str, Any]:
    """
    Probe `models/` for an exported bilingual router/predictor helper, and always
    provide a `detect_language` function from `resolve_detect_language()`.
    """
    detect_fn = resolve_detect_language()
    router_modules = (
        "models.bilingual",
        "models.language",
        "models.baseline_model",
        "models.phobert_model",
    )
    router_symbols = (
        "BilingualFeedbackPredictor",
        "BilingualPredictor",
        "BilingualClassifier",
        "BilingualRouter",
        "predict_bilingual",
        "bilingual_predict",
        "route_prediction",
        "predict_with_routing",
        "load_bilingual_models",
    )
    found: Dict[str, Any] = {"detect_language": detect_fn, "exports": {}}
    for mod_name in router_modules:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for sym in router_symbols:
            obj = getattr(mod, sym, None)
            if obj is not None:
                found["exports"][f"{mod_name}.{sym}"] = obj
    return found


def load_local_tokenizer(ckpt_dir: str) -> Any:
    """
    Load a Hugging Face tokenizer strictly from a local directory (`local_files_only=True`),
    supporting both fast tokenizers (RoBERTa/DistilRoBERTa) and slow tokenizers (PhoBERT).
    """
    from transformers import AutoTokenizer

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: '{ckpt_dir}'")
    try:
        return AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(
            ckpt_dir, use_fast=False, local_files_only=True
        )


def assert_prediction_schema(
    tc: Any, pred: Dict[str, Any], allow_language_key: bool = True
) -> None:
    """Assert standard single-item prediction dict schema, native types, and probability invariants."""
    tc.assertIsInstance(pred, dict, f"Expected dict prediction, got {type(pred)}")
    tc.assertIn("sentiment", pred)
    tc.assertIn("topic", pred)
    if "language" in pred:
        tc.assertTrue(
            allow_language_key,
            "Unexpected 'language' key when allow_language_key=False",
        )
        tc.assertIn(pred["language"], ("vi", "en"))

    for task, expected_map in (("sentiment", SENTIMENT_MAP), ("topic", TOPIC_MAP)):
        block = pred[task]
        tc.assertIsInstance(block, dict, f"'{task}' block must be dict")
        for req_key in ("label", "label_id", "confidence", "distribution"):
            tc.assertIn(req_key, block, f"Missing '{req_key}' in '{task}' prediction")

        label = block["label"]
        label_id = block["label_id"]
        confidence = block["confidence"]
        dist = block["distribution"]

        tc.assertIs(type(label), str, f"{task}.label must be native str")
        tc.assertIs(type(label_id), int, f"{task}.label_id must be native int")
        tc.assertIs(type(confidence), float, f"{task}.confidence must be native float")

        tc.assertIn(label_id, expected_map)
        tc.assertEqual(label, expected_map[label_id])
        tc.assertTrue(math.isfinite(confidence), f"{task}.confidence not finite")
        tc.assertGreaterEqual(confidence, 0.0)
        tc.assertLessEqual(confidence, 1.0)

        tc.assertIsInstance(dist, dict)
        tc.assertEqual(set(dist.keys()), set(expected_map.values()))
        dist_sum = 0.0
        for cls_name, prob in dist.items():
            tc.assertIs(type(prob), float, f"{task}.distribution[{cls_name}] must be float")
            tc.assertTrue(math.isfinite(prob), f"{task}.distribution[{cls_name}] not finite")
            tc.assertGreaterEqual(prob, 0.0)
            tc.assertLessEqual(prob, 1.0)
            dist_sum += prob
        tc.assertTrue(
            math.isclose(dist_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4),
            f"{task}.distribution sums to {dist_sum}, expected ~1.0",
        )
        tc.assertTrue(
            math.isclose(confidence, dist[label], rel_tol=1e-5, abs_tol=1e-5),
            f"{task}.confidence={confidence} != distribution[{label!r}]={dist[label]}",
        )


def compute_majority_baseline(
    labels: Sequence[int], num_classes: int
) -> Tuple[float, float]:
    """
    Compute exact majority-class baseline (accuracy, macro_f1) for ground-truth integer labels.

    When every sample is predicted as the single majority class:
      - maj_acc = max_count / N
      - majority class F1 = 2 * maj_acc / (1 + maj_acc)
      - all other (num_classes - 1) classes have F1 = 0
      - maj_macro_f1 = (1 / num_classes) * (2 * maj_acc / (1 + maj_acc))
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    total = len(labels)
    if total == 0:
        return 0.0, 0.0
    counts = Counter(labels)
    maj_count = max(counts.values())
    maj_acc = maj_count / float(total)
    maj_f1 = (2.0 * maj_acc) / (1.0 + maj_acc) if maj_acc > 0.0 else 0.0
    maj_macro_f1 = maj_f1 / float(num_classes)
    return maj_acc, maj_macro_f1


def find_main_button(at: Any, labels: Iterable[str]) -> Any:
    """Find a button in Streamlit AppTest by case-insensitive substring match against `at.button`."""
    candidates = [lbl.lower() for lbl in labels]
    available = []
    for btn in at.button:
        btn_label = getattr(btn, "label", "") or ""
        available.append(btn_label)
        low = btn_label.lower()
        if any(tok in low for tok in candidates):
            return btn
    raise AssertionError(
        f"Button matching any of {tuple(labels)} not found in at.button={available}"
    )


def select_model_tier(at: Any, tier: str) -> Any:
    """
    Select model tier ('baseline', 'transformer', or 'both') in `at.sidebar.radio[0]`
    supporting both Vietnamese and English UI chrome.
    """
    tier_norm = tier.strip().lower()
    token_map = {
        "baseline": ("trung bình", "medium", "baseline"),
        "transformer": ("cao", "high", "phobert", "transformer"),
        "both": ("cả hai", "both"),
    }
    if tier_norm not in token_map:
        raise ValueError(f"Unsupported tier '{tier}'; expected baseline/transformer/both")
    if len(at.sidebar.radio) == 0:
        raise AssertionError("No sidebar radio widget found in AppTest")
    radio = at.sidebar.radio[0]
    tokens = token_map[tier_norm]
    for opt in radio.options:
        low = str(opt).lower()
        if any(tok in low for tok in tokens):
            return radio.set_value(opt).run()
    raise AssertionError(
        f"Could not map tier '{tier}' onto radio options={list(radio.options)}"
    )


def extract_majority_baseline_metrics(metrics_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Flexibly locate majority-class baseline metrics inside `metrics_en.json`
    (`top-level`, `models`, or `metadata`).
    """
    candidate_keys = (
        "majority_baseline",
        "majority_class_baseline",
        "majority",
        "baseline_majority",
    )
    for container in (
        metrics_json,
        metrics_json.get("models", {}) if isinstance(metrics_json.get("models"), dict) else {},
        metrics_json.get("metadata", {}) if isinstance(metrics_json.get("metadata"), dict) else {},
    ):
        for key in candidate_keys:
            val = container.get(key)
            if isinstance(val, dict):
                return val
    return None


def find_en_dataset_split_path(data_dir: str, split_name: str) -> Optional[str]:
    """Return existing `.parquet` or `.csv` path for `split_name` inside `data_dir`."""
    for ext in ("parquet", "csv"):
        cand = os.path.join(data_dir, f"{split_name}.{ext}")
        if os.path.isfile(cand):
            return cand
    return None


def create_tiny_local_hf_checkpoint(
    target_dir: str, max_position_embeddings: int = 258
) -> str:
    """
    Create a tiny, self-contained local RoBERTa-compatible HF checkpoint + tokenizer
    in `target_dir` so `train_phobert` / `evaluate.py` smoke tests run 100% offline in <2s.
    """
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    os.makedirs(target_dir, exist_ok=True)
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "the": 5,
        "lecturer": 6,
        "teacher": 7,
        "explains": 8,
        "clearly": 9,
        "good": 10,
        "great": 11,
        "bad": 12,
        "broken": 13,
        "classroom": 14,
        "wifi": 15,
        "course": 16,
        "curriculum": 17,
        "tuition": 18,
        "okay": 19,
        "thầy": 20,
        "dạy": 21,
        "hay": 22,
        "phòng": 23,
        "học": 24,
        "tốt": 25,
        "kém": 26,
    }
    vocab_path = os.path.join(target_dir, "vocab.json")
    merges_path = os.path.join(target_dir, "merges.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")

    tok = RobertaTokenizer(vocab_file=vocab_path, merges_file=merges_path)
    tok.save_pretrained(target_dir)

    cfg = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
    )
    backbone = RobertaModel(cfg)
    backbone.save_pretrained(target_dir)
    return target_dir


def write_synthetic_vsfc_dir(
    target_dir: str,
    lang: str = "en",
    n_train: int = 24,
    n_val: int = 12,
    n_test: int = 12,
) -> str:
    """
    Write balanced synthetic `train.csv`, `validation.csv`, `test.csv` into `target_dir`
    covering all 3 sentiment classes and 4 topic classes.
    """
    os.makedirs(target_dir, exist_ok=True)
    if lang == "en":
        templates = [
            ("The lecturer explains clearly and helps students.", 2, 0),
            ("The classroom air conditioner is broken and hot.", 0, 2),
            ("The curriculum schedule is generally okay.", 1, 1),
            ("The professor speaks too fast and skips examples.", 0, 0),
            ("Campus Wi-Fi and projectors work very well.", 2, 2),
            ("Course assignments are too packed before finals.", 0, 1),
            ("Practical lab projects improve coding skills.", 2, 1),
            ("Tuition and administrative paperwork are very slow.", 0, 3),
            ("The teacher follows the syllabus slides on time.", 1, 0),
            ("Student support office answered my inquiry promptly.", 2, 3),
            ("Library hours and seating are acceptable overall.", 1, 2),
            ("Registration portal information is standard.", 1, 3),
        ]
    else:
        templates = [
            ("Thầy giảng bài rất nhiệt tình và dễ hiểu.", 2, 0),
            ("Phòng học nóng nực và máy lạnh bị hư.", 0, 2),
            ("Chương trình học nhìn chung tạm ổn.", 1, 1),
            ("Giảng viên nói quá nhanh và khó theo dõi.", 0, 0),
            ("Phòng máy tính mới và mạng wifi rất mạnh.", 2, 2),
            ("Lịch thi và bài tập quá dày đặc.", 0, 1),
            ("Môn học có nhiều bài thực hành hữu ích.", 2, 1),
            ("Thủ tục hành chính giải quyết rất chậm.", 0, 3),
            ("Thầy dạy đúng theo slide và đúng giờ.", 1, 0),
            ("Phòng công tác sinh viên hỗ trợ nhiệt tình.", 2, 3),
            ("Thư viện có chỗ ngồi tương đối ổn.", 1, 2),
            ("Thông báo đăng ký học phần bình thường.", 1, 3),
        ]

    for split_name, count in (("train", n_train), ("validation", n_val), ("test", n_test)):
        csv_path = os.path.join(target_dir, f"{split_name}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "sentiment", "topic"])
            for idx in range(count):
                text, sent, top = templates[idx % len(templates)]
                writer.writerow([f"{text} (sample {split_name} #{idx})", sent, top])
    return target_dir


_BASELINE_CACHE: Dict[str, Any] = {}
_PHOBERT_CACHE: Dict[str, Tuple[Any, Any]] = {}
_UNIT_SUITE_CACHE: Optional[Dict[str, Any]] = None


def get_cached_baseline(ckpt_path: str) -> Any:
    """Load and cache read-only `VietnameseFeedbackBaseline` instance across tests."""
    abs_path = os.path.abspath(ckpt_path)
    if abs_path not in _BASELINE_CACHE:
        from models.baseline_model import VietnameseFeedbackBaseline

        _BASELINE_CACHE[abs_path] = VietnameseFeedbackBaseline.load(abs_path)
    return _BASELINE_CACHE[abs_path]


def get_cached_phobert(ckpt_dir: str) -> Tuple[Any, Any]:
    """Load and cache read-only `(PhoBERTMultiTask, local_tokenizer)` offline across tests."""
    abs_dir = os.path.abspath(ckpt_dir)
    if abs_dir not in _PHOBERT_CACHE:
        from models.phobert_model import PhoBERTMultiTask

        tok = load_local_tokenizer(abs_dir)
        model = PhoBERTMultiTask.from_pretrained(abs_dir, device="cpu")
        model.eval()
        _PHOBERT_CACHE[abs_dir] = (model, tok)
    return _PHOBERT_CACHE[abs_dir]


def run_unit_test_suite_cached(timeout: int = 300) -> Dict[str, Any]:
    """
    Execute `python -m unittest discover -s tests -p "test_*.py"` once per runner
    process and cache the result so Tier 1 F11 and Tier 4 Scenario 6 do not run a
    ~136s subprocess twice in a single E2E run.
    """
    global _UNIT_SUITE_CACHE
    if _UNIT_SUITE_CACHE is not None:
        return _UNIT_SUITE_CACHE

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    _UNIT_SUITE_CACHE = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_s": elapsed,
    }
    return _UNIT_SUITE_CACHE
