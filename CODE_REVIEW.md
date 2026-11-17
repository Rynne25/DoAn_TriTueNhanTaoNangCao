# Codebase Review Report (R1): Student Feedback Classification

**Repository:** `d:\Puri i love diu`  
**Scope:** Comprehensive technical audit of all 6 core directories (`models/`, `data/`, `training/`, `evaluation/`, `app/`, `tests/`) plus Root Configuration & Documentation (`README.md`, `PROJECT.md`, `requirements.txt`, `.gitignore`).  
**Objective:** Identify concrete bugs, edge cases, silent failure modes, test couplings, and minimal-disruption modifications required to fix R1 issues and support English student feedback classification alongside Vietnamese without degrading existing Vietnamese behavior or benchmarks.

---

## Executive Summary & Fix Allocation

The existing codebase delivers a solid Vietnamese UIT-VSFC pipeline (Classical TF-IDF + Dual Logistic Regression achieving **72.31% Mean Macro-F1** and PhoBERT Multi-Task achieving **81.50% Mean Macro-F1** on the 3,166-sample test set, with **169 passing unit/integration tests** across 12 test files). However, direct line-by-line auditing uncovered **critical blockers for English support**, **silent error-masking bugs**, **hardcoded dataset statistics**, and **Streamlit state-management traps**:

1. **Pervasive Vietnamese Word Segmentation Assumptions**: `data/dataset.py`, `models/baseline_model.py`, `models/phobert_model.py`, `training/train_baseline.py`, `training/train_phobert.py`, and `evaluation/evaluate.py` default to `word_segment=True` (`pyvi.ViTokenizer`), corrupting English tokens (`"The_lecturer"`, `"isn ' t"`), and neither `baseline.joblib` nor `phobert_multitask_meta.json` persists `default_word_segment` or `language`.
2. **Broken Offline/Non-PhoBERT Tokenizer Resolution**: `models/phobert_model.py:202-204` ignores the local `checkpoint_dir` when `tokenizer=None` (using `self.model_name_or_path` instead) and hardcodes `use_fast=False`; `data/dataset.py:437`, `training/train_phobert.py:319`, and `evaluation/evaluate.py:967` also hardcode `use_fast=False`, and `evaluation/evaluate.py:969` silently falls back to `"vinai/phobert-base-v2"`.
3. **Hardcoded Evaluation & Missing Majority Baseline**: `evaluation/evaluate.py:550-752` hardcodes `"UIT-VSFC"`, exact test label counts (`1409/167/1590`, `2290/572/145/159`), Vietnamese examples, and approximate sizes (`~0.95 MB`, `~515 MB`), reports only `model.pt` size instead of total directory footprint (`get_file_or_dir_size_mb:69-72`), and omits majority-class baseline Accuracy/Macro-F1.
4. **Silent Failure Modes in Data & Training**: `data/download_data.py:145` skips `b"PAR1"` magic-byte validation on cached files; `data/dataset.py:202-221` swallows `auto_download` exceptions; `training/train_phobert.py:596,629-632` crashes on `epochs=0` (`history.json` write before `os.makedirs`) and returns the last-epoch `model` instead of `best_model`.
5. **Streamlit UI State & Caching Bugs (`app/app.py`)**: Unkeyed `st.text_area` (`lines 1896-1903`) drops unsaved manual edits when toggling `ui_lang`, `Clear` (`lines 1925-1928`) mutates session state post-widget (requiring an `on_click` callback before adding `key="user_text"`), and CUDA OOM failover (`lines 1662-1670`) leaves `@st.cache_resource` returning `"cuda"`.
6. **Documentation & Test Coupling Risks**: `README.md:123` and `PROJECT.md:127` document a nonexistent `--save_reports` flag; `PROJECT.md:22-27` lists nonexistent `test_tier*.py` files; `tests/test_evaluation.py` monkeypatches `os.path.isfile`/`isdir` globally and uses rigid `predict_labels` lambdas without `**kwargs`; `tests/test_phobert_stress.py:337-338` requires both `model.pt` and `pytorch_model.bin`.

### Milestone Implementation Allocation

| Milestone | Scope Implemented | Target Files |
|---|---|---|
| **M1 (Current)** | Written Review Report (F1) + Core Bilingual Infrastructure & R1 Fixes (F2–F5) + Fast Offline Core Tests | `CODE_REVIEW.md`, `data/download_data.py`, `data/dataset.py`, `models/baseline_model.py`, `models/phobert_model.py`, `models/language.py`, `models/bilingual.py`, `training/train_baseline.py`, `training/train_phobert.py`, `evaluation/evaluate.py`, `evaluation/__init__.py`, `tests/test_bilingual_core.py` |
| **M2 (Deferred)** | English Dataset Preparation, Tier 1 & Tier 2 English Training, Dual VI+EN Benchmark Execution | `data/prepare_english_data.py`, `data/english_feedback/`, `saved_models/baseline_en.joblib`, `saved_models/english_multitask/`, `results/metrics_en.json`, `results/comparison_summary_en.md` |
| **M3 (Deferred)** | Dual VI+EN Sample Libraries & UI Bug Fixes in Streamlit App; App/E2E/Training Tests & Root Doc Updates | `app/app.py`, new tests in `tests/`, `README.md`, `PROJECT.md`, `requirements.txt`, `.gitignore` |

---

## 1. `models/` Directory Review

### 1.1 `models/baseline_model.py` (286 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `34-45`, `120`, `140`, `153`, `175`, `185`, `201` | **High** | Bug (EN) | `fit`, `predict_proba`, `predict_labels`, and `predict` default to `word_segment=True`, unconditionally calling `pyvi.ViTokenizer.tokenize` (`_segment_text`) on English inputs and joining English words/contractions with underscores. Also, `_segment_text` does not normalize NFD Unicode to NFC. | Add `default_word_segment: bool = True` and `language: str = "vi"` to `__init__`; change method defaults to `word_segment: Optional[bool] = None` resolved via `_resolve_word_segment(word_segment)`. | **Fixed in M1** |
| `70-114`, `269-286` | **High** | Backward Compatibility | Legacy `saved_models/baseline.joblib` contains 15 pickled attributes (`ngram_range`..`is_fitted`) and **lacks** `default_word_segment` and `language`. Direct `self.default_word_segment` access would raise `AttributeError` on unpickled checkpoints. | Always resolve via `getattr(self, "default_word_segment", True)` (never bare `self.default_word_segment`); backfill `default_word_segment=True` and `language="vi"` inside `load()`. | **Fixed in M1** |
| `146-147` | **Low** | Edge Case | While `_calibrate_probs` (`lines 48-59`) handles missing classes at prediction time, `LogisticRegression.fit()` raises `ValueError` if a synthetic/tiny training slice has only 1 unique class. | Documented; callers must provide >=2 classes per task in training slices (handled in dataset validation). | **Documented** |
| `152-182` | **Medium** | Performance Coupling | `tests/test_adversarial_baseline.py:302-326` requires `predict_labels(..., word_segment=False)` over 3,166 samples to exceed **10,000 samples/sec** (<0.1 ms/sample). Adding regex language detection inside `predict_proba` would break this throughput guardrail. | Keep `detect_language()` out of `VietnameseFeedbackBaseline.predict_proba()`; place language routing in `models/language.py` and `models/bilingual.py`. | **Fixed in M1** |

### 1.2 `models/phobert_model.py` (510 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `202-204`, `479-486` | **High** | Bug | When `predict_proba(..., tokenizer=None)` is called on a model loaded via `from_pretrained("saved_models/phobert_multitask")`, `self.model_name_or_path` is `"vinai/phobert-base-v2"` (`line 480`), NOT the local directory. Thus `AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)` skips local tokenizer files and hardcodes `use_fast=False`, failing offline or on fast-only English tokenizers. | Store `self.checkpoint_dir` in `__init__` (populated by `from_pretrained` for local directories); check `checkpoint_dir` with `local_files_only=True` first, then `model_name_or_path`, trying `use_fast=False` -> `use_fast=True` (or `use_fast=True` first for non-PhoBERT). | **Fixed in M1** |
| `175`, `224`, `274`, `303`, `400-409` | **High** | Bug (EN) | `predict_proba`, `predict_labels`, and `predict` default `word_segment=True`, and `phobert_multitask_meta.json` omits `default_word_segment` and `language`. | Add `default_word_segment: bool = True` and `language: str = "vi"` to `__init__` and `phobert_multitask_meta.json`; default to `True`/`"vi"` for legacy checkpoints; resolve `word_segment=None` via `getattr(self, "default_word_segment", True)`. | **Fixed in M1** |
| `222` | **Low** | Edge Case | `range(0, len(input_list), batch_size)` raises `ValueError` if `batch_size <= 0`. | Validate `if batch_size <= 0: raise ValueError(...)` in `predict_proba`. | **Fixed in M1** |
| `206-212`, `232-239` | **Low** | Edge Case | Passing an explicit `device` to `predict_proba` moves input tensors to `device` without checking/moving `self` if `self` is on a different device. | Documented (callers in `evaluate.py` and `app/app.py` call `model.to(device)` explicitly; keep minimal behavior to avoid unexpected side effects in tests). | **Documented** |
| `412-415` | **Medium** | Test Coupling / Footprint | `save_pretrained()` writes BOTH `model.pt` (540.08 MB) and `pytorch_model.bin` (540.10 MB), doubling disk footprint to ~1.03 GB per checkpoint. However, `tests/test_phobert_stress.py:337-338` explicitly asserts both files exist. | **Must keep** writing both files to preserve `test_phobert_stress.py`; ensure `.gitignore` covers future English checkpoints in M3. | **Preserved** |
| `152-160` | **Low** | Discrepancy | `forward()` computes unweighted, unsmoothed `nn.CrossEntropyLoss()`, whereas `training/train_phobert.py:506-515` trains with smoothed class weights (`power=0.35`, `label_smoothing=0.03`). Validation loss (`train_phobert.py:207-212`) uses `model.forward()`, putting train and val loss on slightly different scales. | Documented; preserve `forward()` loss formula as asserted by `tests/test_phobert.py:263-266` and `tests/test_phobert_stress.py:261-283`. | **Documented** |
| `489-492` | **Low** | Security / Robustness | `torch.load(weights_file, map_location=map_location, weights_only=True)` silently falls back to `weights_only=False` on any exception. | Documented; keep fallback only for older PyTorch compatibility without altering checkpoint loading tests. | **Documented** |

### 1.3 New Bilingual Modules (`models/language.py`, `models/bilingual.py`) — Added in M1
- **`models/language.py`**: Implements deterministic, NFC-normalizing `detect_language(text: str) -> "vi" | "en"` distinguishing uniquely Vietnamese stacked diacritics (`STRICT_VI_DIACRITIC_RE`) from shared Latin loanword diacritics (`café`, `résumé`) and unambiguous unaccented Vietnamese tokens vs. English feedback markers (`"The campus café is noisy"` -> `"en"`, `"thay giang bai rat nhiet tinh"` -> `"vi"`, empty/`None` -> `"en"`).
- **`models/bilingual.py`**: Implements `BilingualBaselinePredictor`, `BilingualTransformerPredictor`, `load_tokenizer_safe`, `predict_bilingual_baseline`, and `predict_bilingual_transformer`, grouping batches by language, preserving input order, attaching `"language"` metadata, and gracefully falling back if only one language checkpoint exists on disk.

---

## 2. `data/` Directory Review

### 2.1 `data/download_data.py` (264 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `145` | **Medium** | Bug | Cache validation `os.path.exists(target_parquet) and os.path.getsize(target_parquet) > 1000` does not verify `b"PAR1"` Parquet magic bytes (unlike `_download_stream_to_file` at `lines 80-84`). A >1 KB HTML error page or truncated file passes the cache check. | Add `_is_valid_cached_parquet(path)` checking `os.path.isfile(path)`, `getsize > 1000`, and `f.read(4) == b"PAR1"`. | **Fixed in M1** |
| `104` | **Low** | Debugging | `raise RuntimeError(...)` drops exception chaining (`from last_exception`), losing root traceback context. | Chain `from last_exception`. | **Fixed in M1** |
| `204-225` | **Low** | Observability | `_verify_dataset_summary` silently falls back to byte-size printout when neither `polars` nor `pandas` is installed, without noting that row-count validation was skipped. | Print explicit notice/warning when row-count verification is skipped. | **Fixed in M1** |
| `23-41` | **Info** | Scope (EN) | Hardcoded to Vietnamese `uitnlp/vietnamese_students_feedback` (`11426 / 1583 / 3166`). | Keep VI downloader dedicated to UIT-VSFC; add `data/prepare_english_data.py` in M2. | **Deferred to M2** |

### 2.2 `data/dataset.py` (507 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `149-155`, `242-250`, `403-412` | **High** | Bug (EN) | `load_vsfc_split` and `get_dataloaders` default `word_segment: bool = True` with no `language` parameter, running `pyvi.ViTokenizer` on English text unless callers explicitly override. | Add `language: Optional[str] = None` (validated in `{None, "vi", "en"}`) and default `word_segment: Optional[bool] = None`: explicit `True`/`False` always honored; `None` resolves to `False` for `"en"` (or `"english"` in `os.path.basename(data_dir)`), else `True` (preserving `load_vsfc_split("test")` VI default). | **Fixed in M1** |
| `202-221` | **Medium** | Bug | `auto_download` runs Vietnamese `download_vsfc_dataset` even when `data_dir` targets English data, and `except (ImportError, Exception) as err: pass` discards `err`, masking HTTP/network errors behind a generic `FileNotFoundError`. | Only run `download_vsfc_dataset` when `auto_download and resolved_lang == "vi"`; capture `download_err` and include `f" Auto-download error ({type(download_err).__name__}): {download_err}"` + `from download_err`. | **Fixed in M1** |
| `19-24`, `66-68`, `83-85`, `104-106`, `121-123`, `139-142` | **Medium** | Edge Case | `_read_table_file` casts labels with raw `int(...)` across all 5 Polars/Pandas/CSV branches without null/`NaN`/`bool`/float diagnostics or row index; empty cleaned sentences are silently accepted during split loading. *(Note: `VSFCDataset.from_texts(["", ...])` is explicitly tested in `tests/test_empirical_challenger.py:182-199` and must continue accepting empty strings.)* | Add `_parse_int_label(val, col_name, idx, file_path)` with row/column diagnostics; reject empty cleaned sentences in `load_vsfc_split` only; keep `_read_table_file` signature/return tuple and exact `"Invalid sentiment label"` / `"Invalid topic label"` substrings (`test_empirical_challenger.py:304,323`). | **Fixed in M1** |
| `437` | **High** | Bug (EN) | `get_dataloaders` hardcodes `AutoTokenizer.from_pretrained(tokenizer, use_fast=False)` when `tokenizer` is a string, breaking fast-only tokenizers and slowing RoBERTa tokenizers. | Default `use_fast=None`, auto-selecting `("phobert" not in tokenizer.lower())` with fallback to `not resolved_fast` on exception. | **Fixed in M1** |
| `457-473` | **Medium** | Edge Case | `quick_mode` slices prefix `[:subset_train]`, `[:subset_val]`, `[:subset_test]`, risking single-class subsets if input files are label-sorted. | Deterministic `random.Random(42)` shuffle of indices before slicing when `len > subset` (keeping `subset >= batch_size` per `test_data_pipeline.py:236-253`). | **Fixed in M1** |
| `151`, `407` | **Low** | Edge Case | Relative `"data/uit_vsfc"` fails if invoked outside repo root. | Check `PROJECT_ROOT`-relative fallback ONLY when `data_dir` is default/relative (`"data/uit_vsfc"` or `"data/english_feedback"`) and missing under CWD while existing under `PROJECT_ROOT`. | **Fixed in M1** |

---

## 3. `training/` Directory Review

### 3.1 `training/train_baseline.py` (249 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `90-99`, `123`, `143`, `152`, `161`, `188-199` | **High** | Bug (EN) | Hardcodes `data_dir="data/uit_vsfc"`, `save_path="saved_models/baseline.joblib"`, `word_segment=True`. Worse, `train_baseline` loads pre-segmented text (`word_segment=True`) and passes `word_segment=False` to `model.fit()`, while `VietnameseFeedbackBaseline.predict()` previously defaulted to `True` — causing a silent train/inference mismatch for English models trained without segmentation. | Add `--language {vi,en}` (default `"vi"`), `--no_word_segment`, and resolve `None` defaults for `--data_dir`/`--save_path` by language (`saved_models/baseline_en.joblib`, `data/english_feedback`, `word_segment=False` for EN; VI defaults unchanged). Set `model.default_word_segment` and `model.language`. | **Fixed in M1** |
| `155`, `164`, `166-175` | **Low** | Observability | Discards `val_metrics` and `test_metrics` returned by `evaluate_split()` and writes no companion metadata JSON. | Capture `val_metrics`/`test_metrics` and write `<stem>_meta.json` when `language == "en"` or `save_path != "saved_models/baseline.joblib"` (preserving `baseline.joblib` contract). | **Fixed in M1** |

### 3.2 `training/train_phobert.py` (769 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `594-596`, `632` | **Medium** | Bug | Reloads `best_model` for test evaluation at `line 596`, but returns the last-epoch `model` at `line 632` instead of `best_model`. | Initialize `best_model = model`, reload `best_model` whenever `best_epoch > 0 and os.path.isdir(save_dir)` (independent of `eval_test`), and return `best_model, summary`. | **Fixed in M1** |
| `565`, `629-630` | **Medium** | Bug | `os.makedirs(save_dir, exist_ok=True)` only runs when a validation checkpoint improves (`line 565`). With `epochs=0`, writing `history.json` (`line 629`) crashes with `FileNotFoundError`. | Call `os.makedirs(save_dir, exist_ok=True)` unconditionally before writing `history.json`. | **Fixed in M1** |
| `282-285`, `319`, `327`, `700-715` | **High** | Scope (EN) | Hardcodes `model_name="vinai/phobert-base-v2"`, `save_dir="saved_models/phobert_multitask"`, `data_dir="data/uit_vsfc"`, `use_fast=False`, and `word_segment=True`. | Add `--language {vi,en}` (default `"vi"`), `--no_word_segment`, and conditional EN defaults (`distilroberta-base`, `saved_models/english_multitask`, `data/english_feedback`, `word_segment=False`). Load tokenizer with `use_fast=("phobert" not in resolved_model_name.lower())` + fast/slow fallback; persist `"language"` & `"word_segment"` in `training_meta.json` and model attributes. | **Fixed in M1** |
| `113`, `120-130`, `345-349`, `451-470` | **Medium** | Architecture Risk | `_extract_frozen_layer_cache` and cached top-layer training assume `model.phobert.encoder.layer`, `model.phobert._create_attention_masks`, and tensor-only layer outputs (failing on models returning `(hidden,)` tuples or lacking `.encoder`). Also, `freeze_bottom_layers=-1` at `line 347` only freezes when `total_encoder_layers >= 12`, resolving to `0` on 6-layer `distilroberta-base`. | Guard `use_cached_top` on `hasattr(model.phobert, "encoder") and hasattr(model.phobert, "_create_attention_masks")`; unwrap tuple layer outputs (`out[0] if isinstance(out, tuple) else out`); resolve `-1` on CPU to `6` when `>= 12` layers and `total_encoder_layers // 2` when `6 <= total_encoder_layers < 12`. | **Fixed in M1** |
| `458-460`, `373` | **Low** | Minor | `rng.shuffle(cached_batches)` shuffles batch order only (intra-batch pairings stay fixed across epochs); `quick_run=True` writes to `save_dir` if not overridden. | Documented (preserving deterministic bucketed speed). | **Documented** |

---

## 4. `evaluation/` Directory Review (`evaluation/evaluate.py` 1,089 lines, `evaluation/__init__.py` 26 lines)

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required | Status |
|---|---|---|---|---|---|
| `746-752` | **High** | Bug | `generate_json_report` hardcodes `"name": "UIT-VSFC"`, `"sentiment_distribution": {"negative": 1409, "neutral": 167, "positive": 1590}`, and `"topic_distribution": {"lecturer": 2290, "training_program": 572, "facility": 145, "others": 159}` regardless of split (`validation`/`train`), `--quick_eval`, or language. | Compute distributions dynamically from `sentiments`/`topics` or `per_class` supports; parameterize `dataset_name` and `language` while keeping 4-arg & 7-arg signatures intact. | **Fixed in M1** |
| `450`, `457`, `550-552`, `641-660`, `708` | **High** | Bug | `format_terminal_table` and `generate_markdown_report` hardcode `"UIT-VSFC"`, `"PhoBERT"`, Vietnamese sarcasm examples, PyVi commentary, exact test percentages (`50.22%`, `5.27%`, `72.33%`), hardcoded sizes (`~0.95 MB`, `~515 MB`), and CLI string `python evaluation/evaluate.py --split test`. | Parameterize `language`/`dataset_name`/`split_name`/`cli_command`; compute class percentages & sizes dynamically; keep `"Classical Baseline"` and `"PhoBERT"` strings for default VI reports (`test_evaluation.py:303-304,776`). | **Fixed in M1** |
| `69-72` | **Medium** | Bug | `get_file_or_dir_size_mb` returns on the first match of `("model.pt", "pytorch_model.bin")`, reporting `515.06 MB` for `saved_models/phobert_multitask/` instead of the actual `~1,032 MB` directory footprint (`model.pt` + `pytorch_model.bin` + tokenizer). | Sum all files recursively via `os.walk(path)`, wrapping `os.path.getsize` safely against `test_evaluation.py:541,848` global `os.path.isfile` mocks. | **Fixed in M1** |
| `803-872`, `923-927`, `966-969`, `1028-1029` | **High** | Bug (EN) / Gap | Missing `--language`, `--no_word_segment`, `--metrics_out`, `--markdown_out`, and majority-class baseline Accuracy/Macro-F1. Forces `word_segment=True`, `use_fast=False`, and silently falls back to `"vinai/phobert-base-v2"` tokenizer on any load error. | Keep VI defaults in `build_arg_parser()` (`test_evaluation.py:399-403`), resolve EN defaults in `parse_args`/`run_evaluation`, add `evaluate_majority_baseline()`, detect `use_fast` safely, and disallow silent VI fallback when `language == "en"`. | **Fixed in M1** |
| `140` | **Medium** | Test Coupling | `tests/test_evaluation.py:622,706,811` patches `predict_labels` with `lambda b, word_segment=False: (...)` (no `**kwargs`). | **Must preserve** `model.predict_labels(batch_texts, word_segment=False)` call with zero extra kwargs. | **Preserved** |

---

## 5. `app/` Directory Review (`app/app.py`, 2,019 lines — Deferred to M3)

*Note: `app/app.py` is owned by Milestone M3 and was audited read-only in M1.*

| Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required (for M3) |
|---|---|---|---|---|
| `1896-1903`, `1857-1859`, `1925-1928` | **High** | UI State Bug | `st.text_area` has `value=st.session_state["user_text"]` and **no `key=`**. Toggling `ui_lang` changes `label`/`placeholder`/`help`, regenerating the widget ID and **wiping unsaved manual text**. Also, re-clicking `"Load Preset"` on the same preset after manual edits fails to reset the textarea. **M3 Trap:** `Clear` (`line 1926`) mutates `st.session_state["user_text"] = ""` *after* `st.text_area` is rendered; adding `key="user_text"` requires moving `Clear` to an `on_click` callback (or pre-widget flag + `st.rerun()`) to avoid `StreamlitAPIException`. | Bind `st.text_area(..., key="user_text")`, move `Clear` to `on_click=_clear_user_text`, and update `st.session_state["user_text"]` in `"Load Sample"` callbacks before rerun. |
| `560-592`, `1651-1671`, `1979`, `2003` | **High** | Runtime Bug | `_load_phobert_pipeline_cached` caches `(model, tokenizer, "cuda")`. On CUDA OOM, `_predict_phobert_safe` (`lines 1662-1670`) moves `phobert_model.to("cpu")` in-place and returns `"cpu"`, but the cached tuple keeps `"cuda"`, causing device-mismatch `RuntimeError` and spurious OOM warnings on every subsequent prediction. | Check actual `next(model.parameters()).device` inside `_predict_phobert_safe` before moving tensors / emitting OOM warnings. |
| `541-557`, `570-592` | **Medium** | Observability / Cache | Broad `except Exception` in `load_baseline_model` and `_load_phobert_pipeline_cached` masks deserialization/tokenizer errors as `"Thiếu weights"`; `_get_phobert_weights_mtime` checks only the first of `("model.pt", "pytorch_model.bin")`, ignoring `config.json`, `phobert_multitask_meta.json`, or `.safetensors`. | Surface concise load error details in logs/UI expander; hash mtimes across all relevant weight/config/meta files. |
| `1816-1818`, `1945-2016` | **Medium** | UX Bug | When both checkpoints are missing, `selected_mode_key` is `None` and clicking Analyze enters `if should_render_results:` without matching any branch, producing a silent no-op. | Render an explicit warning banner when `selected_mode_key is None`. |
| `40-41`, `189-205`, `213-216`, `247`, `277-304`, `422-528`, `1315`, `1586-1600`, `1905-1916`, `1965-1990` | **High** | Scope (R2/R3) & Hardcoding | Hardcodes `model_size_mb=0.95`/`515.06`, `device_info="CPU"`, VI-only paths (`DEFAULT_BASELINE_PATH`, `DEFAULT_PHOBERT_DIR`), single `get_presets()` VI sample library, and VI-only UI copy/distributions. | Keep `get_presets()` (10 VI samples) at `sidebar.selectbox[0]` + `sidebar.button[0]`; add `get_english_presets()` (>=10 EN samples across all 3 sentiments & 4 topics) at `sidebar.selectbox[1]` + `sidebar.button[1]`; compute disk sizes & device dynamically; route predictions via `models.bilingual` / `detect_language`. |
| `1355`, `1382`, `1536` vs `1914` | **Low** | Security Hygiene | `render_prediction_card` and `render_distribution_bars` interpolate labels into raw HTML without `html.escape()` (currently safe because inputs are trusted `I18N` strings, unlike `matched_preset['text_en']` which uses `html.escape()`). | Apply `html.escape()` consistently to dynamic strings in HTML templates. |
| `sidebar.*[0]`, `get_presets()` | **High** | Test Coupling | `tests/test_app.py:134-202,385-519`, `tests/test_challenger_m5.py:118-138`, `tests/test_adversarial_m5.py:118`, and `tests/test_challenger_m5_it2_empirical.py:62-63` strictly depend on `len(get_presets()) == 10`, `sidebar.button_group[0/1]`, `sidebar.radio[0]`, `sidebar.selectbox[0]`, `sidebar.button[0]`, `text_area[0]`, and exact Vietnamese status/warning strings. | Preserve exact `get_presets()` contents and existing `sidebar.*[0]` widget positions; append English Sample Library as `selectbox[1]` + `button[1]`. |

---

## 6. `tests/` Directory Review (12 Files, 169 Verified Tests)

*Note: Existing 12 test files are kept 100% untouched in M1. Test fixes/expansions for `app/app.py` and `training/` are deferred to M3, while M1 adds `tests/test_bilingual_core.py` for F2–F5.*

| File | Line(s) | Severity | Category | Verified Finding & Impact |
|---|---|---|---|---|
| `tests/test_empirical_challenger.py` | `312-317` | **Low** | Bug | Duplicate `"topic": [5]` dictionary key in `pl.DataFrame({...})` literal. |
| `tests/test_evaluation.py` | `541`, `848` | **Medium** | Fragile Mock | Patches `os.path.isfile` and `os.path.isdir` globally to `True` during `eval_module.main()`, making any `if os.path.isfile(x): open(x)` in `evaluate.py` fail unless wrapped in `try/except OSError`. |
| `tests/test_evaluation.py` | `622`, `706`, `811` | **Medium** | Rigid Mock | Mocks `predict_labels` with `lambda b, word_segment=False: (...)` without `**kwargs`, forcing `evaluate_baseline` to pass no extra kwargs. |
| `tests/test_evaluation.py` | `284-295`, `325-336`, `377-382` | **Low** | Anti-pattern | Catches `TypeError` around 7-arg report calls and falls back to 4-arg calls, masking genuine `TypeError` bugs inside report helpers. |
| `tests/test_phobert.py`, `tests/test_empirical_challenger.py`, `tests/test_phobert_stress.py` | `116`, `517`; `54`, `123`, `180`, `333`; `325` | **Medium** | Network / Offline Risk | Several `AutoConfig.from_pretrained` / `AutoTokenizer.from_pretrained` calls omit `local_files_only=True` (and `test_phobert_stress.py:325` hardcodes `use_fast=False`), risking hub network calls in offline environments. |
| `tests/test_adversarial_m5.py` | `185-228` | **Medium** | Resource Leak / Flaky | `subprocess.Popen(..., stdout=PIPE, stderr=PIPE)` never closes `proc.stdout`/`proc.stderr` in `finally:` (`ResourceWarning` on Python 3.12), and hardcodes port `8503` + strict `<= 5.0s` startup timeout. |
| `tests/test_app.py`, `tests/test_challenger_m5.py` | `97-125`; `94-114`, `500-509` | **Medium** | Cache / Patch Leakage | Global monkeypatching of `PhoBERTMultiTask.from_pretrained` / `AutoTokenizer.from_pretrained` alongside `@st.cache_resource`; `TestAppControlsAndClearAction` never calls `.clear()` on Streamlit resource caches. |
| `tests/test_app.py`, `test_challenger_m5.py`, `test_adversarial_m5.py`, `test_challenger_m5_it2_empirical.py` | Multiple | **Medium** | Brittle Coupling | Positional indexing (`sidebar.selectbox[0]`, `sidebar.button[0]`) and hardcoded Vietnamese labels/warnings constrain `app/app.py` layout changes. |
| `tests/test_baseline.py`, `tests/test_adversarial_baseline.py`, `tests/test_challenger_m4.py` | `404-408`; `246`, `318-325`; `351-355` | **Info** | Artifact Coupling | Hardcodes `baseline.joblib` vocab size (`10,000`), byte size (`<2 MB`), CPU throughput (`>10,000 samples/sec`), and exact VI test metrics (`72.31%` Mean F1); `saved_models/baseline.joblib` and `results/metrics.json` must remain untouched. |
| `tests/test_data_pipeline.py` | `37-54`, `62-63` | **Low** | Test Quality | `DummyPhoBERTTokenizer` returns random `torch.randint` ignoring padding/truncation; `TestDataAcquisition` runs against live `data/uit_vsfc/` (`force=False`), never exercising retries/fallbacks/`PAR1` rejection. |
| `tests/` (overall) | — | **High** | Coverage Gap | **0% test coverage** for `training/train_baseline.py`, `training/train_phobert.py`, `data/download_data.py` retry/corruption paths, `PhoBERTMultiTask.predict_proba(..., tokenizer=None)`, unmocked transformer `AppTest`, and all bilingual/English workflows. *(M1 adds `tests/test_bilingual_core.py` for F2–F5; M3 adds training/app/E2E tests.)* |

---

## 7. Root Configuration & Documentation Review (Deferred to M3)

| File | Line(s) | Severity | Category | Verified Finding & Impact | Minimal Modification Required (for M3) |
|---|---|---|---|---|---|
| `README.md` | `123` | **Medium** | Doc Bug | Documents `python evaluation/evaluate.py --split test --save_reports`, which fails with `unrecognized arguments: --save_reports` (`evaluate.py` saves by default and uses `--no_save`). | Replace `--save_reports` with accurate CLI flags, update UI option names (`Trung bình`/`Cao`/`Cả hai` & `Medium`/`High`/`Both`), and document English training/eval/sample library commands. |
| `README.md` | `177-180` | **Low** | Doc Drift | Documents radio labels as `"Classical Baseline"`, `"PhoBERT Multi-Task"`, `"Side-by-Side Comparison"` instead of the actual `"Trung bình"`/`"Cao"`/`"Cả hai"` (`"Medium"`/`"High"`/`"Both"`). | Update UI label docs in M3. |
| `PROJECT.md` (repo root) | `22-27`, `39`, `63`, `89`, `127`, `144` | **Medium** | Doc Drift | Lists nonexistent test runner and test tier files; states `8` presets instead of `10`; documents `load_vsfc_split(..., cache_dir=...)` instead of `data_dir=...`; documents nonexistent `--save_reports`; claims dependencies are `"Pinned"` while `requirements.txt` uses `>=`. | Sync `PROJECT.md` with bilingual architecture, real `tests/` layout, dual preset counts, and valid CLI flags in M3. |
| `requirements.txt` | `1-30` | **Medium** | Packaging | Uses loose `>=` lower bounds (risking `transformers` / `scikit-learn` `joblib` version drift) and omits `sentencepiece`, `protobuf`, and `safetensors` needed by non-PhoBERT / SentencePiece / MarianMT models on clean installs. | Add `sentencepiece`, `protobuf`, `safetensors` (and consider upper-bound/tested pins) in M3. |
| `.gitignore` | `19-21` | **High** | Repo Hygiene | Ignores `*.pt`, `*.bin`, `*.safetensors` **only** under `saved_models/phobert_multitask/`, not under future `saved_models/english_multitask/` (>100 MB GitHub push limit). Also lists unused `.pytest_cache/`. | Generalize to `saved_models/*/*.pt`, `saved_models/*/*.bin`, `saved_models/*/*.safetensors` before committing/pushing M2 weights (or in M3). |
