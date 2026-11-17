"""
models/bilingual.py
Unified Bilingual (Vietnamese + English) Inference Wrappers for Tier 1 and Tier 2 Models.
Routes inputs deterministically via detect_language(text) while preserving batch ordering
and conforming strictly to PROJECT.md § Interface Contracts.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from models.baseline_model import SENTIMENT_MAP, TOPIC_MAP, VietnameseFeedbackBaseline
from models.language import detect_language
from models.phobert_model import PhoBERTMultiTask


def load_tokenizer_safe(
    model_dir_or_name: str, local_files_only: Optional[bool] = None
) -> Any:
    """
    Safely loads a Hugging Face tokenizer with automatic fast/slow fallback.
    Uses local_files_only=True first when loading from an existing local directory.
    """
    from transformers import AutoTokenizer

    target = str(model_dir_or_name)
    is_local = os.path.isdir(target)
    prefer_fast = "phobert" not in target.lower()
    if is_local and os.path.isfile(os.path.join(target, "bpe.codes")):
        prefer_fast = False

    if local_files_only is None:
        local_try_order = [True, False] if is_local else [False]
    else:
        local_try_order = [bool(local_files_only)]

    last_err: Optional[Exception] = None
    for lfo in local_try_order:
        for use_fast in (prefer_fast, not prefer_fast):
            try:
                return AutoTokenizer.from_pretrained(
                    target, use_fast=use_fast, local_files_only=lfo
                )
            except Exception as exc:
                last_err = exc
    raise RuntimeError(f"Failed to load tokenizer from '{target}': {last_err}") from last_err


def _normalize_batch(
    texts: Union[str, List[Optional[str]], None]
) -> Tuple[bool, List[str]]:
    if texts is None or isinstance(texts, str):
        return True, ["" if texts is None else texts]
    raw_list = list(texts)
    return False, ["" if item is None else str(item) for item in raw_list]


def _normalize_explicit_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    norm = str(language).strip().lower()
    if norm not in {"vi", "en"}:
        raise ValueError(f"Unsupported language '{language}'. Must be 'vi' or 'en'.")
    return norm


class BilingualBaselinePredictor:
    """
    Language-routed Tier 1 Classical Baseline predictor supporting Vietnamese and English.
    Gracefully falls back to whichever single checkpoint exists if only one is available.
    """

    def __init__(
        self,
        vi_path: Optional[str] = "saved_models/baseline.joblib",
        en_path: Optional[str] = "saved_models/baseline_en.joblib",
        lazy: bool = False,
        vi_model: Optional[VietnameseFeedbackBaseline] = None,
        en_model: Optional[VietnameseFeedbackBaseline] = None,
    ):
        self.vi_path = vi_path
        self.en_path = en_path
        self.vi_model: Optional[VietnameseFeedbackBaseline] = vi_model
        self.en_model: Optional[VietnameseFeedbackBaseline] = en_model
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.vi_model is None and self.vi_path and os.path.isfile(self.vi_path):
            self.vi_model = VietnameseFeedbackBaseline.load(self.vi_path)
        if self.en_model is None and self.en_path and os.path.isfile(self.en_path):
            self.en_model = VietnameseFeedbackBaseline.load(self.en_path)

    def _select_model_for_lang(self, lang: str) -> Tuple[VietnameseFeedbackBaseline, str]:
        self._ensure_loaded()
        if lang == "vi":
            if self.vi_model is not None:
                return self.vi_model, "vi"
            if self.en_model is not None:
                return self.en_model, "en"
        else:
            if self.en_model is not None:
                return self.en_model, "en"
            if self.vi_model is not None:
                return self.vi_model, "vi"
        raise FileNotFoundError(
            f"No baseline checkpoint available (checked vi='{self.vi_path}', en='{self.en_path}')."
        )

    def predict_proba(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, input_list = _normalize_batch(texts)
        if len(input_list) == 0:
            return (
                np.empty((0, len(SENTIMENT_MAP)), dtype=np.float64),
                np.empty((0, len(TOPIC_MAP)), dtype=np.float64),
            )

        explicit_lang = _normalize_explicit_language(language)
        detected_langs = [explicit_lang or detect_language(t) for t in input_list]

        grouped_indices: Dict[str, List[int]] = {"vi": [], "en": []}
        for idx, lang in enumerate(detected_langs):
            grouped_indices[lang].append(idx)

        sent_probs = np.zeros((len(input_list), len(SENTIMENT_MAP)), dtype=np.float64)
        topic_probs = np.zeros((len(input_list), len(TOPIC_MAP)), dtype=np.float64)

        for lang, idxs in grouped_indices.items():
            if not idxs:
                continue
            model, _ = self._select_model_for_lang(lang)
            sub_texts = [input_list[i] for i in idxs]
            sub_s, sub_t = model.predict_proba(sub_texts, word_segment=None)
            for local_i, orig_i in enumerate(idxs):
                sent_probs[orig_i] = sub_s[local_i]
                topic_probs[orig_i] = sub_t[local_i]

        return sent_probs, topic_probs

    def predict_labels(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        sent_probs, topic_probs = self.predict_proba(texts, language=language)
        if len(sent_probs) == 0:
            return [], []
        return (
            np.argmax(sent_probs, axis=1).astype(int).tolist(),
            np.argmax(topic_probs, axis=1).astype(int).tolist(),
        )

    def predict(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
        include_language: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_single, input_list = _normalize_batch(texts)
        if len(input_list) == 0:
            return []

        explicit_lang = _normalize_explicit_language(language)
        detected_langs = [explicit_lang or detect_language(t) for t in input_list]

        grouped_indices: Dict[str, List[int]] = {"vi": [], "en": []}
        for idx, lang in enumerate(detected_langs):
            grouped_indices[lang].append(idx)

        ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(input_list)
        for lang, idxs in grouped_indices.items():
            if not idxs:
                continue
            model, _ = self._select_model_for_lang(lang)
            sub_texts = [input_list[i] for i in idxs]
            sub_preds = model.predict(sub_texts, word_segment=None)
            for local_i, orig_i in enumerate(idxs):
                item = dict(sub_preds[local_i])
                if include_language:
                    item["language"] = detected_langs[orig_i]
                ordered_results[orig_i] = item

        final_results: List[Dict[str, Any]] = [r for r in ordered_results if r is not None]
        return final_results[0] if is_single else final_results


class BilingualTransformerPredictor:
    """
    Language-routed Tier 2 Transformer predictor supporting PhoBERT (VI) and English RoBERTa-family.
    Gracefully falls back to whichever single checkpoint exists if only one is available.
    """

    def __init__(
        self,
        vi_dir: Optional[str] = "saved_models/phobert_multitask",
        en_dir: Optional[str] = "saved_models/english_multitask",
        device: Optional[Any] = None,
        lazy: bool = False,
        vi_model: Optional[PhoBERTMultiTask] = None,
        vi_tokenizer: Any = None,
        en_model: Optional[PhoBERTMultiTask] = None,
        en_tokenizer: Any = None,
    ):
        self.vi_dir = vi_dir
        self.en_dir = en_dir
        self.device = device
        self.vi_model = vi_model
        self.vi_tokenizer = vi_tokenizer
        self.en_model = en_model
        self.en_tokenizer = en_tokenizer
        if not lazy:
            self._ensure_loaded()

    @staticmethod
    def _has_weights_dir(directory: Optional[str]) -> bool:
        if not directory or not os.path.isdir(directory):
            return False
        return any(
            os.path.isfile(os.path.join(directory, fname))
            for fname in ("model.pt", "pytorch_model.bin", "model.safetensors")
        )

    def _ensure_loaded(self) -> None:
        if self.vi_model is None and self._has_weights_dir(self.vi_dir):
            assert self.vi_dir is not None
            self.vi_model = PhoBERTMultiTask.from_pretrained(self.vi_dir, device=self.device)
            if self.vi_tokenizer is None:
                self.vi_tokenizer = load_tokenizer_safe(self.vi_dir)
        elif self.vi_model is not None and self.vi_tokenizer is None and self.vi_dir:
            self.vi_tokenizer = load_tokenizer_safe(self.vi_dir)

        if self.en_model is None and self._has_weights_dir(self.en_dir):
            assert self.en_dir is not None
            self.en_model = PhoBERTMultiTask.from_pretrained(self.en_dir, device=self.device)
            if self.en_tokenizer is None:
                self.en_tokenizer = load_tokenizer_safe(self.en_dir)
        elif self.en_model is not None and self.en_tokenizer is None and self.en_dir:
            self.en_tokenizer = load_tokenizer_safe(self.en_dir)

    def _select_model_for_lang(self, lang: str) -> Tuple[PhoBERTMultiTask, Any, str]:
        self._ensure_loaded()
        if lang == "vi":
            if self.vi_model is not None:
                return self.vi_model, self.vi_tokenizer, "vi"
            if self.en_model is not None:
                return self.en_model, self.en_tokenizer, "en"
        else:
            if self.en_model is not None:
                return self.en_model, self.en_tokenizer, "en"
            if self.vi_model is not None:
                return self.vi_model, self.vi_tokenizer, "vi"
        raise FileNotFoundError(
            f"No transformer checkpoint available (checked vi='{self.vi_dir}', en='{self.en_dir}')."
        )

    def predict_proba(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
        device: Optional[Any] = None,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, input_list = _normalize_batch(texts)
        if len(input_list) == 0:
            return (
                np.empty((0, len(SENTIMENT_MAP)), dtype=np.float64),
                np.empty((0, len(TOPIC_MAP)), dtype=np.float64),
            )

        explicit_lang = _normalize_explicit_language(language)
        detected_langs = [explicit_lang or detect_language(t) for t in input_list]

        grouped_indices: Dict[str, List[int]] = {"vi": [], "en": []}
        for idx, lang in enumerate(detected_langs):
            grouped_indices[lang].append(idx)

        sent_probs = np.zeros((len(input_list), len(SENTIMENT_MAP)), dtype=np.float64)
        topic_probs = np.zeros((len(input_list), len(TOPIC_MAP)), dtype=np.float64)

        eff_device = device if device is not None else self.device
        for lang, idxs in grouped_indices.items():
            if not idxs:
                continue
            model, tok, _ = self._select_model_for_lang(lang)
            sub_texts = [input_list[i] for i in idxs]
            sub_s, sub_t = model.predict_proba(
                sub_texts,
                tokenizer=tok,
                device=eff_device,
                max_length=max_length,
                batch_size=batch_size,
                word_segment=None,
            )
            for local_i, orig_i in enumerate(idxs):
                sent_probs[orig_i] = sub_s[local_i]
                topic_probs[orig_i] = sub_t[local_i]

        return sent_probs, topic_probs

    def predict_labels(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
        device: Optional[Any] = None,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> Tuple[List[int], List[int]]:
        sent_probs, topic_probs = self.predict_proba(
            texts,
            language=language,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        if len(sent_probs) == 0:
            return [], []
        return (
            np.argmax(sent_probs, axis=1).astype(int).tolist(),
            np.argmax(topic_probs, axis=1).astype(int).tolist(),
        )

    def predict(
        self,
        texts: Union[str, List[Optional[str]], None],
        language: Optional[str] = None,
        device: Optional[Any] = None,
        max_length: int = 128,
        batch_size: int = 32,
        include_language: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_single, input_list = _normalize_batch(texts)
        if len(input_list) == 0:
            return []

        explicit_lang = _normalize_explicit_language(language)
        detected_langs = [explicit_lang or detect_language(t) for t in input_list]

        grouped_indices: Dict[str, List[int]] = {"vi": [], "en": []}
        for idx, lang in enumerate(detected_langs):
            grouped_indices[lang].append(idx)

        ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(input_list)
        eff_device = device if device is not None else self.device
        for lang, idxs in grouped_indices.items():
            if not idxs:
                continue
            model, tok, _ = self._select_model_for_lang(lang)
            sub_texts = [input_list[i] for i in idxs]
            sub_preds = model.predict(
                sub_texts,
                tokenizer=tok,
                device=eff_device,
                max_length=max_length,
                batch_size=batch_size,
                word_segment=None,
            )
            for local_i, orig_i in enumerate(idxs):
                item = dict(sub_preds[local_i])
                if include_language:
                    item["language"] = detected_langs[orig_i]
                ordered_results[orig_i] = item

        final_results: List[Dict[str, Any]] = [r for r in ordered_results if r is not None]
        return final_results[0] if is_single else final_results


def predict_bilingual_baseline(
    texts: Union[str, List[Optional[str]], None],
    vi_path: Optional[str] = "saved_models/baseline.joblib",
    en_path: Optional[str] = "saved_models/baseline_en.joblib",
    language: Optional[str] = None,
    include_language: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience one-shot helper for bilingual Tier 1 baseline inference."""
    predictor = BilingualBaselinePredictor(vi_path=vi_path, en_path=en_path)
    return predictor.predict(texts, language=language, include_language=include_language)


def predict_bilingual_transformer(
    texts: Union[str, List[Optional[str]], None],
    vi_dir: Optional[str] = "saved_models/phobert_multitask",
    en_dir: Optional[str] = "saved_models/english_multitask",
    device: Optional[Any] = None,
    max_length: int = 128,
    batch_size: int = 32,
    language: Optional[str] = None,
    include_language: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience one-shot helper for bilingual Tier 2 transformer inference."""
    predictor = BilingualTransformerPredictor(vi_dir=vi_dir, en_dir=en_dir, device=device)
    return predictor.predict(
        texts,
        language=language,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        include_language=include_language,
    )
