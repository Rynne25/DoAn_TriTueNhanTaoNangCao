"""
models/language.py
Lightweight deterministic language detection ('vi' vs 'en') for bilingual student feedback.
"""

import re
import unicodedata
from typing import Optional

# Uniquely Vietnamese precomposed NFC characters (stacked/hooked/horn/bar diacritics)
# not shared with standard English or French loanwords.
STRICT_VI_DIACRITIC_RE = re.compile(
    r"[ăằắẳẵặâầấẩẫậđêềếểễệôồốổỗộơờớởỡợưừứửữựảãạẻẽẹỉĩịỏõọủũụỳỷỹỵ"
    r"ĂẰẮẲẴẶÂẦẤẨẪẬĐÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰẢÃẠẺẼẸỈĨỊỎÕỌỦŨỤỲỶỸỴ]"
)

# Acute/grave diacritics shared with common English loanwords (e.g., café, résumé, à la carte).
SHARED_LATIN_DIACRITIC_RE = re.compile(r"[àáèéìíòóùúýÀÁÈÉÌÍÒÓÙÚÝ]")

_ASCII_WORD_RE = re.compile(r"\b[a-z]+\b")

EN_MARKERS = frozenset({
    "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "and", "but", "or", "not", "very", "too", "so", "really", "quite",
    "good", "great", "excellent", "bad", "poor", "terrible", "slow", "fast",
    "lecturer", "professor", "instructor", "teacher", "course", "class",
    "classroom", "exam", "exams", "homework", "assignment", "assignments",
    "curriculum", "syllabus", "facility", "facilities", "tuition", "wifi",
    "campus", "student", "students", "explain", "explains", "explained",
    "helpful", "broken", "heavy", "unfair", "boring", "clear", "clearly",
    "this", "that", "these", "those", "with", "for", "from", "about",
    "have", "has", "had", "doesn", "don", "didn", "can", "could", "will",
    "would", "should", "there", "their", "they", "we", "our", "you", "your",
    "it", "its", "of", "in", "on", "at", "to", "by", "as",
})

# Strictly unambiguous unaccented Vietnamese tokens (excluding English words/loanwords).
VI_ASCII_MARKERS = frozenset({
    "khong", "duoc", "giang", "vien", "sinh", "truong", "phong", "chuong",
    "trinh", "nhiet", "tinh", "nhieu", "nhung", "chua", "hieu", "thuyet",
    "thuc", "hanh", "lieu", "thay", "giao", "nguoi", "cua", "luon", "cac",
    "nay", "nhat", "thuong", "thoi", "gian", "tuyet", "voi", "kem", "thieu",
    "nong", "sach", "nhanh", "cham", "trach", "nhiem", "kien", "huong",
    "giai", "thich", "cai", "thien", "diem", "thi", "bai", "lanh", "nha",
    "chung", "minh", "viec", "phai", "nhu", "vay", "ngay", "tuan", "thang",
    "lop", "cuon",
})


def detect_language(text: Optional[str]) -> str:
    """
    Deterministically detects whether student feedback text is Vietnamese ('vi') or English ('en').

    Returns 'en' for None, empty/whitespace strings, or English text (including common loanwords
    like 'café' or 'pho'), and 'vi' when Vietnamese diacritics or strong Vietnamese markers appear.
    """
    if text is None:
        return "en"
    raw = str(text).strip()
    if not raw:
        return "en"

    norm = unicodedata.normalize("NFC", raw)
    strict_hits = len(STRICT_VI_DIACRITIC_RE.findall(norm))
    shared_hits = len(SHARED_LATIN_DIACRITIC_RE.findall(norm))

    words = _ASCII_WORD_RE.findall(norm.lower())
    en_hits = sum(1 for w in words if w in EN_MARKERS)
    vi_ascii_hits = sum(1 for w in words if w in VI_ASCII_MARKERS)

    if strict_hits >= 2:
        return "vi"

    if strict_hits == 1:
        if en_hits >= 2 and vi_ascii_hits == 0:
            return "en"
        return "vi"

    if shared_hits >= 1:
        if vi_ascii_hits >= 1 and en_hits == 0:
            return "vi"
        return "en"

    if vi_ascii_hits >= 2 and vi_ascii_hits > en_hits:
        return "vi"

    return "en"
