"""Suspect words: the OCR output a human should look at before anything is filled in.

Acrobat calls these "suspects", and the idea carries over exactly.  Two things make a
recognized word suspect:

* **Low confidence.** The engine itself said so, below
  :attr:`~zfp.core.config.OcrConfig.min_word_confidence`.
* **An implausible letter/digit mix.** ``"l23"`` and ``"HEL1O"`` are not confident/
  unconfident questions at all -- the engine can be perfectly sure and still be wrong,
  because ``l``/``1``/``I``, ``O``/``0``, ``S``/``5`` and ``B``/``8`` are the same shapes
  in most fonts.  A token whose *only* letters are confusable and which is otherwise
  digits (or the mirror case) is almost certainly one character away from correct.

The second check is what earns its keep on forms: account numbers, policy numbers, dates
and postcodes are exactly the fields where one confused glyph is both invisible and
expensive, and a value that reaches a PDF is a value someone will rely on.

For each suspect, :func:`suggest_alternatives` produces the obvious confusion-set
variants -- whole-token normalizations first (the all-digits and all-letters readings),
then single-character substitutions -- so a reviewer picks rather than types, and
:func:`apply_correction` writes the choice back without mutating anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import median
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..core.types import RasterWord
from .engine import resolve_ocr_config

if TYPE_CHECKING:  # pragma: no cover - types only; suspects never imports the cascade
    from .cascade import OcrResult

__all__ = [
    "CONFUSION_MAP",
    "CONFUSABLE_CHARS",
    "MAX_ALTERNATIVES",
    "REASON_LOW_CONFIDENCE",
    "REASON_MIXED_ALNUM",
    "HISTOGRAM_BUCKETS",
    "Suspect",
    "suggest_alternatives",
    "is_implausible_mix",
    "find_suspects",
    "apply_correction",
    "confidence_report",
]

_log = get_logger(__name__)

#: Character -> the characters it is routinely confused with, in preference order.
#: Deliberately small: every extra pair costs precision on every page.
CONFUSION_MAP: Dict[str, Tuple[str, ...]] = {
    "l": ("1", "I"),
    "I": ("1", "l"),
    "1": ("l", "I"),
    "O": ("0",),
    "o": ("0",),
    "0": ("O", "o"),
    "S": ("5",),
    "s": ("5",),
    "5": ("S", "s"),
    "B": ("8",),
    "8": ("B",),
    "Z": ("2",),
    "z": ("2",),
    "2": ("Z", "z"),
    "G": ("6",),
    "6": ("G",),
}
#: Every character that appears in :data:`CONFUSION_MAP`.
CONFUSABLE_CHARS = frozenset(CONFUSION_MAP)
#: Ceiling on how many variants :func:`suggest_alternatives` will offer.
MAX_ALTERNATIVES = 16

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_MIXED_ALNUM = "mixed_letters_digits"

#: Bucket edges for :func:`confidence_report`'s histogram.
HISTOGRAM_BUCKETS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass
class Suspect:
    """One recognized word that should not be trusted without a look.

    Attributes:
        word: The word as the engine read it.
        alternatives: Candidate readings, best first; the engine's own alternatives come
            before the generated confusion-set variants.
        reason: Comma-separated reason codes (:data:`REASON_LOW_CONFIDENCE`,
            :data:`REASON_MIXED_ALNUM`).
        index: Position of ``word`` in the list :func:`find_suspects` was given, so
            :func:`apply_correction` can be called without searching for it again.
    """

    word: RasterWord
    alternatives: List[str] = field(default_factory=list)
    reason: str = ""
    index: int = -1

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "word": self.word.as_dict(),
            "alternatives": list(self.alternatives),
            "reason": self.reason,
            "index": self.index,
        }


# ======================================================================================
# Confusion sets
# ======================================================================================


def _digit_form(char: str) -> str:
    """The digit reading of ``char``, or ``char`` when it has none."""
    if char.isdigit():
        return char
    for option in CONFUSION_MAP.get(char, ()):
        if option.isdigit():
            return option
    return char


def _letter_form(char: str) -> str:
    """The letter reading of ``char``, or ``char`` when it has none."""
    if char.isalpha():
        return char
    for option in CONFUSION_MAP.get(char, ()):
        if option.isalpha():
            return option
    return char


def suggest_alternatives(text: str, max_alternatives: int = MAX_ALTERNATIVES) -> List[str]:
    """Generate the obvious confusion-set readings of ``text``.

    The order is the order a reviewer wants them in:

    1. the all-digits reading, when every character can be read as a digit
       (``"l23"`` -> ``"123"``);
    2. the all-letters reading, on the same condition (``"5tate"`` -> ``"State"``);
    3. single-character substitutions, left to right.

    A substitution never makes a token *less* homogeneous: ``"plain"`` offers ``"pIain"``
    but not ``"p1ain"``, and ``"12345"`` offers nothing at all.  Introducing a digit into
    an all-letter word is not a reading a recognizer plausibly got wrong, and a
    shortlist full of them is a shortlist nobody reads.

    The original text is never included, duplicates are dropped, and the list is capped
    at ``max_alternatives`` -- the point is a shortlist a human can scan, not the whole
    combinatorial space.
    """
    if not text:
        return []
    out: List[str] = []
    seen = {text}

    digits = "".join(_digit_form(c) for c in text)
    if digits not in seen and digits.isdigit():
        out.append(digits)
        seen.add(digits)
    letters = "".join(_letter_form(c) for c in text)
    if letters not in seen and letters.isalpha():
        out.append(letters)
        seen.add(letters)

    all_letters = text.isalpha()
    all_digits = text.isdigit()
    for i, char in enumerate(text):
        for option in CONFUSION_MAP.get(char, ()):
            if all_letters and option.isdigit():
                continue
            if all_digits and option.isalpha():
                continue
            candidate = text[:i] + option + text[i + 1 :]
            if candidate in seen:
                continue
            out.append(candidate)
            seen.add(candidate)
            if len(out) >= max_alternatives:
                return out
    return out[:max_alternatives]


def is_implausible_mix(text: str) -> bool:
    """True when ``text`` mixes letters and digits the way a misread glyph does.

    The test is deliberately narrow, because plenty of real form values mix the two
    legitimately (``"A1"``, ``"3D"``, ``"F-150"``).  It fires only when the minority
    class is made up **entirely** of confusable characters and is outnumbered by the
    majority class, which is the signature of one stray misread glyph in an otherwise
    homogeneous token.
    """
    core = (text or "").strip()
    if len(core) < 2:
        return False
    letters = [c for c in core if c.isalpha()]
    digits = [c for c in core if c.isdigit()]
    if not letters or not digits:
        return False
    if all(c in CONFUSABLE_CHARS for c in letters) and len(digits) > len(letters):
        return True
    if all(c in CONFUSABLE_CHARS for c in digits) and len(letters) > len(digits):
        return True
    return False


# ======================================================================================
# Finding and fixing
# ======================================================================================


def find_suspects(words: Sequence[RasterWord], config: Any = None) -> List[Suspect]:
    """Flag the words a reviewer should see, in input order.

    Args:
        words: Recognized words -- normally ``result.words + result.suspects``, since a
            confidently-misread word is exactly the one the confidence split misses.
        config: An :class:`~zfp.core.config.OcrConfig`, a
            :class:`~zfp.core.config.ZfpConfig`, or ``None`` for defaults.

    Returns:
        One :class:`Suspect` per flagged word, carrying its index in ``words``.
    """
    cfg = resolve_ocr_config(config)
    out: List[Suspect] = []
    for index, word in enumerate(words):
        reasons: List[str] = []
        if float(word.confidence) < cfg.min_word_confidence:
            reasons.append(REASON_LOW_CONFIDENCE)
        if is_implausible_mix(word.text):
            reasons.append(REASON_MIXED_ALNUM)
        if not reasons:
            continue
        alternatives: List[str] = []
        seen = {word.text}
        for text, _score in word.alternatives:
            if text not in seen:
                alternatives.append(text)
                seen.add(text)
        for candidate in suggest_alternatives(word.text):
            if candidate not in seen:
                alternatives.append(candidate)
                seen.add(candidate)
        out.append(
            Suspect(
                word=word,
                alternatives=alternatives[:MAX_ALTERNATIVES],
                reason=",".join(reasons),
                index=index,
            )
        )
    return out


def apply_correction(
    words: Sequence[RasterWord], index: int, replacement: str
) -> List[RasterWord]:
    """Return a new word list with ``words[index]`` re-read as ``replacement``.

    The previous reading is kept as an alternative, so a correction is reversible and the
    provenance of a filled value survives.  Confidence is left alone on purpose: a
    correction changes what the word says, not how sure the *engine* was.

    Raises:
        ValidationError: ``index`` is out of range, or ``replacement`` is empty.
    """
    items = list(words)
    if index < 0 or index >= len(items):
        raise ValidationError(
            "cannot correct word %d: only %d word(s) available" % (index, len(items))
        )
    text = (replacement or "").strip()
    if not text:
        raise ValidationError("a correction cannot be empty")
    current = items[index]
    if text == current.text:
        return items
    alternatives: List[Tuple[str, float]] = [(current.text, float(current.confidence))]
    for other, score in current.alternatives:
        if other != text and other != current.text:
            alternatives.append((other, float(score)))
    items[index] = replace(current, text=text, alternatives=alternatives)
    _log.debug("corrected OCR word %d: %r -> %r", index, current.text, text)
    return items


# ======================================================================================
# QA
# ======================================================================================


def _bucket_label(low: float, high: float) -> str:
    """Histogram key, e.g. ``"0.7-0.8"``."""
    return "%.1f-%.1f" % (low, high)


def confidence_report(result: "OcrResult") -> Dict[str, Any]:
    """Summarize an :class:`~zfp.ocr.cascade.OcrResult` for the QA dashboard.

    Everything here is JSON-ready and deterministic: counts, the confidence distribution
    in ten fixed buckets, the ten worst words, and the cascade's own decision trail.  A
    page that came out badly should be explainable from this dictionary alone, without
    re-running anything.
    """
    words = list(getattr(result, "words", []) or [])
    suspects = list(getattr(result, "suspects", []) or [])
    every = words + suspects
    confidences = sorted(float(w.confidence) for w in every)

    histogram: Dict[str, int] = {}
    for i in range(len(HISTOGRAM_BUCKETS) - 1):
        histogram[_bucket_label(HISTOGRAM_BUCKETS[i], HISTOGRAM_BUCKETS[i + 1])] = 0
    for value in confidences:
        slot = int(value * 10.0)
        if slot >= len(HISTOGRAM_BUCKETS) - 1:
            slot = len(HISTOGRAM_BUCKETS) - 2
        if slot < 0:
            slot = 0
        histogram[_bucket_label(HISTOGRAM_BUCKETS[slot], HISTOGRAM_BUCKETS[slot + 1])] += 1

    worst = sorted(every, key=lambda w: (float(w.confidence), w.text, w.rect.as_list()))[:10]
    return {
        "engine": getattr(result, "engine", ""),
        "escalated": bool(getattr(result, "escalated", False)),
        "word_count": len(words),
        "suspect_count": len(suspects),
        "total_words": len(every),
        "mean_confidence": float(getattr(result, "mean_confidence", 0.0)),
        "min_confidence": confidences[0] if confidences else 0.0,
        "max_confidence": confidences[-1] if confidences else 0.0,
        "median_confidence": float(median(confidences)) if confidences else 0.0,
        "suspect_ratio": (len(suspects) / float(len(every))) if every else 0.0,
        "histogram": histogram,
        "per_engine": dict(getattr(result, "per_engine", {}) or {}),
        "worst_words": [
            {"text": w.text, "confidence": float(w.confidence), "rect": w.rect.as_list()}
            for w in worst
        ],
        "report": list(getattr(result, "report", []) or []),
    }
