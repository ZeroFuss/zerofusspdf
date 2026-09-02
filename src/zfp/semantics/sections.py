"""Section headings: the parent context a field inherits from the page around it.

A form is not a flat list of fields.  "Address" under *Billing Information* and
"Address" under *Ship To* are different keys, and the only thing on the page that says so
is a heading printed above both.  This module finds those headings from typography alone
-- no model, no dictionary -- and turns each one into a :class:`Section` whose rectangle
covers everything it governs, so a field can be asked "which headings am I under?" and
answer ``["Applicant", "Mailing Address"]``.

A line is scored as a heading on the evidence a typesetter leaves behind:

* it is set larger than the body text;
* its font name says ``Bold`` / ``Black`` / ``Heavy``;
* it is ALL CAPS;
* it carries a numbering prefix -- ``1.``, ``II.``, ``A)``, ``Part 3``, ``Section 4``;
* it stands alone, with more air above it than below it;
* it does *not* end in a colon (a colon means a field is coming, not a section).

The weighted sum crosses :data:`SECTION_SCORE_THRESHOLD` or the line is body text.
Levels come from relative type size: the largest heading on the page is level 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig
from ..core.geometry import EPS, PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import TextSpan
from .graph import DEFAULT_LINE_HEIGHT, detection_config

__all__ = [
    "Section",
    "detect_sections",
    "section_for",
    "heading_score",
    "is_heading_text",
    "SECTION_SCORE_THRESHOLD",
    "BOLD_MARKERS",
    "NUMBERING_RE",
    "MAX_HEADING_CHARS",
]

LOG = get_logger(__name__)

#: Total score a line needs before it is believed to be a heading.
SECTION_SCORE_THRESHOLD = 0.55
#: Font-name fragments that mean "this is set bold".
BOLD_MARKERS: Tuple[str, ...] = ("bold", "black", "heavy", "semibold", "demibold", "demi")
#: A heading longer than this is almost certainly a paragraph.
MAX_HEADING_CHARS = 80
#: ``1.`` ``1.2`` ``II.`` ``A)`` ``(3)`` ``Part 3`` ``Section 4`` ``Step 2`` ``Item 5``.
NUMBERING_RE = re.compile(
    r"^\s*(?:"
    r"(?:part|section|step|item|schedule|article|chapter|block)\s+[0-9ivxlcdm]+[.):]?"
    r"|\(?\d+(?:\.\d+)*\)?[.):]"
    r"|\(?[ivxlcdm]{1,5}\)?[.)]"
    r"|\(?[A-Z]\)?[.)]"
    r")\s+\S",
    re.IGNORECASE,
)
_SUBNUMBER_RE = re.compile(r"^\s*\(?\d+\.\d+")
_WS_RE = re.compile(r"\s+")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_RULE_RE = re.compile(r"[_.]{4,}")

#: Weight of each independent heading signal.
WEIGHTS: Dict[str, float] = {
    "font_size": 0.35,
    "font_size_large": 0.10,
    "bold": 0.28,
    "all_caps": 0.28,
    "numbering": 0.22,
    "isolated": 0.15,
    "no_colon": 0.10,
    "short": 0.05,
}
#: Weight of each disqualifying signal.
PENALTIES: Dict[str, float] = {
    "trailing_colon": 0.20,
    "too_long": 0.35,
    "digit_heavy": 0.20,
    "fill_rule": 0.30,
}
#: A line is "larger than body" from this ratio up.
HEADING_SIZE_RATIO = 1.10
#: ...and "much larger" from this ratio up.
LARGE_SIZE_RATIO = 1.35


@dataclass
class Section:
    """One heading and the region of the page it governs.

    Attributes:
        title: The printed heading text, whitespace-collapsed.
        rect: The governed region: full column width, from the heading's own line down
            to the next heading of the same or a higher level (or the page bottom).
        level: 1 for the largest heading on the page, 2 for the next size down, ...
        page: Zero-based page index.
        score: The heading score that produced this section, for QA and debugging.
        title_rect: The bounding box of the heading line itself.
    """

    title: str
    rect: Rect
    level: int
    page: int
    score: float = 0.0
    title_rect: Optional[Rect] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "title": self.title,
            "rect": self.rect.as_list(),
            "level": self.level,
            "page": self.page,
            "score": self.score,
            "title_rect": self.title_rect.as_list() if self.title_rect else None,
        }


# --------------------------------------------------------------------------- helpers
def _line_text(spans: Sequence[TextSpan]) -> str:
    """Join a line's spans into one printed string."""
    parts = [str(s.text).strip() for s in spans if s is not None and str(s.text).strip()]
    return _WS_RE.sub(" ", " ".join(parts)).strip()


def _size_of(span: TextSpan) -> float:
    """A usable type size for one span: the declared size, else its box height."""
    if span.font_size and span.font_size > EPS:
        return float(span.font_size)
    return float(span.rect.height) if span.rect.height > EPS else DEFAULT_LINE_HEIGHT


def _median(values: Sequence[float]) -> float:
    """Median of a sequence; ``0.0`` when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _group_lines(spans: Sequence[TextSpan]) -> List[List[TextSpan]]:
    """Cluster spans onto text lines, top of the page first.

    Delegates to :func:`zfp.native.text.group_spans_into_lines` when it is importable
    (it is part of the built foundation) and falls back to a baseline clustering with
    the same tolerance otherwise, so this module never hard-depends on it.
    """
    live = [s for s in spans if s is not None and not s.is_blank()]
    if not live:
        return []
    try:  # pragma: no cover - the fallback only runs on a broken install
        from ..native.text import group_spans_into_lines

        return group_spans_into_lines(live)
    except Exception:  # pragma: no cover
        tolerance = 0.5 * (_median([_size_of(s) for s in live]) or DEFAULT_LINE_HEIGHT)
        ordered = sorted(live, key=lambda s: (-s.rect.center.y, s.rect.x0, s.text))
        lines: List[List[TextSpan]] = []
        reference = 0.0
        for span in ordered:
            centre = span.rect.center.y
            if lines and abs(centre - reference) <= tolerance:
                lines[-1].append(span)
                continue
            lines.append([span])
            reference = centre
        for line in lines:
            line.sort(key=lambda s: (s.rect.x0, s.text))
        return lines


def is_heading_text(text: str) -> bool:
    """True when the text alone (ignoring typography) reads like a heading.

    Used by the linker to discount a span that is a section title rather than a field
    label, and by :func:`heading_score` as one of its inputs.
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if stripped.endswith(":"):
        return False
    letters = _ALPHA_RE.findall(stripped)
    if len(letters) < 3:
        return False
    if NUMBERING_RE.match(stripped):
        return True
    return stripped == stripped.upper()


def heading_score(
    text: str,
    size: float,
    body_size: float,
    font_names: Sequence[str] = (),
    gap_above: float = 0.0,
    gap_below: float = 0.0,
    line_height: float = DEFAULT_LINE_HEIGHT,
) -> Tuple[float, List[str]]:
    """Score one text line as a section heading.

    Args:
        text: The whole line, whitespace-collapsed.
        size: The line's largest type size.
        body_size: The page's median type size.
        font_names: The fonts used on the line; ``Bold``/``Black``/``Heavy`` count.
        gap_above: Vertical white space above the line, in points.
        gap_below: Vertical white space below the line, in points.
        line_height: The page's median line height, used to scale "stands alone".

    Returns:
        ``(score, reasons)``; ``score`` is un-clamped so a caller can see how far past
        the threshold a heading landed, and ``reasons`` names every signal that fired.
    """
    stripped = (text or "").strip()
    if not stripped:
        return (0.0, [])
    reasons: List[str] = []
    score = 0.0

    ratio = (size / body_size) if body_size > EPS else 1.0
    if ratio >= HEADING_SIZE_RATIO:
        score += WEIGHTS["font_size"]
        reasons.append("font_size")
        if ratio >= LARGE_SIZE_RATIO:
            score += WEIGHTS["font_size_large"]
            reasons.append("font_size_large")

    lowered = " ".join(str(n or "").lower() for n in font_names)
    if any(marker in lowered for marker in BOLD_MARKERS):
        score += WEIGHTS["bold"]
        reasons.append("bold")

    letters = _ALPHA_RE.findall(stripped)
    if len(letters) >= 3 and stripped == stripped.upper() and stripped != stripped.lower():
        score += WEIGHTS["all_caps"]
        reasons.append("all_caps")

    if NUMBERING_RE.match(stripped):
        score += WEIGHTS["numbering"]
        reasons.append("numbering")

    if gap_above > gap_below * 1.25 and gap_above >= 0.5 * max(line_height, EPS):
        score += WEIGHTS["isolated"]
        reasons.append("isolated")

    if stripped.endswith(":"):
        score -= PENALTIES["trailing_colon"]
        reasons.append("trailing_colon")
    else:
        score += WEIGHTS["no_colon"]
        reasons.append("no_colon")

    if len(stripped) <= 40:
        score += WEIGHTS["short"]
        reasons.append("short")
    if len(stripped) > MAX_HEADING_CHARS:
        score -= PENALTIES["too_long"]
        reasons.append("too_long")

    digits = len(_DIGIT_RE.findall(stripped))
    if letters and digits > len(letters):
        score -= PENALTIES["digit_heavy"]
        reasons.append("digit_heavy")

    if _RULE_RE.search(stripped):
        score -= PENALTIES["fill_rule"]
        reasons.append("fill_rule")

    return (score, reasons)


def _column_bounds(
    spans: Sequence[TextSpan], geometry: Optional[PageGeometry]
) -> Tuple[float, float, float, float]:
    """Return ``(x0, x1, y_top, y_bottom)`` of the region sections may span."""
    if geometry is not None:
        box = geometry.crop_box
        return (box.x0, box.x1, box.y1, box.y0)
    hull = Rect.bounding([s.rect for s in spans]) if spans else None
    if hull is None:
        return (0.0, 0.0, 0.0, 0.0)
    pad = 6.0
    return (hull.x0 - pad, hull.x1 + pad, hull.y1 + pad, hull.y0 - pad)


def _levels(sizes: Sequence[Tuple[float, bool]]) -> Dict[Tuple[float, bool], int]:
    """Map ``(rounded size, is_caps)`` onto a 1-based level, biggest first."""
    unique = sorted(set(sizes), key=lambda item: (-item[0], not item[1]))
    return {item: index + 1 for index, item in enumerate(unique)}


# ------------------------------------------------------------------------ detection
def detect_sections(
    spans: Sequence[TextSpan],
    geometry: Optional[PageGeometry] = None,
    *,
    config: Any = None,
) -> List[Section]:
    """Find the section headings on a page and the region each one governs.

    Args:
        spans: Every text span of the page (several pages are tolerated; they are
            processed independently and the result is concatenated in page order).
        geometry: The page geometry.  When given, section rectangles span the full crop
            box width; otherwise the text hull is used.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`; only used for tolerances.

    Returns:
        Sections ordered by page, then top to bottom.  A section's ``rect`` runs from
        its own line down to the next heading of the same or a higher level.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> from zfp.core.types import TextSpan
        >>> head = TextSpan("APPLICANT", Rect(50, 720, 140, 734), 0, "Helvetica-Bold", 14)
        >>> body = TextSpan("Name:", Rect(50, 690, 90, 700), 0, "Helvetica", 10)
        >>> [s.title for s in detect_sections([head, body])]
        ['APPLICANT']
    """
    det: DetectionConfig = detection_config(config)
    live = [s for s in spans or () if s is not None and not s.is_blank()]
    if not live:
        return []

    by_page: Dict[int, List[TextSpan]] = {}
    for span in live:
        by_page.setdefault(int(span.page), []).append(span)

    out: List[Section] = []
    for page in sorted(by_page):
        out.extend(_detect_page(page, by_page[page], geometry, det))
    return out


def _detect_page(
    page: int,
    spans: Sequence[TextSpan],
    geometry: Optional[PageGeometry],
    det: DetectionConfig,
) -> List[Section]:
    """Detect the sections of a single page."""
    lines = _group_lines(spans)
    if not lines:
        return []

    body_size = _median([_size_of(s) for s in spans]) or DEFAULT_LINE_HEIGHT
    line_height = _median([s.rect.height for s in spans if s.rect.height > EPS]) or body_size

    rects = [Rect.bounding([s.rect for s in line]) or Rect(0, 0, 0, 0) for line in lines]
    texts = [_line_text(line) for line in lines]
    sizes = [max((_size_of(s) for s in line), default=body_size) for line in lines]

    found: List[Tuple[int, Section, float, bool]] = []
    for index, line in enumerate(lines):
        text = texts[index]
        if not text:
            continue
        gap_above = (
            rects[index - 1].y0 - rects[index].y1 if index > 0 else 2.0 * line_height
        )
        gap_below = (
            rects[index].y0 - rects[index + 1].y1
            if index + 1 < len(lines)
            else 2.0 * line_height
        )
        score, reasons = heading_score(
            text,
            sizes[index],
            body_size,
            [s.font_name for s in line],
            max(0.0, gap_above),
            max(0.0, gap_below),
            line_height,
        )
        if score < SECTION_SCORE_THRESHOLD:
            continue
        letters = _ALPHA_RE.findall(text)
        if len(letters) < 3:
            continue
        caps = text == text.upper() and text != text.lower()
        section = Section(
            title=text,
            rect=rects[index],
            level=1,
            page=page,
            score=round(score, 4),
            title_rect=rects[index],
        )
        LOG.debug("section candidate %r on page %d: %.2f %s", text, page, score, reasons)
        found.append((index, section, round(sizes[index] * 4.0) / 4.0, caps))

    if not found:
        return []

    level_of = _levels([(size, caps) for _, _, size, caps in found])
    for _, section, size, caps in found:
        section.level = level_of[(size, caps)]

    x0, x1, _top, bottom = _column_bounds(spans, geometry)
    ordered = sorted(found, key=lambda item: (-item[1].rect.y1, item[1].rect.x0))
    for position, (_index, section, _size, _caps) in enumerate(ordered):
        end = bottom
        for later in ordered[position + 1 :]:
            if later[1].level <= section.level:
                end = later[1].rect.y1
                break
        left = min(x0, section.rect.x0)
        right = max(x1, section.rect.x1)
        top = section.rect.y1
        section.rect = Rect(left, min(end, top), right, top)
    return [section for _index, section, _size, _caps in ordered]


def section_for(
    rect: Rect, sections: Sequence[Section], page: Optional[int] = None
) -> List[Section]:
    """Return the chain of sections enclosing ``rect``, outermost first.

    This is the ``parent_context`` hierarchy: a street-address field printed under
    *Mailing Address*, itself under *Applicant*, answers
    ``["Applicant", "Mailing Address"]``.

    Args:
        rect: The field (or label) rectangle, in PDF user space.
        sections: Candidate sections, normally from :func:`detect_sections`.
        page: When given, only sections on this page are considered.

    Returns:
        Sections ordered by level (outermost first), then by decreasing area.
    """
    if rect is None or not sections:
        return []
    target = rect.normalized()
    hits: List[Section] = []
    for section in sections:
        if section is None or section.rect is None:
            continue
        if page is not None and int(section.page) != int(page):
            continue
        if _covers(section.rect, target):
            hits.append(section)
    hits.sort(key=lambda s: (s.level, -s.rect.area, -s.rect.y1, s.title))
    return hits


def _covers(outer: Rect, inner: Rect) -> bool:
    """True when ``outer`` holds at least 60% of ``inner`` (or its centre)."""
    if inner.area <= EPS:
        return outer.inflated(EPS, EPS).contains_point(inner.center)
    overlap = outer.intersection(inner)
    if overlap is None:
        return False
    return overlap.area >= 0.6 * inner.area
