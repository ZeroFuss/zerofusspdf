"""Shared, precomputed perception context for the archetype detectors.

Eleven detectors look at the same page.  If each one re-derived the horizontal rules,
the boxes and the blank regions from the raw primitives the run would be both slow and
*inconsistent*: two detectors could disagree about where a rule starts.  So everything
shared is derived exactly once, here, by :func:`build_context`, and every detector reads
it off :class:`CandidateContext`.

The heavy geometric work belongs to :mod:`zfp.vision`.  This module calls into it
through :func:`vision_call`, which

* tolerates the module or the function being absent,
* tolerates it raising,
* tolerates the ``config`` argument being a :class:`~zfp.core.config.ZfpConfig` or a
  bare :class:`~zfp.core.config.DetectionConfig` (the contract does not say which), and
* falls back to a conservative pure-python implementation in this module so a missing
  or broken vision backend degrades the run instead of killing it.

All rectangles here are PDF user space: y-up, origin at the page origin, points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig, ZfpConfig
from ..core.geometry import EPS, PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import RasterWord, TextSpan, VectorPrimitive

# The vision layer is imported at module scope but never *required*: a missing or
# half-written backend must degrade, never crash (contract rule 2).
try:  # pragma: no cover - exercised implicitly by whichever branch is installed
    from ..vision import primitives as vision_primitives  # type: ignore
except Exception:  # pragma: no cover
    vision_primitives = None  # type: ignore[assignment]
try:  # pragma: no cover
    from ..vision import blanks as vision_blanks  # type: ignore
except Exception:  # pragma: no cover
    vision_blanks = None  # type: ignore[assignment]

try:  # pragma: no cover - zfp.native.text is part of the built foundation
    from ..native.text import baseline_of as _native_baseline_of
    from ..native.text import group_spans_into_lines as _native_group_lines
    from ..native.text import span_size as _native_span_size
except Exception:  # pragma: no cover
    _native_baseline_of = None  # type: ignore[assignment]
    _native_group_lines = None  # type: ignore[assignment]
    _native_span_size = None  # type: ignore[assignment]

LOG = get_logger(__name__)

__all__ = [
    "CandidateContext",
    "build_context",
    "detect_comb_runs",
    "detection_config",
    "label_columns",
    "label_entry_regions",
    "right_margin",
    "visible_spans",
    "vision_call",
    "baseline_of",
    "span_size",
    "group_lines",
    "median",
    "CHECKBOX_GLYPHS",
    "GRID_CELL_PT",
]

#: Side of one spatial-index bucket, in points.  Roughly one text line by one word.
GRID_CELL_PT: float = 48.0

#: Text that is drawn as a checkbox rather than stroked as a path.
CHECKBOX_GLYPHS: Tuple[str, ...] = (
    "☐",  # BALLOT BOX
    "☑",  # BALLOT BOX WITH CHECK
    "☒",  # BALLOT BOX WITH X
    "□",  # WHITE SQUARE
    "■",  # BLACK SQUARE
    "▢",  # WHITE SQUARE WITH ROUNDED CORNERS
    "◻",  # WHITE MEDIUM SQUARE
    "◼",  # BLACK MEDIUM SQUARE
    "▫",  # WHITE SMALL SQUARE
    "❏",  # LOWER RIGHT DROP-SHADOWED WHITE SQUARE
    "[ ]",
    "[]",
    "[_]",
    "( )",
    "()",
)


# ------------------------------------------------------------------------------ config
def detection_config(config: Any) -> DetectionConfig:
    """Return the :class:`DetectionConfig` inside ``config``.

    Accepts a :class:`ZfpConfig`, a bare :class:`DetectionConfig` or ``None`` so that a
    detector never has to care which one it was handed.
    """
    if config is None:
        return DetectionConfig()
    inner = getattr(config, "detection", None)
    if inner is not None:
        return inner
    if isinstance(config, DetectionConfig):
        return config
    return DetectionConfig()


def _zfp_config(config: Any) -> ZfpConfig:
    """Return a full :class:`ZfpConfig`, promoting a bare detection config if needed."""
    if isinstance(config, ZfpConfig):
        return config
    full = ZfpConfig.default()
    if isinstance(config, DetectionConfig):
        full.detection = config
    return full


# ------------------------------------------------------------------------ small maths
def median(values: Sequence[float]) -> float:
    """Median of ``values``; ``0.0`` for an empty sequence."""
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def baseline_of(span: TextSpan) -> float:
    """Baseline of ``span``, falling back to the vertical centre of its box."""
    if _native_baseline_of is not None:
        try:
            return float(_native_baseline_of(span))
        except Exception:  # pragma: no cover - defensive
            pass
    if span.baseline is not None:
        return float(span.baseline)
    return (span.rect.y0 + span.rect.y1) / 2.0


def span_size(span: TextSpan) -> float:
    """Usable font size for ``span``: its declared size, else its box height."""
    if _native_span_size is not None:
        try:
            return float(_native_span_size(span))
        except Exception:  # pragma: no cover - defensive
            pass
    if span.font_size and span.font_size > 0.0:
        return float(span.font_size)
    return float(span.rect.height)


def group_lines(spans: Sequence[TextSpan]) -> List[List[TextSpan]]:
    """Group ``spans`` onto text lines, top of the page first."""
    if _native_group_lines is not None:
        try:
            return [list(line) for line in _native_group_lines(list(spans))]
        except Exception:  # pragma: no cover - defensive
            pass
    return _fallback_group_lines(spans)


def _fallback_group_lines(spans: Sequence[TextSpan]) -> List[List[TextSpan]]:
    """Baseline clustering used when :mod:`zfp.native.text` is unavailable."""
    items = [s for s in spans if s is not None]
    if not items:
        return []
    tol = max(1.0, 0.5 * median([span_size(s) for s in items]))
    ordered = sorted(items, key=lambda s: (-baseline_of(s), s.rect.x0, s.text))
    lines: List[List[TextSpan]] = []
    current: List[TextSpan] = []
    anchor = 0.0
    for span in ordered:
        base = baseline_of(span)
        if not current or abs(base - anchor) <= tol:
            if not current:
                anchor = base
            current.append(span)
        else:
            lines.append(sorted(current, key=lambda s: (s.rect.x0, s.text)))
            current = [span]
            anchor = base
    if current:
        lines.append(sorted(current, key=lambda s: (s.rect.x0, s.text)))
    return lines


# ------------------------------------------------------------------------ vision glue
def vision_call(
    name: str,
    *args: Any,
    config: Any = None,
    fallback: Optional[Callable[..., Any]] = None,
    listify: bool = True,
) -> Any:
    """Call ``zfp.vision.<name>(*args, config)`` defensively, never raising.

    Args:
        name: Function name to look up in :mod:`zfp.vision.primitives` then
            :mod:`zfp.vision.blanks`.
        args: Positional arguments *before* the trailing config argument.
        config: The config argument.  Both the ``ZfpConfig`` and the
            ``DetectionConfig`` spelling are attempted, because the contract does not
            pin which one the vision layer expects.
        fallback: Pure-python implementation with the same ``(*args, config)``
            signature, used when the vision function is missing or raises.
        listify: Coerce the result to a ``list``.  Set ``False`` for helpers that
            return a mapping (``rule_spans``), whose keys would otherwise be all that
            survived.

    Returns:
        The result, or the fallback's result, or an empty list.
    """
    fn = _vision_function(name)
    if fn is not None:
        for cfg in _config_variants(config):
            try:
                result = fn(*args, cfg)
            except TypeError as exc:
                LOG.debug("zfp.vision.%s rejected config %s: %s", name, type(cfg).__name__, exc)
                continue
            except Exception as exc:
                LOG.warning("zfp.vision.%s failed: %s", name, exc)
                break
            if not listify:
                return result if result is not None else []
            return list(result or [])
    if fallback is not None:
        try:
            produced = fallback(*args, detection_config(config))
            if not listify:
                return produced if produced is not None else []
            return list(produced or [])
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("candidate fallback for %s failed: %s", name, exc)
    return []


def _vision_function(name: str) -> Optional[Callable[..., Any]]:
    """Return the named vision helper from whichever vision module carries it."""
    for module in (vision_primitives, vision_blanks):
        if module is None:
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _config_variants(config: Any) -> List[Any]:
    """Config arguments to try, most likely first, without duplicates."""
    out: List[Any] = []
    if config is not None:
        out.append(config)
    det = detection_config(config)
    if det is not config:
        out.append(det)
    return out


# --------------------------------------------------------------- pure-python fallbacks
def _fb_normalize_primitives(
    prims: Sequence[VectorPrimitive], det: DetectionConfig
) -> List[VectorPrimitive]:
    """Normalize rectangles and drop primitives with no extent at all."""
    out: List[VectorPrimitive] = []
    for p in prims:
        rect = p.rect.normalized()
        if rect.width <= EPS and rect.height <= EPS:
            continue
        out.append(
            VectorPrimitive(
                kind=p.kind,
                rect=rect,
                page=p.page,
                stroke_width=p.stroke_width,
                filled=p.filled,
                stroked=p.stroked,
                points=list(p.points),
            )
        )
    return out


def _is_rule_shaped(p: VectorPrimitive, det: DetectionConfig, horizontal: bool) -> bool:
    """True when ``p`` is thin enough and long enough to act as a ruling line."""
    if p.kind not in ("line", "rect", "path"):
        return False
    rect = p.rect
    thickness = rect.height if horizontal else rect.width
    length = rect.width if horizontal else rect.height
    if thickness > det.max_line_thickness_pt:
        return False
    if length < max(2.0, det.max_line_thickness_pt):
        return False
    return length >= max(thickness, EPS) * 3.0


def _fb_horizontal_rules(
    prims: Sequence[VectorPrimitive], det: DetectionConfig
) -> List[VectorPrimitive]:
    """Thin, wide primitives, in top-to-bottom then left-to-right order."""
    out = [p for p in prims if _is_rule_shaped(p, det, True)]
    out.sort(key=lambda p: (-p.rect.y1, p.rect.x0, p.rect.x1))
    return out


def _fb_vertical_rules(
    prims: Sequence[VectorPrimitive], det: DetectionConfig
) -> List[VectorPrimitive]:
    """Thin, tall primitives, in left-to-right then top-to-bottom order."""
    out = [p for p in prims if _is_rule_shaped(p, det, False)]
    out.sort(key=lambda p: (p.rect.x0, -p.rect.y1))
    return out


def _fb_merge_collinear(
    rules: Sequence[VectorPrimitive], det: DetectionConfig
) -> List[VectorPrimitive]:
    """Weld rules that share an axis and are separated by at most the merge tolerance."""
    horizontal = [r for r in rules if r.rect.width >= r.rect.height]
    vertical = [r for r in rules if r.rect.width < r.rect.height]
    out = _merge_axis(horizontal, det.line_merge_tolerance_pt, True)
    out.extend(_merge_axis(vertical, det.line_merge_tolerance_pt, False))
    out.sort(key=lambda p: (-p.rect.y1, p.rect.x0, p.rect.x1, p.rect.y0))
    return out


def _merge_axis(
    rules: Sequence[VectorPrimitive], tol: float, horizontal: bool
) -> List[VectorPrimitive]:
    """Merge one orientation of rules: cluster across the axis, then along it."""
    if not rules:
        return []
    tol = max(float(tol), EPS)

    def across(p: VectorPrimitive) -> float:
        return (p.rect.y0 + p.rect.y1) / 2.0 if horizontal else (p.rect.x0 + p.rect.x1) / 2.0

    def start(p: VectorPrimitive) -> float:
        return p.rect.x0 if horizontal else p.rect.y0

    def end(p: VectorPrimitive) -> float:
        return p.rect.x1 if horizontal else p.rect.y1

    ordered = sorted(rules, key=lambda p: (across(p), start(p), end(p)))
    bands: List[List[VectorPrimitive]] = []
    for rule in ordered:
        if bands and abs(across(rule) - across(bands[-1][-1])) <= tol:
            bands[-1].append(rule)
        else:
            bands.append([rule])

    merged: List[VectorPrimitive] = []
    for band in bands:
        band.sort(key=lambda p: (start(p), end(p)))
        run: List[VectorPrimitive] = [band[0]]
        reach = end(band[0])
        for rule in band[1:]:
            if start(rule) <= reach + tol:
                run.append(rule)
                reach = max(reach, end(rule))
            else:
                merged.append(_weld(run))
                run = [rule]
                reach = end(rule)
        merged.append(_weld(run))
    return merged


def _weld(run: Sequence[VectorPrimitive]) -> VectorPrimitive:
    """Collapse a run of collinear rules into a single primitive."""
    first = run[0]
    if len(run) == 1:
        return first
    rect = Rect.bounding([r.rect for r in run]) or first.rect
    return VectorPrimitive(
        kind="line",
        rect=rect,
        page=first.page,
        stroke_width=max(r.stroke_width for r in run),
        filled=any(r.filled for r in run),
        stroked=any(r.stroked for r in run),
        points=[],
    )


def _fb_detect_boxes(prims: Sequence[VectorPrimitive], det: DetectionConfig) -> List[Rect]:
    """Stroked/filled rectangles, plus rectangles closed by four separate rules."""
    seen: Dict[Tuple[float, float, float, float], Rect] = {}

    def remember(rect: Rect) -> None:
        r = rect.normalized()
        if r.width < 3.0 or r.height < 3.0:
            return
        key = tuple(round(v, 2) for v in r.as_list())  # type: ignore[assignment]
        seen.setdefault(key, r)  # type: ignore[arg-type]

    for p in prims:
        if p.kind != "rect":
            continue
        if p.rect.width < 3.0 or p.rect.height < 3.0:
            continue
        remember(p.rect)

    h_rules = _fb_merge_collinear(_fb_horizontal_rules(prims, det), det)
    v_rules = [r for r in _fb_merge_collinear(_fb_vertical_rules(prims, det), det)]
    tol = max(det.line_merge_tolerance_pt, 2.0)
    for i, top in enumerate(h_rules):
        for bottom in h_rules[i + 1 :]:
            if bottom.rect.y1 >= top.rect.y0 - EPS:
                continue
            x0 = max(top.rect.x0, bottom.rect.x0)
            x1 = min(top.rect.x1, bottom.rect.x1)
            if x1 - x0 < 3.0:
                continue
            if abs(top.rect.x0 - bottom.rect.x0) > tol or abs(top.rect.x1 - bottom.rect.x1) > tol:
                continue
            y0, y1 = bottom.rect.y0, top.rect.y1
            if y1 - y0 < 3.0:
                continue
            left = any(
                abs(v.rect.x0 - x0) <= tol and v.rect.y0 <= y0 + tol and v.rect.y1 >= y1 - tol
                for v in v_rules
            )
            right = any(
                abs(v.rect.x1 - x1) <= tol and v.rect.y0 <= y0 + tol and v.rect.y1 >= y1 - tol
                for v in v_rules
            )
            if left and right:
                remember(Rect(x0, y0, x1, y1))
    out = list(seen.values())
    out.sort(key=lambda r: (-r.y1, r.x0, r.x1, r.y0))
    return out


def _fb_detect_circles(prims: Sequence[VectorPrimitive], det: DetectionConfig) -> List[Rect]:
    """Bounding boxes of circle-classified subpaths."""
    out = [p.rect.normalized() for p in prims if p.kind == "circle" and p.rect.width > 0.0]
    out.sort(key=lambda r: (-r.y1, r.x0))
    return out


def _is_checkbox_sized(rect: Rect, det: DetectionConfig) -> bool:
    """True when ``rect`` is a small, roughly square box."""
    w, h = rect.width, rect.height
    if w < det.checkbox_min_pt or h < det.checkbox_min_pt:
        return False
    if w > det.checkbox_max_pt or h > det.checkbox_max_pt:
        return False
    longest = max(w, h)
    if longest <= 0.0:
        return False
    return abs(w - h) <= det.checkbox_aspect_tolerance * longest


def _fb_detect_checkbox_glyphs(
    prims: Sequence[VectorPrimitive], spans: Sequence[TextSpan], det: DetectionConfig
) -> List[Rect]:
    """Small square boxes plus text characters that are drawn as a checkbox."""
    out: List[Rect] = []
    for p in prims:
        if p.kind not in ("rect", "circle"):
            continue
        rect = p.rect.normalized()
        if _is_checkbox_sized(rect, det):
            out.append(rect)
    for span in spans:
        text = (span.text or "").strip()
        if text and text in CHECKBOX_GLYPHS:
            out.append(span.rect.normalized())
    out.sort(key=lambda r: (-r.y1, r.x0))
    return out


def _fb_detect_table_cells(
    h_rules: Sequence[VectorPrimitive],
    v_rules: Sequence[VectorPrimitive],
    det: DetectionConfig,
) -> List[Rect]:
    """Cells of a ruled grid: every four-sided opening between adjacent rules."""
    tol = max(det.line_merge_tolerance_pt, 2.0)
    rows = sorted(h_rules, key=lambda p: -((p.rect.y0 + p.rect.y1) / 2.0))
    cols = sorted(v_rules, key=lambda p: (p.rect.x0 + p.rect.x1) / 2.0)
    if len(rows) < 2 or len(cols) < 2:
        return []
    out: List[Rect] = []
    for i in range(len(rows) - 1):
        top, bottom = rows[i], rows[i + 1]
        y1 = (top.rect.y0 + top.rect.y1) / 2.0
        y0 = (bottom.rect.y0 + bottom.rect.y1) / 2.0
        if y1 - y0 < 4.0:
            continue
        # Only the rules that actually cross this band are this band's columns: two
        # tables on one page have different column positions and must not interleave.
        band = [c for c in cols if _spans_interval(c, y0, y1, tol, False)]
        for j in range(len(band) - 1):
            left, right = band[j], band[j + 1]
            x0 = (left.rect.x0 + left.rect.x1) / 2.0
            x1 = (right.rect.x0 + right.rect.x1) / 2.0
            if x1 - x0 < 4.0:
                continue
            if not _spans_interval(top, x0, x1, tol, True):
                continue
            if not _spans_interval(bottom, x0, x1, tol, True):
                continue
            out.append(Rect(x0, y0, x1, y1))
    out.sort(key=lambda r: (-r.y1, r.x0))
    return out


def _spans_interval(
    rule: VectorPrimitive, lo: float, hi: float, tol: float, horizontal: bool
) -> bool:
    """True when ``rule`` covers ``[lo, hi]`` along its own axis."""
    r = rule.rect
    start, end = (r.x0, r.x1) if horizontal else (r.y0, r.y1)
    return start <= lo + tol and end >= hi - tol


def _fb_blank_regions(
    spans: Sequence[TextSpan],
    prims: Sequence[VectorPrimitive],
    geometry: PageGeometry,
    det: DetectionConfig,
) -> List[Rect]:
    """Whitespace big enough to hold an entry, right of and below the printed text.

    Two families are produced:

    * **in-line gaps** -- horizontal runs of nothing inside a text line's band, which is
      the ``Label:            `` case, and
    * **inter-line gaps** -- vertical space between two consecutive text lines, sliced
      into columns by the labels of the line above and capped to one line of writing
      space, which is the borderless ``Label`` / blank-underneath case.
    """
    visible = [s for s in spans if not s.is_blank() and s.confidence > 0.0]
    if not visible:
        return []
    page = geometry.crop_box.normalized()
    ink: List[Rect] = [s.rect.normalized() for s in visible]
    ink.extend(p.rect.normalized() for p in prims)
    extent = Rect.bounding(ink) or page
    left = max(page.x0, extent.x0)
    right = min(page.x1, extent.x1)
    if right - left < det.blank_min_width_pt:
        return []

    body = max(median([span_size(s) for s in visible]), 6.0)
    entry_h = max(det.blank_min_height_pt + 5.0, 1.5 * body)
    lines = group_lines(visible)
    out: List[Rect] = []

    for line in lines:
        band = Rect.bounding([s.rect for s in line])
        if band is None:
            continue
        probe = Rect(left, band.y0 - 4.0, right, band.y1 + 1.0)
        blocked: List[Tuple[float, float]] = []
        for rect in ink:
            if rect.y1 < probe.y0 or rect.y0 > probe.y1:
                continue
            blocked.append((rect.x0 - 1.0, rect.x1 + 1.0))
        for gap0, gap1 in _free_intervals(blocked, left, right):
            if gap1 - gap0 < det.blank_min_width_pt:
                continue
            height = max(band.height, det.blank_min_height_pt)
            out.append(Rect(gap0, band.y0, gap1, band.y0 + height))

    for index in range(len(lines) - 1):
        upper = Rect.bounding([s.rect for s in lines[index]])
        lower = Rect.bounding([s.rect for s in lines[index + 1]])
        if upper is None or lower is None:
            continue
        top = upper.y0 - 2.0
        bottom = lower.y1
        if top - bottom < det.blank_min_height_pt:
            continue
        height = min(top - bottom, entry_h)
        for col0, col1 in _label_columns(lines[index], left, right):
            if col1 - col0 < det.blank_min_width_pt:
                continue
            region = Rect(col0, top - height, col1, top)
            if _covered_by_ink(region, ink):
                continue
            out.append(region)

    deduped: Dict[Tuple[float, float, float, float], Rect] = {}
    for rect in out:
        clipped = geometry.clamp(rect)
        if clipped.width < det.blank_min_width_pt or clipped.height < det.blank_min_height_pt:
            continue
        key = tuple(round(v, 2) for v in clipped.as_list())  # type: ignore[assignment]
        deduped.setdefault(key, clipped)  # type: ignore[arg-type]
    result = list(deduped.values())
    result.sort(key=lambda r: (-r.y1, r.x0, r.x1))
    return result


def _covered_by_ink(region: Rect, ink: Sequence[Rect]) -> bool:
    """True when printed content already occupies a fifth of ``region``."""
    if region.area <= 0.0:
        return False
    covered = 0.0
    for rect in ink:
        inter = region.intersection(rect)
        if inter is not None:
            covered += inter.area
            if covered / region.area > 0.2:
                return True
    return False


def _free_intervals(
    blocked: Sequence[Tuple[float, float]], lo: float, hi: float
) -> List[Tuple[float, float]]:
    """Complement of ``blocked`` inside ``[lo, hi]``."""
    merged: List[List[float]] = []
    for start, end in sorted(blocked):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: List[Tuple[float, float]] = []
    cursor = lo
    for start, end in merged:
        if start > cursor:
            out.append((cursor, min(start, hi)))
        cursor = max(cursor, end)
        if cursor >= hi:
            break
    if cursor < hi:
        out.append((cursor, hi))
    return [(a, b) for a, b in out if b > a]


def _label_columns(
    line: Sequence[TextSpan], left: float, right: float
) -> List[Tuple[float, float]]:
    """Column extents implied by the labels on one line, left to right."""
    return [(a, b) for a, b, _ in label_columns(line, left, right, 10.0)]


def label_columns(
    line: Sequence[TextSpan], left: float, right: float, body: float
) -> List[Tuple[float, float, TextSpan]]:
    """Split one text line into ``(x0, x1, first span)`` columns.

    A new column starts wherever a run of spans is separated from the previous one by
    more than two ems of white space -- the gutter of a two- or three-up form row.  Each
    column ends a small gutter short of the next one so two adjacent fields never touch.
    """
    ordered = sorted(line, key=lambda s: (s.rect.x0, s.rect.x1))
    if not ordered:
        return []
    gutter = max(4.0, 0.5 * body)
    starts: List[TextSpan] = [ordered[0]]
    reach = ordered[0].rect.x1
    for span in ordered[1:]:
        if span.rect.x0 - reach > 2.0 * body:
            starts.append(span)
        reach = max(reach, span.rect.x1)
    out: List[Tuple[float, float, TextSpan]] = []
    for index, span in enumerate(starts):
        x0 = max(left, span.rect.x0)
        if index + 1 < len(starts):
            x1 = starts[index + 1].rect.x0 - gutter
        else:
            x1 = right
        if x1 > x0:
            out.append((x0, min(x1, right), span))
    return out


def right_margin(page: Rect, extent: Optional[Rect]) -> float:
    """Where the printed column ends, assuming the page's margins are symmetric.

    A borderless form's rightmost field often has no ink in it at all, so the text
    extent stops well short of the real column edge.  Mirroring the left margin
    recovers it; the actual extent wins whenever it reaches further.
    """
    if extent is None or extent.width <= 0.0:
        return page.x1
    mirrored = page.x1 - max(extent.x0 - page.x0, 0.0)
    return max(extent.x1, min(page.x1, mirrored))


def _is_heading_span(span: TextSpan, body: float) -> bool:
    """True when a span is set as a heading -- bold, or noticeably larger than the body."""
    if "bold" in (span.font_name or "").lower():
        return True
    return body > 0.0 and span_size(span) >= 1.10 * body


def _captions_rule_above(
    span: TextSpan, prims: Sequence[VectorPrimitive], body: float
) -> bool:
    """True when ``span`` is the caption printed under a rule.

    The signature-block idiom puts the rule first and the word under it.  Such a label
    already has its field -- above -- and must not also claim the space below.
    """
    ceiling = span.rect.y1 + 1.2 * max(body, EPS)
    for prim in prims:
        rect = prim.rect
        if rect.height > 3.0:
            continue
        if rect.y0 < span.rect.y1 - 0.5 or rect.y0 > ceiling:
            continue
        if rect.horizontal_overlap(span.rect) >= 0.5 * max(span.rect.width, EPS):
            return True
    return False


def _touches_ink(rect: Rect, ink: Sequence[Rect]) -> bool:
    """True when any printed mark reaches into ``rect``.

    Rules are inflated vertically first: a stroked line has zero height, so a plain
    intersection test would miss the very thing that proves a row is not blank.
    """
    probe = rect.normalized()
    for mark in ink:
        if probe.intersects(mark.inflated(0.0, 0.5)):
            return True
    return False


def label_entry_regions(
    spans: Sequence[TextSpan],
    prims: Sequence[VectorPrimitive],
    geometry: PageGeometry,
    det: DetectionConfig,
) -> List[Rect]:
    """Entry areas printed *under* their labels -- the borderless form idiom.

    Whitespace analysis on its own reports maximal empty rectangles, which on a sparse
    page run all the way to the bottom margin.  This narrows that down to the shape a
    person would actually write in: one line of writing space directly beneath a label,
    as wide as the label's column, and only where the label does not already have an
    entry area beside it.
    """
    visible = visible_spans(spans)
    if not visible:
        return []
    page = geometry.crop_box.normalized()
    ink: List[Rect] = [s.rect.normalized() for s in visible]
    ink.extend(p.rect.normalized() for p in prims)
    extent = Rect.bounding(ink)
    left = max(page.x0, extent.x0) if extent is not None else page.x0
    right = right_margin(page, extent)
    body = max(median([span_size(s) for s in visible]), 6.0)
    entry_h = max(det.blank_min_height_pt + 5.0, 1.5 * body)
    out: List[Rect] = []
    for line in group_lines(visible):
        band = Rect.bounding([s.rect for s in line])
        if band is None:
            continue
        for x0, x1, anchor in label_columns(line, left, right, body):
            if x1 - x0 < det.blank_min_width_pt:
                continue
            if _is_heading_span(anchor, body):
                continue  # a section heading does not own the space beneath it
            if _captions_rule_above(anchor, prims, body):
                continue  # the label belongs to the rule over it, not to the space under it
            top = anchor.rect.y0 - 2.0
            # A rule or a box beside the label means the entry area is already there.
            gap_start = anchor.rect.x1 + 1.0
            if x1 > gap_start:
                beside = Rect(gap_start, band.y0 - 3.0, x1, band.y1)
                if _touches_ink(beside, ink):
                    continue
            floor = page.y0
            probe = Rect(x0, page.y0, x1, top)
            for mark in ink:
                if mark.y1 >= top - EPS or mark.horizontal_overlap(probe) <= 0.0:
                    continue
                floor = max(floor, mark.y1)
            available = top - floor
            if available < det.blank_min_height_pt:
                continue
            region = Rect(x0, top - min(available, entry_h), x1, top)
            if _touches_ink(region, ink):
                continue
            out.append(region)
    out.sort(key=lambda r: (-r.y1, r.x0))
    return out


# ------------------------------------------------------------------------- the context
@dataclass
class CandidateContext:
    """Everything the eleven archetype detectors share about one page.

    The first six fields are the contract's; every other field is *derived* and is
    normally produced by :func:`build_context`.  Passing derived fields explicitly is
    supported (and is what the unit tests do) so a detector can be exercised against
    hand-built geometry without going anywhere near :mod:`zfp.vision`.
    """

    page: int
    geometry: PageGeometry
    spans: List[TextSpan] = field(default_factory=list)
    primitives: List[VectorPrimitive] = field(default_factory=list)
    words: List[RasterWord] = field(default_factory=list)
    config: ZfpConfig = field(default_factory=ZfpConfig.default)

    #: Widget rectangles that already exist on the page; candidates never overlap them.
    existing_widgets: Tuple[Rect, ...] = ()

    # -- derived ------------------------------------------------------------------
    all_spans: List[TextSpan] = field(default_factory=list)
    text_spans: List[TextSpan] = field(default_factory=list)
    lines: List[List[TextSpan]] = field(default_factory=list)
    h_rules: List[VectorPrimitive] = field(default_factory=list)
    v_rules: List[VectorPrimitive] = field(default_factory=list)
    boxes: List[Rect] = field(default_factory=list)
    circles: List[Rect] = field(default_factory=list)
    checkbox_glyphs: List[Rect] = field(default_factory=list)
    table_cells: List[Rect] = field(default_factory=list)
    comb_runs: List[List[Rect]] = field(default_factory=list)
    blank_regions: List[Rect] = field(default_factory=list)
    #: The subset of :attr:`blank_regions` anchored directly under a printed label.
    label_blanks: List[Rect] = field(default_factory=list)
    median_font_size: float = 0.0
    median_stroke_width: float = 0.0
    content_extent: Optional[Rect] = None
    #: The horizontal band the page prints in, margins mirrored (see :func:`right_margin`).
    text_column: Optional[Rect] = None

    #: Scratch space shared by detectors within one run (memoized derivations).
    cache: Dict[str, Any] = field(default_factory=dict)

    _grid: Dict[Tuple[int, int], List[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.config = _zfp_config(self.config)
        self.spans = list(self.spans)
        self.primitives = list(self.primitives)
        self.words = list(self.words)
        self.existing_widgets = tuple(r.normalized() for r in self.existing_widgets)
        if not self.all_spans:
            self.all_spans = list(self.spans)
        if not self.text_spans:
            self.text_spans = visible_spans(self.all_spans)
        if not self.lines:
            self.lines = group_lines(self.text_spans)
        if self.median_font_size <= 0.0:
            self.median_font_size = median([span_size(s) for s in self.text_spans])
        if self.median_stroke_width <= 0.0:
            widths = [p.stroke_width for p in self.primitives if p.stroke_width > 0.0]
            self.median_stroke_width = median(widths)
        if self.content_extent is None:
            self.content_extent = Rect.bounding([s.rect for s in self.text_spans])
        if self.text_column is None:
            self.text_column = self._derive_text_column()
        self._build_index()

    def _derive_text_column(self) -> Rect:
        """The horizontal band the page prints in.

        Taken from every mark on the page, text and vector alike, with the right margin
        mirrored from the left when nothing is printed that far over -- a borderless
        form's rightmost column is empty by definition, and would otherwise look like it
        was outside the page's own text area.
        """
        page = self.page_rect
        ink = [s.rect for s in self.text_spans] + [p.rect for p in self.primitives]
        extent = Rect.bounding(ink)
        if extent is None:
            return page
        return Rect(max(page.x0, extent.x0), page.y0, right_margin(page, extent), page.y1)

    # -- convenience --------------------------------------------------------------
    @property
    def detection(self) -> DetectionConfig:
        """The detection thresholds in force for this page."""
        return detection_config(self.config)

    @property
    def page_rect(self) -> Rect:
        """The page's crop box, normalized."""
        return self.geometry.crop_box.normalized()

    @property
    def page_width(self) -> float:
        """Crop-box width in points."""
        return self.page_rect.width

    @property
    def page_height(self) -> float:
        """Crop-box height in points."""
        return self.page_rect.height

    @property
    def body_font_size(self) -> float:
        """Median font size, never below a sane 6 pt floor."""
        return max(self.median_font_size, 6.0)

    def clamp(self, rect: Rect) -> Rect:
        """Normalize ``rect`` and clip it to the page's crop box."""
        return self.geometry.clamp(rect.normalized())

    # -- spatial index ------------------------------------------------------------
    def _build_index(self) -> None:
        """Bucket the visible spans into a fixed-pitch grid for neighbour lookups."""
        self._grid = {}
        for index, span in enumerate(self.text_spans):
            for key in _cells_for(span.rect):
                self._grid.setdefault(key, []).append(index)

    def spans_near(self, rect: Rect, pad: float = 0.0) -> List[TextSpan]:
        """Return the visible spans intersecting ``rect`` inflated by ``pad``.

        Args:
            rect: Query rectangle in user space.
            pad: Extra margin added on all four sides.

        Returns:
            Spans in their original (reading) order, each returned at most once.
        """
        probe = rect.normalized().inflated(float(pad))
        found: List[int] = []
        seen = set()
        for key in _cells_for(probe):
            for index in self._grid.get(key, ()):
                if index in seen:
                    continue
                seen.add(index)
                if self.text_spans[index].rect.intersects(probe):
                    found.append(index)
        found.sort()
        return [self.text_spans[i] for i in found]

    def local_font_size(self, rect: Rect, pad: float = 36.0) -> float:
        """Median font size of the text around ``rect``, falling back to the page median."""
        nearby = self.spans_near(rect, pad)
        local = median([span_size(s) for s in nearby if span_size(s) > 0.0])
        if local > 0.0:
            return local
        return self.body_font_size

    def stroke_width_for(self, rect: Rect, default: Optional[float] = None) -> float:
        """Stroke width of the primitive that drew ``rect``.

        Matching is by geometry: the primitive whose rectangle agrees with ``rect`` on
        all four edges to within 1 pt.  Falls back to the page's median stroke width,
        then to ``default``, then to ``1.0``.
        """
        target = rect.normalized()
        best: Optional[float] = None
        best_error = 1.0 + EPS
        for prim in self.primitives:
            if prim.stroke_width <= 0.0:
                continue
            r = prim.rect.normalized()
            error = max(
                abs(r.x0 - target.x0),
                abs(r.y0 - target.y0),
                abs(r.x1 - target.x1),
                abs(r.y1 - target.y1),
            )
            if error < best_error:
                best_error = error
                best = prim.stroke_width
        if best is not None:
            return float(best)
        if self.median_stroke_width > 0.0:
            return float(self.median_stroke_width)
        return float(default if default is not None else 1.0)

    def blocked_by_widget(self, rect: Rect) -> bool:
        """True when ``rect`` overlaps a widget that already exists on the page."""
        probe = rect.normalized()
        for widget in self.existing_widgets:
            if not probe.intersects(widget):
                continue
            if probe.iou(widget) > 0.05:
                return True
            inter = probe.intersection(widget)
            if inter is not None and probe.area > 0.0 and inter.area / probe.area > 0.25:
                return True
        return False


def _cells_for(rect: Rect) -> List[Tuple[int, int]]:
    """Grid buckets covered by ``rect``."""
    r = rect.normalized()
    x0 = int(r.x0 // GRID_CELL_PT)
    x1 = int(r.x1 // GRID_CELL_PT)
    y0 = int(r.y0 // GRID_CELL_PT)
    y1 = int(r.y1 // GRID_CELL_PT)
    return [(cx, cy) for cx in range(x0, x1 + 1) for cy in range(y0, y1 + 1)]


def visible_spans(spans: Sequence[TextSpan]) -> List[TextSpan]:
    """Filter out spans that carry no visible glyphs.

    A span is dropped when it holds nothing but whitespace, or when its confidence is
    ``0.0``.  Zero confidence is how the parser marks text render mode 3 (invisible) and
    how OCR marks a rejected word: an invisible OCR layer sitting behind a scan must
    never be mistaken for a printed label.
    """
    return [s for s in spans if s is not None and not s.is_blank() and s.confidence > 0.0]


# ------------------------------------------------------------------------- the factory
def build_context(
    page_index: int,
    geometry: PageGeometry,
    spans: Sequence[TextSpan],
    primitives: Sequence[VectorPrimitive],
    words: Sequence[RasterWord] = (),
    config: Optional[ZfpConfig] = None,
    existing_widgets: Sequence[Rect] = (),
) -> CandidateContext:
    """Derive every shared structure the detectors need, exactly once.

    Args:
        page_index: Zero-based page index.
        geometry: The page's :class:`PageGeometry`.
        spans: Native (or already converted OCR) text spans in user space.
        primitives: Vector primitives in user space.
        words: OCR words in user space.  When ``spans`` is empty these are promoted to
            spans so a scanned page runs through exactly the same detectors.
        config: Engine configuration; defaults to :meth:`ZfpConfig.default`.
        existing_widgets: Rectangles of widgets already present on the page.

    Returns:
        A fully populated :class:`CandidateContext`.
    """
    cfg = _zfp_config(config)
    det = cfg.detection
    all_spans = list(spans)
    if not all_spans and words:
        all_spans = [_span_from_word(w) for w in words]

    prims = vision_call(
        "normalize_primitives", list(primitives), config=cfg, fallback=_fb_normalize_primitives
    )
    if not prims:
        prims = list(primitives)

    raw_h = vision_call("horizontal_rules", prims, config=cfg, fallback=_fb_horizontal_rules)
    raw_v = vision_call("vertical_rules", prims, config=cfg, fallback=_fb_vertical_rules)
    h_rules = vision_call("merge_collinear", raw_h, config=cfg, fallback=_fb_merge_collinear)
    v_rules = vision_call("merge_collinear", raw_v, config=cfg, fallback=_fb_merge_collinear)
    if not h_rules:
        h_rules = list(raw_h)
    if not v_rules:
        v_rules = list(raw_v)

    boxes = vision_call("detect_boxes", prims, config=cfg, fallback=_fb_detect_boxes)
    circles = vision_call("detect_circles", prims, config=cfg, fallback=_fb_detect_circles)
    glyphs = vision_call(
        "detect_checkbox_glyphs",
        prims,
        visible_spans(all_spans),
        config=cfg,
        fallback=_fb_detect_checkbox_glyphs,
    )
    cells = vision_call(
        "detect_table_cells", h_rules, v_rules, config=cfg, fallback=_fb_detect_table_cells
    )
    blanks = vision_call(
        "blank_regions",
        visible_spans(all_spans),
        prims,
        geometry,
        config=cfg,
        fallback=_fb_blank_regions,
    )
    # Maximal empty rectangles say where the page is free; they do not say what shape a
    # field would be.  The label-anchored regions add exactly that, for the borderless
    # archetype where nothing but the label is printed.
    anchored = label_entry_regions(visible_spans(all_spans), prims, geometry, det)

    ctx = CandidateContext(
        page=int(page_index),
        geometry=geometry,
        spans=all_spans,
        primitives=list(prims),
        words=list(words),
        config=cfg,
        existing_widgets=tuple(Rect.from_list(r.as_list()) for r in existing_widgets),
        all_spans=list(all_spans),
        h_rules=[r for r in h_rules if isinstance(r, VectorPrimitive)],
        v_rules=[r for r in v_rules if isinstance(r, VectorPrimitive)],
        boxes=_as_rects(boxes),
        circles=_as_rects(circles),
        checkbox_glyphs=_as_rects(glyphs),
        table_cells=_as_rects(cells),
        blank_regions=_dedupe_rects(_as_rects(anchored) + _as_rects(blanks)),
        label_blanks=_as_rects(anchored),
    )
    ctx.comb_runs = detect_comb_runs(ctx)
    LOG.debug(
        "page %d context: %d spans, %d h-rules, %d v-rules, %d boxes, %d cells, %d blanks",
        ctx.page,
        len(ctx.text_spans),
        len(ctx.h_rules),
        len(ctx.v_rules),
        len(ctx.boxes),
        len(ctx.table_cells),
        len(ctx.blank_regions),
    )
    return ctx


def _span_from_word(word: RasterWord) -> TextSpan:
    """Promote an OCR word to a text span so scanned pages use the same detectors."""
    return TextSpan(
        text=word.text,
        rect=word.rect.normalized(),
        page=word.page,
        source="ocr",
        confidence=word.confidence,
    )


def _dedupe_rects(rects: Sequence[Rect]) -> List[Rect]:
    """Drop exact duplicates, keeping the first occurrence."""
    seen: Dict[Tuple[float, ...], Rect] = {}
    for rect in rects:
        seen.setdefault(tuple(round(v, 2) for v in rect.as_list()), rect)
    return list(seen.values())


def _as_rects(values: Sequence[Any]) -> List[Rect]:
    """Coerce a vision result into a clean list of normalized rectangles."""
    out: List[Rect] = []
    for value in values:
        if isinstance(value, Rect):
            out.append(value.normalized())
        elif isinstance(value, VectorPrimitive):
            out.append(value.rect.normalized())
        elif isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                out.append(Rect.from_list([float(v) for v in value[:4]]))
            except Exception:  # pragma: no cover - defensive
                continue
    return out


# ------------------------------------------------------------------------- comb runs
def detect_comb_runs(ctx: CandidateContext) -> List[List[Rect]]:
    """Return every run of >= 3 equal, evenly spaced cells on this page.

    Two idioms are covered, because both appear in real forms and in the synthetic
    corpus: a row of separate little boxes (``[ ][ ][ ]``) and one outer box divided by
    internal vertical rules (``[ | | ]``).  ``zfp.vision.detect_comb_cells`` is used for
    the first when it exists.
    """
    det = ctx.detection
    runs: List[List[Rect]] = []

    supplied = vision_call("detect_comb_cells", list(ctx.boxes), config=ctx.config)
    for run in _normalize_runs(supplied, det):
        runs.append(run)
    if not runs:
        runs.extend(_runs_from_boxes(ctx.boxes, det))
    runs.extend(_runs_from_separators(ctx, det))

    # A comb cell holds one character.  Three equal boxes across a row are three
    # fields, not a nine-character comb, so anything much wider than the body text is
    # rejected here rather than stealing the row from the box detector.
    widest = max(3.5 * ctx.body_font_size, 30.0)
    unique: Dict[Tuple[Any, ...], List[Rect]] = {}
    for run in runs:
        if len(run) < 3:
            continue
        cell_width = median([c.width for c in run])
        cell_height = median([c.height for c in run])
        if cell_width > widest or cell_width > 2.5 * max(cell_height, EPS):
            continue
        # Comb cells touch.  Evenly spaced marks that do *not* touch are a row of
        # checkboxes, and stealing them here would cost the checkbox detector the row.
        gaps = [run[i + 1].x0 - run[i].x1 for i in range(len(run) - 1)]
        if median(gaps) > max(0.5 * cell_width, det.comb_cell_tolerance_pt):
            continue
        key = tuple(tuple(round(v, 2) for v in cell.as_list()) for cell in run)
        unique.setdefault(key, run)
    out = list(unique.values())
    out.sort(key=lambda run: (-run[0].y1, run[0].x0))
    return out


def _normalize_runs(values: Sequence[Any], det: DetectionConfig) -> List[List[Rect]]:
    """Interpret a ``detect_comb_cells`` result as runs, whatever shape it arrived in."""
    if not values:
        return []
    first = values[0]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], Rect):
        return [[c.normalized() for c in run] for run in values if len(run) >= 3]
    cells = _as_rects(values)
    if not cells:
        return []
    return _runs_from_boxes(cells, det)


def _runs_from_boxes(boxes: Sequence[Rect], det: DetectionConfig) -> List[List[Rect]]:
    """Group equal-width boxes sharing a row into evenly spaced runs of >= 3 cells."""
    tol = max(det.comb_cell_tolerance_pt, 0.5)
    rows: List[List[Rect]] = []
    for rect in sorted(boxes, key=lambda r: (-r.y1, r.x0)):
        placed = False
        for row in rows:
            head = row[0]
            if abs(head.y0 - rect.y0) <= tol and abs(head.y1 - rect.y1) <= tol:
                row.append(rect)
                placed = True
                break
        if not placed:
            rows.append([rect])

    runs: List[List[Rect]] = []
    for row in rows:
        row.sort(key=lambda r: r.x0)
        run: List[Rect] = [row[0]]
        for cell in row[1:]:
            previous = run[-1]
            same_width = abs(cell.width - previous.width) <= tol
            gap = cell.x0 - previous.x1
            pitch_ok = -tol <= gap <= max(tol, 0.5 * previous.width)
            even = True
            if len(run) >= 2:
                expected = run[1].x0 - run[0].x0
                even = abs((cell.x0 - previous.x0) - expected) <= tol
            if same_width and pitch_ok and even:
                run.append(cell)
                continue
            if len(run) >= 3:
                runs.append(run)
            run = [cell]
        if len(run) >= 3:
            runs.append(run)
    return runs


def _runs_from_separators(ctx: CandidateContext, det: DetectionConfig) -> List[List[Rect]]:
    """Split each outer box that internal vertical rules divide into equal cells."""
    tol = max(det.comb_cell_tolerance_pt, 0.5)
    runs: List[List[Rect]] = []
    for box in ctx.boxes:
        if box.width < 3 * det.checkbox_min_pt or box.height <= 0.0:
            continue
        if _crossed_horizontally(ctx, box, tol):
            continue  # rules cross it both ways: that is a table, not a comb
        cuts: List[float] = []
        for rule in ctx.v_rules:
            r = rule.rect
            x = (r.x0 + r.x1) / 2.0
            if x <= box.x0 + tol or x >= box.x1 - tol:
                continue
            if r.y0 > box.y0 + tol or r.y1 < box.y1 - tol:
                continue
            cuts.append(x)
        if len(cuts) < 2:
            continue
        cuts.sort()
        deduped = [cuts[0]]
        for x in cuts[1:]:
            if x - deduped[-1] > tol:
                deduped.append(x)
        if len(deduped) < 2:
            continue
        edges = [box.x0] + deduped + [box.x1]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        if min(widths) <= 0.0:
            continue
        if max(widths) - min(widths) > tol:
            continue
        runs.append([Rect(edges[i], box.y0, edges[i + 1], box.y1) for i in range(len(widths))])
    return runs


def _crossed_horizontally(ctx: CandidateContext, box: Rect, tol: float) -> bool:
    """True when a horizontal rule runs across the inside of ``box``."""
    for rule in ctx.h_rules:
        r = rule.rect
        y = (r.y0 + r.y1) / 2.0
        if y <= box.y0 + tol or y >= box.y1 - tol:
            continue
        if r.x0 > box.x0 + tol or r.x1 < box.x1 - tol:
            continue
        return True
    return False
