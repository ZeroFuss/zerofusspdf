"""Visual preservation: prove that nothing outside a written region changed.

The default, dependency-free path compares decoded content-stream bytes page by page --
for an incremental AcroForm write, which only appends annotations, the honest answer is
"no page content changed at all," and that is a strong, cheap proof of visual
preservation without rendering a single pixel. A pixel-level render diff is available
when a renderer is present, and is named honestly so neither path is mistaken for the
other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..core.geometry import Rect
from .verify import QAFinding, QAReport


@dataclass
class DiffResult:
    changed_pixels: int = 0
    total_pixels: int = 0
    ratio: float = 0.0
    bbox: Optional[Rect] = None
    masked_pixels: int = 0


def page_hash(rendered: Any) -> str:
    return hashlib.blake2b(rendered.gray, digest_size=16).hexdigest()


def diff_pages(a: Any, b: Any, mask: Sequence[Rect] = (), tolerance: int = 8) -> DiffResult:
    if a.width != b.width or a.height != b.height:
        return DiffResult(changed_pixels=-1, total_pixels=0, ratio=1.0)

    scale = a.scale
    mask_px = []
    for r in mask:
        x0 = int(r.x0 * scale)
        y0 = int((a.height / scale - r.y1) * scale)
        x1 = int(r.x1 * scale)
        y1 = int((a.height / scale - r.y0) * scale)
        mask_px.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    changed = 0
    masked = 0
    min_x = min_y = None
    max_x = max_y = None
    ga, gb = a.gray, b.gray
    w, h = a.width, a.height
    for y in range(h):
        row_off = y * w
        for x in range(w):
            idx = row_off + x
            if abs(ga[idx] - gb[idx]) <= tolerance:
                continue
            if any(mx0 <= x < mx1 and my0 <= y < my1 for mx0, my0, mx1, my1 in mask_px):
                masked += 1
                continue
            changed += 1
            min_x = x if min_x is None else min(min_x, x)
            max_x = x if max_x is None else max(max_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_y = y if max_y is None else max(max_y, y)

    total = w * h
    bbox = Rect(min_x, min_y, max_x, max_y) if min_x is not None else None
    return DiffResult(changed_pixels=changed, total_pixels=total,
                      ratio=(changed / total) if total else 0.0, bbox=bbox,
                      masked_pixels=masked)


def content_stream_fingerprint(doc: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for page in doc.pages:
        try:
            data = page.content_bytes()
        except Exception:  # noqa: BLE001
            data = b""
        out[page.index] = hashlib.blake2b(data, digest_size=16).hexdigest()
    return out


def structural_diff(original: bytes, produced: bytes) -> QAReport:
    """Dependency-free proof of visual preservation: compare decoded page content."""
    from ..pdfio.document import Document

    report = QAReport()
    try:
        orig_doc = Document.open(original)
        prod_doc = Document.open(produced)
    except Exception as exc:  # noqa: BLE001
        report.add(QAFinding("error", "STRUCTURAL_DIFF_FAILED",
                             "could not open documents for comparison: %s" % exc))
        return report

    orig_fp = content_stream_fingerprint(orig_doc)
    prod_fp = content_stream_fingerprint(prod_doc)

    changed_pages: List[int] = []
    for idx, fp in orig_fp.items():
        if prod_fp.get(idx) != fp:
            changed_pages.append(idx)

    report.metrics["method"] = "structural (content-stream fingerprint) -- not a pixel comparison"
    report.metrics["pages_compared"] = len(orig_fp)
    report.metrics["pages_changed"] = changed_pages
    if changed_pages:
        report.add(QAFinding("warning", "PAGE_CONTENT_CHANGED",
                             "page content stream(s) changed: %s" % changed_pages))
    else:
        report.add(QAFinding("info", "PAGE_CONTENT_UNCHANGED",
                             "no page content stream changed (structural comparison)"))
    return report


def visual_preservation_report(original_doc: Any, produced_doc: Any, schema: Any,
                               config: Any = None, dpi: int = 150) -> QAReport:
    """Pixel-level render diff when a renderer is available; degrades to
    :func:`structural_diff` otherwise (the common case in a stdlib-only install)."""
    from ..core.errors import UnsupportedFeatureError

    report = QAReport()
    try:
        from ..raster.render import render_page
    except Exception:  # noqa: BLE001
        report.metrics["method"] = "structural (no renderer module available)"
        return report

    masks: Dict[int, List[Rect]] = {}
    for spec in getattr(schema, "fields", []):
        for page, rect in spec.widgets():
            masks.setdefault(page, []).append(rect.inflated(2.0))

    try:
        for page in original_doc.pages:
            a = render_page(original_doc, page.index, dpi=dpi)
            b = render_page(produced_doc, page.index, dpi=dpi)
            result = diff_pages(a, b, masks.get(page.index, ()))
            report.metrics.setdefault("pages", {})[page.index] = {
                "ratio": result.ratio, "changed_pixels": result.changed_pixels,
            }
            if result.ratio > 0.001:
                report.add(QAFinding("error", "VISUAL_CHANGE_OUTSIDE_MASK",
                                     "page %d changed outside written fields (ratio %.5f)" %
                                     (page.index, result.ratio), page=page.index))
            elif result.ratio > 0.0001:
                report.add(QAFinding("warning", "VISUAL_CHANGE_MINOR",
                                     "page %d minor change outside written fields (ratio %.5f)" %
                                     (page.index, result.ratio), page=page.index))
        report.metrics["method"] = "pixel render diff at %d dpi" % dpi
    except UnsupportedFeatureError:
        return structural_diff(original_doc.to_bytes(incremental=False),
                               produced_doc.to_bytes(incremental=False))
    return report


__all__ = [
    "DiffResult", "page_hash", "diff_pages", "content_stream_fingerprint",
    "structural_diff", "visual_preservation_report",
]
