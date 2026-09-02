"""Structural verification of a written PDF: integrity, prefix preservation, and the
absolute invariants the overlay approach depends on.

Every check returns findings rather than raising; :func:`verify_document` runs them all
and reports a pass/fail verdict plus the finding list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class QAFinding:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    page: Optional[int] = None
    field: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "message": self.message,
                "page": self.page, "field": self.field}


@dataclass
class QAReport:
    findings: List[QAFinding] = field(default_factory=list)
    passed: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add(self, finding: QAFinding) -> None:
        self.findings.append(finding)
        if finding.severity == "error":
            self.passed = False

    def extend(self, findings: List[QAFinding]) -> None:
        for f in findings:
            self.add(f)

    def errors(self) -> List[QAFinding]:
        return [f for f in self.findings if f.severity == "error"]

    def warnings(self) -> List[QAFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "findings": [f.as_dict() for f in self.findings],
                "metrics": dict(self.metrics)}


def check_integrity(data: bytes) -> List[QAFinding]:
    """Reparse ``data`` and walk its page tree; report anything that fails."""
    from ..pdfio.parser import PdfFile

    findings: List[QAFinding] = []
    try:
        pf = PdfFile.load(data)
    except Exception as exc:  # noqa: BLE001
        return [QAFinding("error", "PDF_PARSE_FAILED", "could not parse output: %s" % exc)]

    if getattr(pf, "rebuilt", False):
        findings.append(QAFinding("error", "XREF_REBUILD_REQUIRED",
                                  "ZFP's own output required xref reconstruction"))

    try:
        pages = pf.page_dicts()
    except Exception as exc:  # noqa: BLE001
        return findings + [QAFinding("error", "PAGE_TREE_BROKEN",
                                     "page tree could not be walked: %s" % exc)]

    for i, page in enumerate(pages):
        try:
            contents = page.get("Contents")
            resolved = pf.resolve(contents) if contents is not None else None
            if resolved is not None and hasattr(resolved, "decoded"):
                resolved.decoded(pf)
        except Exception as exc:  # noqa: BLE001
            findings.append(QAFinding("error", "CONTENT_DECODE_FAILED",
                                      "page %d content stream failed to decode: %s" % (i, exc),
                                      page=i))

    try:
        _ = pf.catalog
    except Exception as exc:  # noqa: BLE001
        findings.append(QAFinding("error", "CATALOG_UNRESOLVED",
                                  "document catalog did not resolve: %s" % exc))

    return findings


def check_prefix_preserved(original: bytes, produced: bytes) -> List[QAFinding]:
    """The invariant the whole overlay approach rests on."""
    if produced.startswith(original):
        return [QAFinding("info", "PREFIX_PRESERVED", "original bytes are a literal prefix")]
    limit = min(len(original), len(produced))
    offset = next((i for i in range(limit) if original[i] != produced[i]), limit)
    return [QAFinding("error", "PREFIX_NOT_PRESERVED",
                      "output diverges from the original at byte offset %d" % offset)]


def check_no_overlap(schema: Any, threshold: float = 0.10) -> List[QAFinding]:
    findings: List[QAFinding] = []
    fields = list(getattr(schema, "fields", []))
    by_page: Dict[int, List[Any]] = {}
    for spec in fields:
        for page, rect in spec.widgets():
            by_page.setdefault(page, []).append((spec.name, rect.normalized()))
    for page, entries in by_page.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_a, rect_a = entries[i]
                name_b, rect_b = entries[j]
                overlap = rect_a.iou(rect_b)
                if overlap > threshold:
                    findings.append(QAFinding(
                        "error", "FIELD_OVERLAP",
                        "%r and %r overlap by IoU %.3f on page %d" % (name_a, name_b, overlap, page),
                        page=page))
    return findings


def check_in_page_bounds(doc: Any, schema: Any) -> List[QAFinding]:
    findings: List[QAFinding] = []
    for spec in getattr(schema, "fields", []):
        for page_idx, rect in spec.widgets():
            try:
                page = doc.page(page_idx)
                bounds = page.geometry.crop_box
            except Exception:  # noqa: BLE001
                continue
            r = rect.normalized()
            if not bounds.contains_rect(r):
                findings.append(QAFinding(
                    "error", "FIELD_OUT_OF_BOUNDS",
                    "field %r rect %s is outside page %d bounds %s" %
                    (spec.name, r.as_list(), page_idx, bounds.as_list()),
                    page=page_idx, field=spec.name))
    return findings


def check_fields_roundtrip(produced: bytes, schema: Any) -> List[QAFinding]:
    from ..pdfio.document import Document

    findings: List[QAFinding] = []
    try:
        doc = Document.open(produced)
        existing = {f.name: f for f in doc.existing_fields()}
    except Exception as exc:  # noqa: BLE001
        return [QAFinding("error", "REOPEN_FAILED", "could not reopen output: %s" % exc)]

    for spec in getattr(schema, "fields", []):
        key = spec.group if (spec.field_type.value == "radio" and spec.group) else spec.name
        found = existing.get(key)
        if found is None:
            findings.append(QAFinding("error", "FIELD_MISSING",
                                      "field %r not found after reopening" % key, field=key))
            continue
        if spec.field_type.value != "radio" and found.field_type != spec.field_type:
            findings.append(QAFinding("error", "FIELD_TYPE_MISMATCH",
                                      "field %r: wrote %s, read %s" %
                                      (key, spec.field_type, found.field_type), field=key))
        if spec.value is not None and found.value != spec.value and spec.field_type.value != "radio":
            findings.append(QAFinding("warning", "FIELD_VALUE_MISMATCH",
                                      "field %r: wrote %r, read %r" %
                                      (key, spec.value, found.value), field=key))
        if spec.max_length and found.max_length and found.max_length != spec.max_length:
            findings.append(QAFinding("warning", "FIELD_MAXLEN_MISMATCH",
                                      "field %r max_length differs" % key, field=key))
    return findings


def check_field_names_unique(schema: Any) -> List[QAFinding]:
    seen: Dict[str, int] = {}
    for spec in getattr(schema, "fields", []):
        seen[spec.name] = seen.get(spec.name, 0) + 1
    return [QAFinding("warning", "DUPLICATE_FIELD_NAME", "name %r used %d times" % (n, c))
           for n, c in seen.items() if c > 1]


def check_values_validate(schema: Any, fill_report: Any) -> List[QAFinding]:
    from ..resolver.validators import validate_against_constraints

    findings: List[QAFinding] = []
    by_name = {s.name: s for s in getattr(schema, "fields", [])}
    for filled in getattr(fill_report, "values", []):
        if filled.status != "filled" or filled.value is None:
            continue
        spec = by_name.get(filled.field_name)
        if spec is None:
            continue
        outcome = validate_against_constraints(filled.value, spec)
        if not outcome.ok:
            findings.append(QAFinding("error", "VALUE_CONSTRAINT_VIOLATION",
                                      "%r: %s" % (filled.field_name, outcome.message),
                                      field=filled.field_name))
    return findings


def check_security_preserved(original: bytes, produced: bytes) -> List[QAFinding]:
    from ..pdfio.parser import PdfFile

    findings: List[QAFinding] = []
    try:
        orig_pf = PdfFile.load(original)
        prod_pf = PdfFile.load(produced)
    except Exception:  # noqa: BLE001
        return findings
    if orig_pf.is_encrypted and not prod_pf.is_encrypted:
        findings.append(QAFinding("error", "ENCRYPTION_LOST",
                                  "input was encrypted but output is not"))
    return findings


def verify_document(original: bytes, produced: bytes, schema: Any, config: Any = None) -> QAReport:
    report = QAReport()
    report.extend(check_prefix_preserved(original, produced))
    report.extend(check_integrity(produced))
    report.extend(check_no_overlap(schema))
    report.extend(check_fields_roundtrip(produced, schema))
    report.extend(check_field_names_unique(schema))
    report.extend(check_security_preserved(original, produced))

    report.metrics["field_count"] = len(getattr(schema, "fields", []))
    report.metrics["error_count"] = len(report.errors())
    report.metrics["warning_count"] = len(report.warnings())
    return report


__all__ = [
    "QAFinding", "QAReport", "check_integrity", "check_prefix_preserved",
    "check_no_overlap", "check_in_page_bounds", "check_fields_roundtrip",
    "check_field_names_unique", "check_values_validate", "check_security_preserved",
    "verify_document",
]
