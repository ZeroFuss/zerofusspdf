"""Read AcroForm fields back from a document: values, the field tree, and export formats.

Complements :mod:`zfp.acroform.writer` -- both a document ZFP wrote and one it did not
(an arbitrary interactive PDF) go through the same reader, which is what makes
:meth:`AcroFormWriter.set_values` and :meth:`~AcroFormWriter.flatten` work on either.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..core.types import FieldSpec, FieldType
from ..pdfio.document import Document
from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfString

_BTN_RADIO = 1 << 15
_BTN_PUSH = 1 << 16
_CH_COMBO = 1 << 17


def read_fields(doc: Document) -> List[FieldSpec]:
    """Every terminal field in the document, as :class:`FieldSpec` values."""
    return doc.existing_fields()


def read_values(doc: Document) -> Dict[str, str]:
    """``{field name: current value}`` for every field that has one."""
    out: Dict[str, str] = {}
    for spec in doc.existing_fields():
        if spec.value is not None:
            out[spec.name] = spec.value
    return out


def _field_type_from_dict(d: PdfDict, resolve) -> FieldType:
    ft = d.get_name("FT") if hasattr(d, "get_name") else None
    ff = int(resolve(d.get("Ff")) or 0) if d.get("Ff") is not None else 0
    if ft == "Btn":
        if ff & _BTN_RADIO:
            return FieldType.RADIO
        if ff & _BTN_PUSH:
            return FieldType.BUTTON
        return FieldType.CHECKBOX
    if ft == "Ch":
        return FieldType.CHOICE if ff & _CH_COMBO else FieldType.LISTBOX
    if ft == "Sig":
        return FieldType.SIGNATURE
    if ft == "Tx":
        return FieldType.COMB if ff & (1 << 24) else (
            FieldType.MULTILINE_TEXT if ff & (1 << 12) else FieldType.TEXT)
    return FieldType.UNKNOWN


def field_tree(doc: Document) -> Dict[str, Dict[str, Any]]:
    """A name-indexed view of the field tree with enough structure to drive
    :meth:`AcroFormWriter.set_values` and :meth:`~AcroFormWriter.flatten`.

    Each entry carries: ``dict`` (the field/top object), ``num`` (its object number),
    ``field_type``, ``widgets`` (``[(page_index, rect, ap_N_ref)]``), ``widget_refs``,
    and, for grouped buttons, ``kids``/``kid_states``.
    """
    acroform = doc.acroform()
    if not isinstance(acroform, PdfDict):
        return {}
    fields = doc.resolve(acroform.get("Fields"))
    if not isinstance(fields, (PdfArray, list)):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    page_lookup = _page_lookup(doc)

    def walk(ref: Any, name_parts: List[str], parent_ft: Optional[str]) -> None:
        node = doc.resolve(ref)
        if not isinstance(node, PdfDict):
            return
        t = node.get("T")
        seg = _text(t) if t is not None else None
        parts = name_parts + ([seg] if seg else [])
        kids_raw = doc.resolve(node.get("Kids")) if node.get("Kids") is not None else None
        ft = node.get_name("FT") if hasattr(node, "get_name") else None
        ft = ft or parent_ft

        is_terminal_with_kids_as_widgets = False
        if isinstance(kids_raw, (PdfArray, list)) and kids_raw:
            first_kid = doc.resolve(kids_raw[0])
            if isinstance(first_kid, PdfDict) and "T" not in first_kid:
                is_terminal_with_kids_as_widgets = True

        if isinstance(kids_raw, (PdfArray, list)) and kids_raw and not is_terminal_with_kids_as_widgets:
            for kid_ref in kids_raw:
                walk(kid_ref, parts, ft)
            return

        if not seg and not parts:
            return
        full_name = ".".join(parts) if parts else (seg or "")
        if not full_name:
            return

        field_type = _field_type_from_dict(node, doc.resolve) if ft else FieldType.UNKNOWN
        widgets: List[Any] = []
        widget_refs: List[PdfRef] = []
        kid_entries: List[Any] = []
        kid_states: Dict[int, Set[str]] = {}

        widget_sources = kids_raw if is_terminal_with_kids_as_widgets else [ref]
        for w_ref in widget_sources:
            w = doc.resolve(w_ref)
            if not isinstance(w, PdfDict):
                continue
            rect_vals = doc.resolve(w.get("Rect"))
            page_idx = page_lookup.get(w_ref.num if isinstance(w_ref, PdfRef) else -1, 0)
            ap = doc.resolve(w.get("AP")) if w.get("AP") is not None else None
            ap_n = doc.resolve(ap.get("N")) if isinstance(ap, PdfDict) else None
            ap_ref = ap.get("N") if isinstance(ap, PdfDict) else None
            from ..core.geometry import Rect as _Rect
            rect = _Rect.from_list([float(v) for v in rect_vals]) if rect_vals else None
            widgets.append((page_idx, rect, ap_ref if isinstance(ap_ref, PdfRef) else None))
            if isinstance(w_ref, PdfRef):
                widget_refs.append(w_ref)
            if isinstance(ap_n, PdfDict) and isinstance(w_ref, PdfRef):
                kid_states[w_ref.num] = set(ap_n.keys())
                kid_entries.append((w_ref.num, w))

        num = ref.num if isinstance(ref, PdfRef) else None
        out[full_name] = {
            "dict": node,
            "num": num,
            "field_type": field_type,
            "widgets": widgets,
            "widget_refs": widget_refs,
            "kids": kid_entries,
            "kid_states": kid_states,
            "_top_ref": ref if isinstance(ref, PdfRef) else None,
        }

    for top_ref in fields:
        walk(top_ref, [], None)
    return out


def _page_lookup(doc: Document) -> Dict[int, int]:
    """``{annotation object number: page index}`` by scanning every page's /Annots."""
    out: Dict[int, int] = {}
    for page in doc.pages:
        for ref in page.annotation_refs():
            if isinstance(ref, PdfRef):
                out[ref.num] = page.index
    return out


def _text(value: Any) -> str:
    if isinstance(value, PdfString):
        return value.text()
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("latin-1", "replace")
    return str(value) if value is not None else ""


def export_json(doc: Document) -> Dict[str, Any]:
    """``{field name: value}`` for every field carrying a value."""
    return dict(read_values(doc))


def import_json(doc: Document, data: Dict[str, Any], schema: Optional[Any] = None) -> Dict[str, str]:
    """Resolve a ``{name-or-canonical-key: value}`` mapping to ``{field name: value}``.

    When ``schema`` (a :class:`~zfp.core.types.FormSchema`) is given, a key matching a
    field's ``canonical_key`` is accepted as well as its literal field name.
    """
    resolved: Dict[str, str] = {}
    by_key: Dict[str, str] = {}
    if schema is not None:
        for spec in schema.fields:
            if spec.canonical_key:
                by_key[spec.canonical_key] = spec.name
    for k, v in data.items():
        name = by_key.get(k, k)
        resolved[name] = "" if v is None else str(v)
    return resolved


def export_xml(doc: Document) -> str:
    import xml.sax.saxutils as sax
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<fields>"]
    for name, value in read_values(doc).items():
        parts.append('  <field name="%s">%s</field>' % (
            sax.escape(name), sax.escape(value)))
    parts.append("</fields>")
    return "\n".join(parts)


def export_csv(doc: Document) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "value"])
    for name, value in read_values(doc).items():
        writer.writerow([name, value])
    return buf.getvalue()


def export_fdf(doc: Document) -> bytes:
    """A minimal, real FDF (Forms Data Format) document."""
    lines = [b"%FDF-1.2", b"1 0 obj", b"<< /FDF << /Fields ["]
    for name, value in read_values(doc).items():
        name_bytes = _fdf_string(name)
        value_bytes = _fdf_string(value)
        lines.append(b"<< /T " + name_bytes + b" /V " + value_bytes + b" >>")
    lines.append(b"] >> >>")
    lines.append(b"endobj")
    lines.append(b"trailer")
    lines.append(b"<< /Root 1 0 R >>")
    lines.append(b"%%EOF")
    return b"\n".join(lines) + b"\n"


def _fdf_string(s: str) -> bytes:
    encoded = s.encode("utf-16-be")
    body = (b"\xfe\xff" + encoded).replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + body + b")"


def roundtrip_check(doc: Document, schema: Any) -> List[str]:
    """Discrepancies between ``schema`` and what actually reads back from ``doc``."""
    issues: List[str] = []
    existing = {f.name: f for f in doc.existing_fields()}
    for spec in schema.fields:
        # Radio group members are read back under the group name, not the member name.
        key = spec.group if (spec.field_type == FieldType.RADIO and spec.group) else spec.name
        found = existing.get(key)
        if found is None:
            issues.append("field %r missing after write" % key)
            continue
        if found.field_type != spec.field_type and spec.field_type != FieldType.RADIO:
            issues.append("field %r type mismatch: wrote %s, read %s" %
                          (key, spec.field_type, found.field_type))
    return issues


__all__ = [
    "read_fields", "read_values", "field_tree", "export_json", "import_json",
    "export_xml", "export_csv", "export_fdf", "roundtrip_check",
]
