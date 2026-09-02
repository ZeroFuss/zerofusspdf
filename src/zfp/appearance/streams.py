"""Appearance-stream content generation for AcroForm widgets.

Every function here returns the *content bytes* of a form XObject -- the operators that
go between the appearance stream's implicit save/restore, in the XObject's own
coordinate space (``(0, 0)`` to ``(rect.width, rect.height)``, established by its
``/BBox``). :func:`build_xobject` wraps those bytes into the actual indirect object.

Getting the coordinate space right matters more than anything else in this module: an
appearance stream that draws in page coordinates instead of XObject-local coordinates
renders off-screen in every conformant viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ..core.geometry import Rect
from ..core.types import FieldSpec, FieldType
from ..pdfio import fonts
from ..pdfio.filters import encode_flate
from ..pdfio.fonts import escape_pdf_text
from . import layout as _layout

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from ..pdfio.document import Document
    from ..pdfio.objects import PdfDict, PdfRef

_Color = Tuple[float, float, float]


def _fmt(n: float) -> str:
    """Compact, deterministic number formatting for content-stream operands."""
    r = round(float(n), 3)
    if r == int(r):
        return str(int(r))
    return ("%.3f" % r).rstrip("0").rstrip(".")


def _ops(*parts: object) -> bytes:
    return (" ".join(str(p) for p in parts) + "\n").encode("latin-1")


def _color_op(color: Optional[_Color], fill: bool) -> bytes:
    if color is None:
        color = (0.0, 0.0, 0.0)
    r, g, b = color
    if abs(r - g) < 1e-6 and abs(g - b) < 1e-6:
        return _ops(_fmt(r), "g" if fill else "G")
    return _ops(_fmt(r), _fmt(g), _fmt(b), "rg" if fill else "RG")


def border_and_background_ops(spec: FieldSpec, rect: Rect) -> bytes:
    """Border stroke and background fill for a widget's own BBox, if configured."""
    rect = rect.normalized()
    out = bytearray()
    if spec.background_color is not None:
        out += _ops("q")
        out += _color_op(spec.background_color, fill=True)
        out += _ops(_fmt(0), _fmt(0), _fmt(rect.width), _fmt(rect.height), "re", "f")
        out += _ops("Q")
    if spec.border_color is not None and spec.border_width > 0:
        w = spec.border_width
        inset = w / 2.0
        out += _ops("q")
        out += _ops(_fmt(w), "w")
        out += _color_op(spec.border_color, fill=False)
        out += _ops(_fmt(inset), _fmt(inset), _fmt(max(0.0, rect.width - w)),
                    _fmt(max(0.0, rect.height - w)), "re", "S")
        out += _ops("Q")
    return bytes(out)


def _path_ops(path: List[Tuple[str, Tuple[float, ...]]]) -> bytes:
    out = bytearray()
    for op, operands in path:
        if operands:
            out += _ops(*[_fmt(v) for v in operands], op)
        else:
            out += _ops(op)
    return bytes(out)


def _font_resource_name(spec: FieldSpec) -> str:
    canonical = fonts.resolve_base_font(spec.font_name)
    return fonts.resource_name(canonical)


def _text_block(layout_result: "_layout.TextLayout", spec: FieldSpec) -> bytes:
    if not layout_result.lines:
        return b""
    res = _font_resource_name(spec)
    out = bytearray()
    out += _ops("BT")
    out += _ops("/" + res, _fmt(layout_result.font_size), "Tf")
    out += _color_op(spec.text_color, fill=True)
    x0, y0 = layout_result.origins[0]
    out += _ops(_fmt(x0), _fmt(y0), "Td")
    out += (b"(" + escape_pdf_text(layout_result.lines[0]) + b") Tj\n")
    prev_x, prev_y = x0, y0
    for line, (x, y) in zip(layout_result.lines[1:], layout_result.origins[1:]):
        dx, dy = x - prev_x, y - prev_y
        out += _ops(_fmt(dx), _fmt(dy), "Td")
        out += (b"(" + escape_pdf_text(line) + b") Tj\n")
        prev_x, prev_y = x, y
    out += _ops("ET")
    return bytes(out)


def _clip_ops(rect: Rect, padding: float) -> bytes:
    rect = rect.normalized()
    x0 = max(0.0, padding * 0.4)
    y0 = x0
    w = max(0.0, rect.width - 2 * x0)
    h = max(0.0, rect.height - 2 * y0)
    return _ops(_fmt(x0), _fmt(y0), _fmt(w), _fmt(h), "re", "W", "n")


def text_appearance(spec: FieldSpec, value: Optional[str], resources: "PdfDict") -> bytes:
    """Content bytes for a text, date, number, currency, email or phone field.

    Wrapped in a ``/Tx`` marked-content section, which is the PDF convention for a form
    field's generated appearance and lets a viewer distinguish it from page content.
    Multiline values are laid out with :func:`zfp.appearance.layout.layout_text`, one
    ``Td``/``Tj`` pair per line; a value too wide or tall for the rectangle is clipped
    with ``W n`` rather than allowed to bleed into neighbouring fields.
    """
    rect = spec.rect
    out = bytearray()
    out += _ops("/Tx", "BMC")
    out += _ops("q")
    out += border_and_background_ops(spec, rect)
    result = _layout.layout_text(value or "", spec, rect, multiline=spec.multiline or None)
    if result.lines:
        out += _clip_ops(rect, _layout.DEFAULT_PADDING)
        out += _text_block(result, spec)
    out += _ops("Q")
    out += _ops("EMC")
    return bytes(out)


def comb_appearance(spec: FieldSpec, value: Optional[str]) -> bytes:
    """Content bytes for a comb (``/Ff`` bit 24) text field: one glyph per cell."""
    rect = spec.rect
    result = _layout.layout_comb(value or "", spec, rect)
    out = bytearray()
    out += _ops("/Tx", "BMC")
    out += _ops("q")
    out += border_and_background_ops(spec, rect)
    if result.characters:
        res = _font_resource_name(spec)
        out += _ops("BT")
        out += _ops("/" + res, _fmt(result.font_size), "Tf")
        out += _color_op(spec.text_color, fill=True)
        prev_x = 0.0
        for ch, x in zip(result.characters, result.positions):
            out += _ops(_fmt(x - prev_x), _fmt(result.baseline if prev_x == 0.0 else 0.0), "Td")
            out += (b"(" + escape_pdf_text(ch) + b") Tj\n")
            prev_x = x
        out += _ops("ET")
    out += _ops("Q")
    out += _ops("EMC")
    return bytes(out)


def checkbox_appearance(spec: FieldSpec, on: bool) -> bytes:
    """Content bytes for one checkbox state (either the ``Off`` or the export state)."""
    rect = spec.rect
    out = bytearray()
    out += _ops("q")
    out += border_and_background_ops(spec, rect)
    if on:
        style = "check"
        path = _layout.check_mark_path(rect, style)
        if path:
            out += _color_op(spec.text_color, fill=False)
            w = max(0.8, min(rect.width, rect.height) * 0.14)
            out += _ops(_fmt(w), "w", "1 J 1 j")
            out += _path_ops(path)
            out += _ops("S")
    out += _ops("Q")
    return bytes(out)


def radio_appearance(spec: FieldSpec, on: bool) -> bytes:
    """Content bytes for one radio-button state: a ring, filled when selected."""
    rect = spec.rect.normalized()
    cx, cy = rect.width / 2.0, rect.height / 2.0
    r = max(0.5, min(rect.width, rect.height) / 2.0 - max(1.0, spec.border_width))
    out = bytearray()
    out += _ops("q")
    out += _color_op(spec.border_color or (0.0, 0.0, 0.0), fill=False)
    out += _ops(_fmt(max(0.6, r * 0.12)), "w")
    out += _path_ops(_layout.circle_path(cx, cy, r))
    out += _ops("S")
    if on:
        dot_r = r * _layout.RADIO_DOT_RATIO
        out += _color_op(spec.text_color, fill=True)
        out += _path_ops(_layout.circle_path(cx, cy, dot_r))
        out += _ops("f")
    out += _ops("Q")
    return bytes(out)


def choice_appearance(spec: FieldSpec, value: Optional[str]) -> bytes:
    """Content bytes for a combo/list choice field: the selected text plus an arrow."""
    rect = spec.rect
    out = bytearray()
    out += _ops("/Tx", "BMC")
    out += _ops("q")
    out += border_and_background_ops(spec, rect)
    result = _layout.layout_choice(value or "", spec, rect)
    if result.lines:
        out += _clip_ops(rect, _layout.DEFAULT_PADDING)
        out += _text_block(result, spec)
    arrow = _layout.arrow_path(rect)
    if arrow:
        out += _ops("q")
        out += _color_op((0.4, 0.4, 0.4), fill=True)
        out += _path_ops(arrow)
        out += _ops("f")
        out += _ops("Q")
    out += _ops("Q")
    out += _ops("EMC")
    return bytes(out)


def signature_placeholder_appearance(spec: FieldSpec) -> bytes:
    """An unsigned placeholder: a dashed border and, if present, a small label.

    Never resembles an actual signature -- ZFP creates the field and leaves it visibly
    empty; applying a signature is a separate, policy-gated operation.
    """
    rect = spec.rect
    out = bytearray()
    out += _ops("q")
    out += _color_op((0.55, 0.55, 0.55), fill=False)
    out += _ops("0.75", "w", "[2 2] 0 d")
    inset = 1.0
    out += _ops(_fmt(inset), _fmt(inset), _fmt(max(0.0, rect.width - 2 * inset)),
               _fmt(max(0.0, rect.height - 2 * inset)), "re", "S")
    out += _ops("[] 0 d")
    label = spec.tooltip or ""
    if label:
        small = FieldSpec(
            name=spec.name, field_type=FieldType.TEXT, page=spec.page, rect=rect,
            font_name=spec.font_name, font_size=min(7.0, rect.height * 0.4),
            alignment=1, text_color=(0.55, 0.55, 0.55),
        )
        result = _layout.layout_text(label, small, rect)
        out += _text_block(result, small)
    out += _ops("Q")
    return bytes(out)


def sanitize_export_value(value: str) -> str:
    """A checkbox/radio export value safe to use as a PDF name."""
    value = (value or "Yes").strip() or "Yes"
    safe = []
    for ch in value:
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("_") or "Yes"
    return result[:127]


def build_xobject(doc: "Document", spec: FieldSpec, content: bytes,
                  resources: "PdfDict") -> "PdfRef":
    """Wrap ``content`` as a ``/Type /XObject /Subtype /Form`` and register it.

    ``resources`` is shared, not copied, into the XObject dictionary -- callers pass the
    AcroForm's ``/DR`` (or a subset naming just the font in use) so the resource name a
    ``Tf`` operand references actually resolves.
    """
    from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfStream

    rect = spec.rect.normalized()
    encoded = encode_flate(content)
    stream_dict = PdfDict({
        "Type": PdfName("XObject"),
        "Subtype": PdfName("Form"),
        "FormType": 1,
        "BBox": PdfArray([0, 0, round(rect.width, 3), round(rect.height, 3)]),
        "Matrix": PdfArray([1, 0, 0, 1, 0, 0]),
        "Resources": resources,
        "Filter": PdfName("FlateDecode"),
        "Length": len(encoded),
    })
    stream = PdfStream(stream_dict, encoded)
    return doc.writer.add_object(stream)


def appearance_for(doc: "Document", spec: FieldSpec, value: Optional[str]
                   ) -> Union["PdfRef", Dict[str, "PdfRef"]]:
    """The single dispatch point the AcroForm writer calls for any field type.

    Returns one :class:`PdfRef` (``/AP /N`` directly) for text-like, choice and
    signature fields, or a ``{state_name: ref}`` mapping for checkboxes and radio
    buttons, whose ``/AP /N`` is itself a state dictionary.
    """

    resources = _resources_for(doc, spec)

    if spec.field_type in (FieldType.CHECKBOX,):
        on_name = sanitize_export_value(spec.export_value or "Yes")
        off_bytes = checkbox_appearance(spec, on=False)
        on_bytes = checkbox_appearance(spec, on=True)
        return {
            "Off": build_xobject(doc, spec, off_bytes, resources),
            on_name: build_xobject(doc, spec, on_bytes, resources),
        }
    if spec.field_type is FieldType.RADIO:
        on_name = sanitize_export_value(spec.export_value or "Yes")
        off_bytes = radio_appearance(spec, on=False)
        on_bytes = radio_appearance(spec, on=True)
        return {
            "Off": build_xobject(doc, spec, off_bytes, resources),
            on_name: build_xobject(doc, spec, on_bytes, resources),
        }
    if spec.field_type is FieldType.SIGNATURE:
        content = signature_placeholder_appearance(spec)
        return build_xobject(doc, spec, content, resources)
    if spec.field_type is FieldType.COMB:
        content = comb_appearance(spec, value)
        return build_xobject(doc, spec, content, resources)
    if spec.field_type in (FieldType.CHOICE, FieldType.LISTBOX):
        content = choice_appearance(spec, value)
        return build_xobject(doc, spec, content, resources)
    content = text_appearance(spec, value, resources)
    return build_xobject(doc, spec, content, resources)


def _resources_for(doc: "Document", spec: FieldSpec) -> "PdfDict":
    """A ``/Resources`` dict naming the field's font, backed by the AcroForm's ``/DR``."""
    from ..pdfio.objects import PdfDict

    short, _ref = fonts.ensure_standard_font(doc, spec.font_name)
    acroform = doc.ensure_acroform()
    dr = doc.resolve(acroform.get("DR"))
    font_dict = doc.resolve(dr.get("Font")) if isinstance(dr, PdfDict) else PdfDict()
    return PdfDict({"Font": font_dict})


__all__ = [
    "text_appearance", "comb_appearance", "checkbox_appearance", "radio_appearance",
    "choice_appearance", "signature_placeholder_appearance", "border_and_background_ops",
    "build_xobject", "appearance_for", "sanitize_export_value",
]
