"""The AcroForm writer: candidates and vault values become real, interoperable PDF fields.

"Creating the actual form should be native PDF, not just a visual overlay in the web UI.
Each accepted candidate becomes a genuine AcroForm widget, so the finished PDF remains
fillable in Acrobat, browser viewers and other compliant readers." This module is where
that thesis is implemented: every :class:`~zfp.core.types.FieldSpec` in a
:class:`~zfp.core.types.FormSchema` becomes a field dictionary, a widget annotation with a
real appearance stream, and an entry in the page's ``/Annots`` and the AcroForm's
``/Fields`` -- written as an incremental update, so the original page content is never
touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..core.geometry import Rect
from ..core.logging import get_logger
from ..core.types import FieldSpec, FieldType, FormSchema
from ..pdfio import fonts
from ..pdfio.document import Document
from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfString
from . import flags as F

_log = get_logger(__name__)

try:  # pragma: no cover - exercised via write(); optional so import never hard-fails
    from ..appearance import streams as _appearance
except Exception:  # pragma: no cover
    _appearance = None  # type: ignore[assignment]


@dataclass
class WriteReport:
    """What :meth:`AcroFormWriter.write` actually did."""

    fields_written: int = 0
    widgets_written: int = 0
    pages_touched: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_refs: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fields_written": self.fields_written,
            "widgets_written": self.widgets_written,
            "pages_touched": list(self.pages_touched),
            "warnings": list(self.warnings),
            "field_refs": dict(self.field_refs),
        }


def _sanitize_name_segment(name: str) -> str:
    name = (name or "field").strip()
    out = []
    for ch in name:
        if ch in "().[]{}/<>\\" or ch.isspace():
            out.append("_")
        else:
            out.append(ch)
    result = "".join(out).strip("_.") or "field"
    return result


def _dedupe(name: str, seen: Dict[str, int]) -> str:
    count = seen.get(name, 0)
    seen[name] = count + 1
    return name if count == 0 else "%s_%d" % (name, count)


def _radio_group_key(spec: FieldSpec) -> Optional[str]:
    if spec.field_type is FieldType.RADIO and spec.group:
        return spec.group
    return None


class AcroFormWriter:
    """Writes a :class:`FormSchema` into a :class:`Document` as real AcroForm fields."""

    def __init__(self, doc: Document, config: Any = None) -> None:
        self.doc = doc
        self.config = config
        self._name_registry: Dict[str, int] = {}
        self._field_refs: Dict[str, PdfRef] = {}
        self._needs_appearances = False

    # -- public API -------------------------------------------------------------------

    def write(self, schema: FormSchema) -> WriteReport:
        """Write every field in ``schema`` and return a summary of what happened."""
        report = WriteReport()
        acroform = self.doc.ensure_acroform()
        self._ensure_dr(acroform)

        fields_array = self.doc.resolve(acroform.get("Fields"))
        if not isinstance(fields_array, PdfArray):
            fields_array = PdfArray()
            acroform["Fields"] = fields_array

        radio_groups: Dict[str, List[FieldSpec]] = {}
        singles: List[FieldSpec] = []
        for spec in schema.fields:
            key = _radio_group_key(spec)
            if key is not None:
                radio_groups.setdefault(key, []).append(spec)
            else:
                singles.append(spec)

        pages_touched: set = set()

        for spec in singles:
            deduped_name = _dedupe(spec.name or "field", self._name_registry)
            if deduped_name != spec.name:
                import dataclasses
                spec = dataclasses.replace(spec, name=deduped_name)
            try:
                ref = self.write_field(spec)
            except Exception as exc:  # noqa: BLE001 - one bad field must not sink the run
                report.warnings.append("field %r failed: %s" % (spec.name, exc))
                continue
            self._register_top_level(spec.name, ref, fields_array, report)
            for page, _rect in spec.widgets():
                pages_touched.add(page)
                report.widgets_written += 1
            report.fields_written += 1

        for group_name, members in radio_groups.items():
            try:
                ref = self._write_radio_group(group_name, members)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append("radio group %r failed: %s" % (group_name, exc))
                continue
            self._register_top_level(group_name, ref, fields_array, report)
            for spec in members:
                for page, _rect in spec.widgets():
                    pages_touched.add(page)
                    report.widgets_written += 1
            report.fields_written += 1

        acroform["NeedAppearances"] = bool(self._needs_appearances)
        report.pages_touched = sorted(pages_touched)
        return report

    def write_field(self, spec: FieldSpec) -> PdfRef:
        """Write one field (text-like, checkbox, choice, signature, or comb) and its
        widget annotation(s), returning the field's own reference.

        A dotted ``spec.name`` (``"Applicant.Address.City"``) builds a real field tree:
        intermediate ``/T`` nodes with ``/Kids``, not a flat name with dots in it.
        """
        segments = [_sanitize_name_segment(s) for s in (spec.name or "field").split(".") if s]
        if not segments:
            segments = ["field"]
        return self._write_leaf(spec, segments[-1])

    def set_values(self, values: Mapping[str, str]) -> None:
        """Update ``/V`` (and ``/AS`` for buttons) on existing fields and regenerate
        their appearance streams. Works on a document ZFP did not create."""
        from .reader import field_tree

        tree = field_tree(self.doc)
        for name, value in values.items():
            entry = tree.get(name)
            if entry is None:
                continue
            self._apply_value(entry, value)

    def flatten(self, field_names: Optional[Iterable[str]] = None) -> None:
        """Draw each named field's current appearance into its page and remove the
        widget. Content is appended as a new content stream; nothing existing is
        rewritten. When ``field_names`` is ``None``, every field is flattened."""
        from .reader import field_tree

        tree = field_tree(self.doc)
        names = list(field_names) if field_names is not None else list(tree.keys())
        acroform = self.doc.ensure_acroform()
        fields_array = self.doc.resolve(acroform.get("Fields"))
        if not isinstance(fields_array, PdfArray):
            fields_array = PdfArray()

        for name in names:
            entry = tree.get(name)
            if entry is None:
                continue
            self._flatten_entry(entry)
            top_ref = entry.get("_top_ref")
            if isinstance(top_ref, PdfRef) and top_ref in fields_array:
                fields_array.remove(top_ref)
        acroform["Fields"] = fields_array

    # -- internals: leaf field construction ---------------------------------------------

    def _write_leaf(self, spec: FieldSpec, short_name: str) -> PdfRef:
        rect0 = spec.rect.normalized()
        page0 = self.doc.page(spec.page)
        appearance = self._appearance_for(spec)

        field_dict = PdfDict()
        field_dict["FT"] = PdfName(spec.pdf_kind)
        field_dict["T"] = PdfString.from_text(short_name)
        if spec.tooltip:
            field_dict["TU"] = PdfString.from_text(spec.tooltip)
        ff = F.field_flags(spec)
        if ff:
            field_dict["Ff"] = ff
        if spec.required:
            pass  # already folded into Ff above

        self._apply_type_specific(field_dict, spec, appearance)

        widgets: List[Tuple[int, Rect, PdfDict]] = []
        primary = self._build_widget_dict(spec, rect0, page0.ref, appearance, is_kid=False)
        widgets.append((spec.page, rect0, primary))

        if spec.extra_widgets:
            field_dict["Kids"] = kids = PdfArray()
            main_ref = self.doc.writer.add_object(field_dict)
            for extra_page, extra_rect in spec.extra_widgets:
                extra_appearance = self._appearance_for(spec, rect_override=extra_rect)
                page_n = self.doc.page(extra_page)
                kid = self._build_widget_dict(spec, extra_rect.normalized(), page_n.ref,
                                              extra_appearance, is_kid=True, parent=main_ref)
                kid["Rect"] = _rect_array(extra_rect.normalized())
                kid_ref = self.doc.writer.add_object(kid)
                kids.append(kid_ref)
                page_n.add_annotation(kid_ref)
            primary["Parent"] = main_ref
            primary_ref = self.doc.writer.add_object(primary)
            kids.append(primary_ref)
            page0.add_annotation(primary_ref)
            self.doc.writer.set_object(main_ref.num, field_dict)
            self._field_refs[spec.name] = main_ref
            return main_ref

        # Single widget: the field dictionary and the widget annotation are one object,
        # which is the common and simplest form.
        field_dict.update(primary)
        ref = self.doc.writer.add_object(field_dict)
        page0.add_annotation(ref)
        self._field_refs[spec.name] = ref
        return ref

    def _apply_type_specific(self, field_dict: PdfDict, spec: FieldSpec, appearance: Any) -> None:
        ft = spec.field_type
        if ft in (FieldType.CHECKBOX,):
            export = _appearance.sanitize_export_value(spec.export_value or "Yes") \
                if _appearance else (spec.export_value or "Yes")
            on = bool(spec.value) and str(spec.value).lower() not in ("off", "0", "false", "")
            field_dict["V"] = PdfName(export if on else "Off")
            if spec.default_value:
                field_dict["DV"] = PdfName(export if spec.default_value else "Off")
        elif ft is FieldType.RADIO:
            pass  # handled by _write_radio_group
        elif ft in (FieldType.CHOICE, FieldType.LISTBOX):
            opts = PdfArray()
            for choice in spec.choices:
                opts.append(PdfString.from_text(choice))
            field_dict["Opt"] = opts
            if spec.value:
                field_dict["V"] = PdfString.from_text(spec.value)
                if spec.value in spec.choices:
                    field_dict["I"] = PdfArray([spec.choices.index(spec.value)])
            if spec.alignment:
                field_dict["Q"] = spec.alignment
        elif ft is FieldType.SIGNATURE:
            pass  # never fabricate a /V
        else:
            da = "/%s %s Tf %s" % (
                fonts.resource_name(fonts.resolve_base_font(spec.font_name)),
                _fmt(spec.font_size or 0),
                _rgb(spec.text_color),
            )
            field_dict["DA"] = PdfString.from_text(da)
            if spec.value is not None:
                field_dict["V"] = PdfString.from_text(spec.value)
            if spec.default_value is not None:
                field_dict["DV"] = PdfString.from_text(spec.default_value)
            max_len = spec.max_length or spec.comb_cells
            if max_len:
                field_dict["MaxLen"] = int(max_len)
            if spec.alignment:
                field_dict["Q"] = spec.alignment

    def _build_widget_dict(self, spec: FieldSpec, rect: Rect, page_ref: Optional[PdfRef],
                           appearance: Any, *, is_kid: bool, parent: Optional[PdfRef] = None
                           ) -> PdfDict:
        widget = PdfDict()
        widget["Type"] = PdfName("Annot")
        widget["Subtype"] = PdfName("Widget")
        widget["Rect"] = _rect_array(rect)
        widget["F"] = F.annotation_flags(spec)
        if page_ref is not None:
            widget["P"] = page_ref
        if parent is not None:
            widget["Parent"] = parent

        mk = PdfDict()
        if spec.border_color is not None:
            mk["BC"] = PdfArray([round(c, 3) for c in spec.border_color])
        if spec.background_color is not None:
            mk["BG"] = PdfArray([round(c, 3) for c in spec.background_color])
        if mk:
            widget["MK"] = mk

        if isinstance(appearance, dict):
            ap = PdfDict({"N": PdfDict({str(k): v for k, v in appearance.items()})})
            widget["AP"] = ap
            on_state = next((k for k in appearance if k != "Off"), "Off")
            selected = bool(spec.value) and str(spec.value).lower() not in ("off", "0", "false", "")
            widget["AS"] = PdfName(on_state if selected else "Off")
        elif appearance is not None:
            widget["AP"] = PdfDict({"N": appearance})
        return widget

    def _appearance_for(self, spec: FieldSpec, *, rect_override: Optional[Rect] = None) -> Any:
        if _appearance is None:
            return None
        use_spec = spec
        if rect_override is not None:
            use_spec = _with_rect(spec, rect_override)
        try:
            return _appearance.appearance_for(self.doc, use_spec, spec.value)
        except Exception as exc:  # noqa: BLE001 - degrade to /NeedAppearances rather than fail
            _log.warning("appearance generation failed for %r: %s; falling back to "
                        "/NeedAppearances", spec.name, exc)
            self._needs_appearances = True
            return None

    # -- internals: radio groups ---------------------------------------------------------

    def _write_radio_group(self, group_name: str, members: List[FieldSpec]) -> PdfRef:
        parent = PdfDict()
        parent["FT"] = PdfName("Btn")
        parent["T"] = PdfString.from_text(_sanitize_name_segment(group_name))
        parent["Ff"] = F.RADIO | F.NO_TOGGLE_TO_OFF
        selected_export = None
        kids = PdfArray()
        parent["Kids"] = kids
        parent_ref = self.doc.writer.add_object(parent)

        for spec in members:
            export = _appearance.sanitize_export_value(spec.export_value or spec.name) \
                if _appearance else (spec.export_value or "Yes")
            on = bool(spec.value) and str(spec.value).lower() not in ("off", "0", "false", "")
            if on:
                selected_export = export
            appearance = self._appearance_for(spec)
            page = self.doc.page(spec.page)
            kid = self._build_widget_dict(spec, spec.rect.normalized(), page.ref,
                                          appearance, is_kid=True, parent=parent_ref)
            if isinstance(appearance, dict):
                kid["AS"] = PdfName(export if on else "Off")
            kid_ref = self.doc.writer.add_object(kid)
            kids.append(kid_ref)
            page.add_annotation(kid_ref)

        parent["V"] = PdfName(selected_export) if selected_export else PdfName("Off")
        self.doc.writer.set_object(parent_ref.num, parent)
        self._field_refs[group_name] = parent_ref
        return parent_ref

    # -- internals: top-level registration -----------------------------------------------

    def _register_top_level(self, name: str, ref: PdfRef, fields_array: PdfArray,
                            report: WriteReport) -> None:
        if ref not in fields_array:
            fields_array.append(ref)
        report.field_refs[name] = str(ref)

    def _ensure_dr(self, acroform: PdfDict) -> None:
        fonts.ensure_standard_font(self.doc, "Helvetica")

    # -- internals: set_values / flatten --------------------------------------------------

    def _apply_value(self, entry: Dict[str, Any], value: str) -> None:
        obj = entry.get("dict")
        num = entry.get("num")
        if not isinstance(obj, PdfDict) or num is None:
            return
        kind = entry.get("field_type")
        if kind in (FieldType.CHECKBOX, FieldType.RADIO):
            export = value or "Off"
            obj["V"] = PdfName(export)
            for kid_num, kid_dict in entry.get("kids", []):
                on_states = entry.get("kid_states", {}).get(kid_num, set())
                kid_dict["AS"] = PdfName(export) if export in on_states else PdfName("Off")
                self.doc.writer.set_object(kid_num, kid_dict)
        else:
            obj["V"] = PdfString.from_text(value)
        self.doc.writer.set_object(num, obj)

    def _flatten_entry(self, entry: Dict[str, Any]) -> None:
        for page_index, rect, ap_ref in entry.get("widgets", []):
            if ap_ref is None:
                continue
            stream = self.doc.resolve(ap_ref)
            if not hasattr(stream, "decoded"):
                continue
            try:
                content = stream.decoded(self.doc)
            except Exception:  # noqa: BLE001
                continue
            page = self.doc.page(page_index)
            r = rect.normalized()
            wrapper = ("q 1 0 0 1 %s %s cm\n" % (_fmt(r.x0), _fmt(r.y0))).encode("latin-1")
            wrapper += content + b"\nQ\n"
            self._append_content(page, wrapper)
            annots = page.dict.get("Annots")
            arr = self.doc.resolve(annots) if not isinstance(annots, PdfArray) else annots
            if isinstance(arr, PdfArray):
                for widget_ref in list(entry.get("widget_refs", [])):
                    if widget_ref in arr:
                        arr.remove(widget_ref)
            page.touch()

    def _append_content(self, page: Any, extra: bytes) -> None:
        from ..pdfio.filters import encode_flate
        from ..pdfio.objects import PdfStream

        existing = page.dict.get("Contents")
        encoded = encode_flate(extra)
        new_stream = PdfStream(PdfDict({"Filter": PdfName("FlateDecode"), "Length": len(encoded)}),
                               encoded)
        new_ref = self.doc.writer.add_object(new_stream)
        if isinstance(existing, PdfArray):
            existing.append(new_ref)
            arr = existing
        elif isinstance(existing, PdfRef):
            resolved = self.doc.resolve(existing)
            if isinstance(resolved, PdfArray):
                resolved.append(new_ref)
                self.doc.writer.set_object(existing.num, resolved)
                page.touch()
                return
            arr = PdfArray([existing, new_ref])
        elif existing is not None:
            arr = PdfArray([existing, new_ref])
        else:
            arr = PdfArray([new_ref])
        page.dict["Contents"] = arr
        page.touch()


def _fmt(n: float) -> str:
    r = round(float(n), 3)
    if r == int(r):
        return str(int(r))
    return ("%.3f" % r).rstrip("0").rstrip(".")


def _rgb(color: Tuple[float, float, float]) -> str:
    return "%s %s %s rg" % (_fmt(color[0]), _fmt(color[1]), _fmt(color[2]))


def _rect_array(rect: Rect) -> PdfArray:
    return PdfArray([round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3)])


def _with_rect(spec: FieldSpec, rect: Rect) -> FieldSpec:
    import dataclasses
    return dataclasses.replace(spec, rect=rect)


__all__ = ["AcroFormWriter", "WriteReport"]
