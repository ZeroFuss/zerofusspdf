"""The synthetic form generator: real PDFs with exact, machine-checkable ground truth.

Everything downstream of perception is measured against what this module emits.  A
:class:`SyntheticForm` carries the PDF bytes *and* the rectangle, canonical key and
expected value of every field that was drawn -- derived from the drawing numbers
themselves, so a detector's output can be scored without a human ever looking at the
page.

Two invariants matter more than anything else here:

* **Determinism.**  A seed fully determines the bytes.  All randomness comes from a
  private ``random.Random(seed)``; the global :mod:`random` module is never touched, and
  no wall-clock value reaches the file.
* **Rotation independence.**  ``options.rotation`` writes ``/Rotate`` on every page but
  leaves the ground-truth rectangles in unrotated PDF user space, because that is the
  space a detector reports in.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.errors import ValidationError
from ..core.geometry import Rect
from ..core.types import FieldType
from ..pdfio.document import Document
from .content import attach_page_content
from .layouts import (
    PAGE_KINDS,
    REPEAT_LABELS,
    TITLES,
    FieldMark,
    LabelSpec,
    Style,
    build_style,
    draw_page,
    specs_for,
)

__all__ = [
    "KINDS",
    "ROTATIONS",
    "GroundTruthField",
    "SyntheticForm",
    "SynthOptions",
    "generate",
    "generate_corpus",
]

#: Every document kind :func:`generate` understands.
KINDS: Tuple[str, ...] = PAGE_KINDS + ("multipage",)

#: The ``/Rotate`` values a synthetic page may carry.
ROTATIONS: Tuple[int, ...] = (0, 90, 180, 270)

#: Pages a ``multipage`` document gets when the caller does not ask for more.
DEFAULT_MULTIPAGE_PAGES = 3


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------


@dataclass
class GroundTruthField:
    """One field that was really drawn, and where exactly it is.

    Attributes:
        name: Unique, PDF-safe field name derived from the canonical key.
        canonical_key: The :mod:`zfp.ontology` key the label resolves to.  Never empty.
        field_type: The semantic type the field should be detected as.
        page: Zero-based page index.
        rect: The field rectangle in unrotated PDF user space (y-up, points).
        label: The printed label as drawn.
        expected_value: A deterministic value the autofill path may fill in.
    """

    name: str
    canonical_key: str
    field_type: FieldType
    page: int
    rect: Rect
    label: str
    expected_value: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-able mapping of this field."""
        return {
            "name": self.name,
            "canonical_key": self.canonical_key,
            "field_type": self.field_type.value,
            "page": int(self.page),
            "rect": [round(v, 4) for v in self.rect.as_list()],
            "label": self.label,
            "expected_value": self.expected_value,
        }


@dataclass
class SyntheticForm:
    """A generated PDF together with its complete ground truth.

    Attributes:
        pdf_bytes: The serialized document -- a real, parseable PDF.
        fields: Every ground-truth field, in reading order (page, top to bottom, left to
            right).
        seed: The seed that produced it.
        kind: The requested document kind.
        pages: How many pages the document actually has.
    """

    pdf_bytes: bytes
    fields: List[GroundTruthField] = field(default_factory=list)
    seed: int = 0
    kind: str = "underline"
    pages: int = 1

    def save(self, path: "str | os.PathLike") -> None:
        """Write the PDF to ``path``."""
        with open(os.fspath(path), "wb") as handle:
            handle.write(self.pdf_bytes)

    def truth_dict(self) -> Dict[str, Any]:
        """Return ``{"kind", "seed", "fields"}`` as plain JSON-able values."""
        return {
            "kind": self.kind,
            "seed": int(self.seed),
            "fields": [f.as_dict() for f in self.fields],
        }

    def document(self) -> Document:
        """Open the bytes as a :class:`~zfp.pdfio.document.Document`."""
        return Document.open(self.pdf_bytes)

    def fields_on_page(self, page: int) -> List[GroundTruthField]:
        """Every ground-truth field on ``page``, in reading order."""
        return [f for f in self.fields if f.page == int(page)]

    def labels(self) -> List[str]:
        """Every printed label, in reading order (duplicates kept)."""
        return [f.label for f in self.fields]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SyntheticForm(kind=%r, seed=%d, pages=%d, fields=%d, bytes=%d)" % (
            self.kind,
            self.seed,
            self.pages,
            len(self.fields),
            len(self.pdf_bytes),
        )


# --------------------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------------------


@dataclass
class SynthOptions:
    """What to generate.

    The three appearance attributes (``font``, ``font_size``, ``line_width``) are
    *defaults, not commands*: left at their declared value they are drawn from the seeded
    generator, so a corpus varies realistically; set to anything else they are honoured
    verbatim on every page.  Either way the output depends only on the options and the
    seed.
    """

    kind: str = "underline"
    pages: int = 1
    seed: int = 0
    font: str = "Helvetica"
    font_size: float = 10.0
    line_width: float = 0.6
    include_sections: bool = True
    rotation: int = 0
    locale: str = "en_US"
    page_width: float = 612.0
    page_height: float = 792.0

    def validated(self) -> "SynthOptions":
        """Return a normalized copy, raising on anything unusable.

        Raises:
            ValidationError: Unknown kind, non-positive page count or page size, or a
                rotation that is not a multiple of 90.
        """
        kind = str(self.kind)
        if kind not in KINDS:
            raise ValidationError(
                "unknown synth kind %r; expected one of %s" % (kind, ", ".join(KINDS))
            )
        pages = int(self.pages)
        if pages < 1:
            raise ValidationError("pages must be at least 1, got %d" % pages)
        rotation = int(self.rotation) % 360
        if rotation not in ROTATIONS:
            raise ValidationError(
                "rotation must be a multiple of 90, got %r" % (self.rotation,)
            )
        width = float(self.page_width)
        height = float(self.page_height)
        if width < 144.0 or height < 144.0:
            raise ValidationError(
                "page size must be at least 144x144 points, got %rx%r" % (width, height)
            )
        return SynthOptions(
            kind=kind,
            pages=pages,
            seed=int(self.seed),
            font=str(self.font),
            font_size=float(self.font_size),
            line_width=float(self.line_width),
            include_sections=bool(self.include_sections),
            rotation=rotation,
            locale=str(self.locale),
            page_width=width,
            page_height=height,
        )


_DEFAULTS = SynthOptions()


# --------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Lowercase ``text`` down to ``[a-z0-9_]``, collapsing runs of other characters."""
    out: List[str] = []
    previous_underscore = False
    for ch in str(text).lower():
        if ch.isalnum() and ch.isascii():
            out.append(ch)
            previous_underscore = False
        elif not previous_underscore:
            out.append("_")
            previous_underscore = True
    return "".join(out).strip("_")


def _field_name(mark: FieldMark, used: Set[str]) -> str:
    """Return a unique PDF-safe field name for ``mark``."""
    base = _slug(mark.canonical_key.replace(".", "_")) or "field"
    if mark.option:
        suffix = _slug(mark.option)
        if suffix:
            base = "%s_%s" % (base, suffix)
    name = base
    counter = 2
    while name in used:
        name = "%s_%d" % (base, counter)
        counter += 1
    used.add(name)
    return name


def _reading_order(marks: Sequence[FieldMark]) -> List[FieldMark]:
    """Sort one page's marks top-to-bottom then left-to-right, stably and exactly."""
    return sorted(
        marks,
        key=lambda m: (
            -round(m.rect.y1, 3),
            round(m.rect.x0, 3),
            round(m.rect.x1, 3),
            m.label,
            m.option,
        ),
    )


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def _page_plan(options: SynthOptions) -> List[str]:
    """Return the per-page template for each sheet of the document."""
    if options.kind != "multipage":
        return [options.kind] * options.pages
    pages = max(DEFAULT_MULTIPAGE_PAGES, options.pages)
    return ["mixed"] * (pages - 1) + ["signature"]


def _style_for(options: SynthOptions, rng: random.Random) -> Style:
    """Build the document style, honouring any appearance the caller pinned."""
    return build_style(
        rng,
        font=options.font,
        font_size=options.font_size,
        line_width=options.line_width,
        page_width=options.page_width,
        page_height=options.page_height,
        randomize_font=options.font == _DEFAULTS.font,
        randomize_size=options.font_size == _DEFAULTS.font_size,
        randomize_width=options.line_width == _DEFAULTS.line_width,
    )


def generate(options: SynthOptions) -> SyntheticForm:
    """Generate one synthetic form.

    Args:
        options: What to build.  See :class:`SynthOptions`.

    Returns:
        A :class:`SyntheticForm` whose ``pdf_bytes`` reopen through
        :meth:`zfp.pdfio.document.Document.open` and whose ``fields`` describe every
        drawn field exactly.

    Raises:
        ValidationError: The options are unusable.
    """
    opts = options.validated()
    rng = random.Random(opts.seed)
    style = _style_for(opts, rng)
    title = rng.choice(TITLES)
    plan = _page_plan(opts)
    repeat: Sequence[LabelSpec] = specs_for(REPEAT_LABELS) if opts.kind == "multipage" else ()

    doc = Document.from_pages_blank(len(plan), opts.page_width, opts.page_height)
    used_names: Set[str] = set()
    fields: List[GroundTruthField] = []

    for index, page_kind in enumerate(plan):
        drawing = draw_page(
            page_kind,
            rng,
            style,
            index,
            title=title,
            include_sections=opts.include_sections,
            first_page=index == 0,
            repeat=repeat,
        )
        page = doc.page(index)
        if opts.rotation:
            page.dict["Rotate"] = opts.rotation
        attach_page_content(doc, index, drawing.content, drawing.fonts)
        for mark in _reading_order(drawing.marks):
            fields.append(
                GroundTruthField(
                    name=_field_name(mark, used_names),
                    canonical_key=mark.canonical_key,
                    field_type=mark.field_type,
                    page=index,
                    rect=mark.rect,
                    label=mark.label,
                    expected_value=mark.expected_value,
                )
            )

    return SyntheticForm(
        pdf_bytes=doc.to_bytes(incremental=False),
        fields=fields,
        seed=opts.seed,
        kind=opts.kind,
        pages=len(plan),
    )


def generate_corpus(
    n: int,
    *,
    seed: int = 0,
    kinds: Optional[Iterable[str]] = None,
) -> List[SyntheticForm]:
    """Generate ``n`` forms, cycling through ``kinds`` and advancing the seed each time.

    Args:
        n: How many documents to build.  ``0`` yields an empty list.
        seed: Seed of the first document; document ``i`` uses ``seed + i``.
        kinds: The kinds to cycle through.  Defaults to every kind in :data:`KINDS`.

    Returns:
        The generated forms, in the order they were produced.

    Raises:
        ValidationError: ``n`` is negative, or ``kinds`` is empty or names an unknown
            kind.
    """
    count = int(n)
    if count < 0:
        raise ValidationError("generate_corpus needs a non-negative count, got %d" % count)
    wanted: Tuple[str, ...] = tuple(str(k) for k in kinds) if kinds is not None else KINDS
    if not wanted:
        raise ValidationError("generate_corpus needs at least one kind")
    unknown = [k for k in wanted if k not in KINDS]
    if unknown:
        raise ValidationError(
            "unknown synth kind(s) %s; expected one of %s"
            % (", ".join(repr(k) for k in unknown), ", ".join(KINDS))
        )
    return [
        generate(SynthOptions(kind=wanted[i % len(wanted)], seed=seed + i))
        for i in range(count)
    ]
