"""Deterministic form layout templates and their exact ground truth.

Every template here draws a realistic printed form -- a title, section headings, labels,
rules, boxes, checkboxes, comb cells, tables -- and, for each field it draws, records the
rectangle a detector is expected to produce.  The rectangle is *derived from the same
numbers that were painted*, never measured afterwards, which is what makes the corpus
usable as ground truth.

The conventions the detectors are measured against:

* **Underline** -- the field is the rule's x-range, starting
  ``DetectionConfig.underline_gap_pt`` above the rule and
  ``DetectionConfig.field_height_pt`` tall.
* **Box / table cell / comb** -- the drawn rectangle deflated by the stroke width, so the
  ground truth is the *inside* of the ink.
* **Checkbox / radio** -- the glyph's own bounding box.
* **Borderless** -- the blank region under the label; nothing is drawn at all.

All coordinates are PDF user space (y-up, page origin, points).  Page rotation is applied
by the generator as ``/Rotate`` and deliberately does *not* move these rectangles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig
from ..core.errors import ValidationError
from ..core.geometry import Rect
from ..core.types import FieldType
from ..ontology import get as ontology_get
from ..ontology import lookup as ontology_lookup
from ..pdfio.fonts import LEADING_FACTOR, resolve_base_font, text_width, wrap_text
from .content import ContentBuilder

__all__ = [
    "DET",
    "PAGE_KINDS",
    "LabelSpec",
    "FieldMark",
    "PageDraw",
    "Style",
    "Canvas",
    "build_style",
    "draw_page",
    "section_specs",
    "spec_for",
    "specs_for",
    "SIGNATURE_LABELS",
    "SECTIONS",
    "CHECKBOX_GROUPS",
    "COMB_FIELDS",
    "TABLE_COLUMNS",
    "REPEAT_LABELS",
    "TITLES",
    "SAMPLE_VALUES",
]

#: Detection thresholds the ground truth is expressed in terms of.
DET = DetectionConfig()

#: The per-page templates :func:`draw_page` understands.  ``multipage`` is a *document*
#: kind: the generator expands it into a sequence of these.
PAGE_KINDS: Tuple[str, ...] = (
    "underline",
    "boxed",
    "checkbox",
    "table",
    "comb",
    "borderless",
    "mixed",
    "signature",
)

# Layout metrics that are not worth randomizing.
COL_GAP = 16.0          # horizontal gutter between columns of a row
LABEL_GAP = 5.0         # space between a label and the input area that follows it
RULE_DROP = 3.0         # how far under the label baseline an underline rule sits
MIN_RULE = 44.0         # shortest acceptable underline (min_line_length_pt is 24)
MAX_FIELDS_PER_PAGE = 25
MIN_FIELDS_PER_PAGE = 8

_TEXT_FAMILIES: Tuple[str, ...] = ("Helvetica", "Times-Roman", "Courier")
_BOLD_OF: Dict[str, str] = {
    "Helvetica": "Helvetica-Bold",
    "Times-Roman": "Times-Bold",
    "Courier": "Courier-Bold",
}


# --------------------------------------------------------------------------------------
# Label vocabulary, all of it ontology-backed
# --------------------------------------------------------------------------------------

#: ``(heading, labels)``.  Every label is resolved through :func:`zfp.ontology.lookup`;
#: any that does not resolve is dropped by :func:`specs_for`, so a ground-truth field
#: without a canonical key can never be emitted.
SECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Applicant Information",
        (
            "First Name",
            "Middle Initial",
            "Last Name",
            "Suffix",
            "Date of Birth",
            "Social Security Number",
            "Citizenship",
            "Occupation",
        ),
    ),
    (
        "Contact Information",
        (
            "Street Address",
            "Address Line 2",
            "City",
            "State",
            "ZIP Code",
            "Country",
            "Email Address",
            "Phone Number",
            "Mobile Phone",
            "Fax",
        ),
    ),
    (
        "Employment History",
        (
            "Employer",
            "Job Title",
            "Department",
            "Employee ID",
            "Date of Hire",
            "Supervisor",
            "Salary",
            "Employer Address",
        ),
    ),
    (
        "Business Information",
        (
            "Company Name",
            "DBA",
            "EIN",
            "Website",
            "Invoice Number",
            "Reference Number",
        ),
    ),
    (
        "Financial Information",
        (
            "Bank Name",
            "Account Number",
            "Routing Number",
            "Gross Income",
            "Card Number",
            "Expiration",
            "CVV",
        ),
    ),
    (
        "Insurance",
        (
            "Member ID",
            "Group Number",
            "Policy Number",
            "Policy Holder",
            "Effective Date",
            "Expiration Date",
        ),
    ),
    (
        "Education",
        ("School", "Degree", "Major", "GPA", "Graduation Date", "Student ID"),
    ),
    (
        "Medical Information",
        (
            "Blood Type",
            "Allergies",
            "Medications",
            "Primary Care Physician",
            "Emergency Contact",
        ),
    ),
)

#: Checkbox and radio groups: ``(stem label, options)``.
CHECKBOX_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Marital Status", ("Single", "Married", "Divorced", "Widowed")),
    ("Gender", ("Male", "Female", "Other")),
    ("Account Type", ("Checking", "Savings")),
    ("Payment Method", ("Check", "Credit Card", "ACH")),
    ("Filing Status", ("Single", "Married Filing Jointly", "Head of Household")),
    ("Preferred Contact Method", ("Email", "Phone", "Mail")),
    ("Veteran Status", ("Yes", "No")),
    ("Citizenship", ("Yes", "No")),
    ("Relationship", ("Spouse", "Child", "Parent", "Other")),
)

#: Comb fields: ``(label, cell count)`` -- one value split over N adjacent cells.
COMB_FIELDS: Tuple[Tuple[str, int], ...] = (
    ("Social Security Number", 9),
    ("Phone Number", 10),
    ("Date of Birth", 8),
    ("ZIP Code", 5),
    ("EIN", 9),
    ("Account Number", 10),
    ("Routing Number", 9),
    ("Card Number", 12),
    ("Policy Number", 10),
    ("Member ID", 9),
    ("Employee ID", 6),
    ("Student ID", 8),
    ("Invoice Number", 8),
)

#: Short labels that fit in a table column heading.
TABLE_COLUMNS: Tuple[str, ...] = (
    "First Name",
    "Last Name",
    "City",
    "State",
    "ZIP Code",
    "Date",
    "Amount",
    "Phone",
    "Email",
    "Title",
    "Department",
    "Employer",
)

#: The block a multi-page form repeats on every sheet.  These are what the
#: repeat-consistency detectors are supposed to notice.
REPEAT_LABELS: Tuple[str, ...] = (
    "Case Number",
    "Date",
    "Last Name",
    "Social Security Number",
)

#: Signature block, in drawing order.
SIGNATURE_LABELS: Tuple[str, ...] = (
    "Signature",
    "Date",
    "Printed Name",
    "Title",
    "Initials",
)

TITLES: Tuple[str, ...] = (
    "Employment Application",
    "Account Opening Form",
    "Patient Intake Form",
    "Insurance Enrollment Form",
    "Loan Application",
    "Vendor Registration Form",
    "Student Enrollment Form",
    "Tax Information Update",
)

_CERTIFICATION_TEXT = (
    "By signing below I certify that the information provided in this form is true, "
    "complete and correct to the best of my knowledge, and I authorize verification of "
    "the statements made herein."
)

_INTRO_TEXT = (
    "Please complete every section in ink. Print clearly and do not leave any required "
    "item blank. Attach additional sheets if more space is needed."
)

#: Deterministic sample values, keyed by canonical key.  Used for
#: :attr:`~zfp.synth.generator.GroundTruthField.expected_value`, which the autofill and
#: verification tests fill against.
SAMPLE_VALUES: Dict[str, str] = {
    "person.name.first": "Jordan",
    "person.name.middle_initial": "Q",
    "person.name.last": "Avery",
    "person.name.full": "Jordan Q Avery",
    "person.name.prefix": "Ms",
    "person.name.suffix": "Jr",
    "person.initials": "JQA",
    "person.date_of_birth": "04/17/1984",
    "person.ssn": "123-45-6789",
    "person.gender": "Female",
    "person.marital_status": "Married",
    "person.citizenship": "United States",
    "person.nationality": "American",
    "person.place_of_birth": "Dayton, OH",
    "person.occupation": "Systems Analyst",
    "person.job_title": "Senior Analyst",
    "person.employer": "Northwind Traders",
    "person.email": "jordan.avery@example.com",
    "person.phone.mobile": "(555) 014-2280",
    "person.phone.home": "(555) 014-9931",
    "person.phone.work": "(555) 014-7712",
    "person.phone.fax": "(555) 014-7713",
    "person.address.street_1": "418 Kestrel Lane",
    "person.address.street_2": "Suite 210",
    "person.address.unit": "210",
    "person.address.city": "Fairview",
    "person.address.region": "OH",
    "person.address.postal_code": "45402",
    "person.address.country": "USA",
    "person.address.county": "Montgomery",
    "person.signature": "Jordan Q Avery",
    "person.emergency_contact.name": "Riley Avery",
    "person.driver_license.number": "OH-4471822",
    "person.passport.number": "X4471822",
    "company.legal_name": "Northwind Traders LLC",
    "company.dba_name": "Northwind Supply",
    "company.tax_id.ein": "31-4471822",
    "company.website": "www.example.com",
    "bank.name": "First Meridian Bank",
    "bank.account_number": "0041188207",
    "bank.routing_number": "021000021",
    "bank.account_type": "Checking",
    "card.number": "4111111111111111",
    "card.expiration": "09/29",
    "card.cvv": "417",
    "document.today": "03/14/2024",
    "document.signed_date": "03/14/2024",
    "document.effective_date": "04/01/2024",
    "document.expiration_date": "03/31/2025",
    "document.case_number": "CV-2024-0188",
    "document.invoice_number": "INV-100482",
    "document.reference_number": "REF-88120",
    "document.policy_number": "POL-4471822",
    "document.amount": "1,250.00",
    "document.total": "1,250.00",
    "document.department": "Operations",
    "document.payment_method": "Check",
    "employment.employee_id": "E-44718",
    "employment.start_date": "06/02/2019",
    "employment.supervisor": "Dana Whitfield",
    "employment.salary": "82,500",
    "employment.work_address": "1200 Industrial Parkway",
    "education.school_name": "Fairview State University",
    "education.degree": "B.S.",
    "education.field_of_study": "Information Systems",
    "education.gpa": "3.62",
    "education.graduation_date": "05/18/2006",
    "education.student_id": "S-9920417",
    "insurance.member_id": "M-44718220",
    "insurance.group_number": "GRP-8841",
    "insurance.subscriber_name": "Jordan Q Avery",
    "medical.blood_type": "O+",
    "medical.allergies": "Penicillin",
    "medical.medications": "None",
    "medical.physician_name": "Dr. Lin Osei",
    "military.status": "No",
    "misc.relationship": "Spouse",
    "misc.preferred_contact_method": "Email",
    "misc.comments": "None",
    "tax.gross_income": "82,500",
    "tax.filing_status": "Married Filing Jointly",
}


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSpec:
    """A printable label that is known to resolve to a canonical ontology key."""

    label: str
    key: str
    field_type: FieldType

    @property
    def value(self) -> str:
        """The deterministic sample value for this key (``""`` when none is defined)."""
        return SAMPLE_VALUES.get(self.key, "")


@dataclass(frozen=True)
class FieldMark:
    """One ground-truth field as the layout knows it, before the document names it."""

    label: str
    canonical_key: str
    field_type: FieldType
    rect: Rect
    expected_value: str = ""
    option: str = ""


@dataclass
class PageDraw:
    """Everything one page produced: its operators, its fonts and its ground truth."""

    content: bytes
    fonts: Dict[str, str] = field(default_factory=dict)
    marks: List[FieldMark] = field(default_factory=list)


_SPEC_CACHE: Dict[str, Optional[LabelSpec]] = {}


def spec_for(label: str) -> Optional[LabelSpec]:
    """Resolve one printed label, or ``None`` when the ontology does not know it."""
    if label in _SPEC_CACHE:
        return _SPEC_CACHE[label]
    key = ontology_lookup(label)
    spec: Optional[LabelSpec] = None
    if key is not None:
        declared = ontology_get(key)
        ftype = declared.field_type if declared is not None else FieldType.TEXT
        spec = LabelSpec(label=label, key=key, field_type=ftype)
    _SPEC_CACHE[label] = spec
    return spec


def specs_for(labels: Sequence[str]) -> List[LabelSpec]:
    """Resolve a sequence of labels, silently dropping the ones with no canonical key."""
    out: List[LabelSpec] = []
    for label in labels:
        spec = spec_for(label)
        if spec is not None:
            out.append(spec)
    return out


_SECTION_CACHE: Dict[str, List[LabelSpec]] = {}


def section_specs(heading: str) -> List[LabelSpec]:
    """Return the resolved specs of the named section."""
    cached = _SECTION_CACHE.get(heading)
    if cached is None:
        for name, labels in SECTIONS:
            if name == heading:
                cached = specs_for(labels)
                break
        else:
            raise ValidationError("unknown section %r" % (heading,))
        _SECTION_CACHE[heading] = cached
    return cached


# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """The randomized-but-seeded look of one synthetic document."""

    base_font: str = "Helvetica"
    bold_font: str = "Helvetica-Bold"
    font_size: float = 10.0
    line_width: float = 0.6
    row_gap: float = 10.0
    margin: float = 54.0
    page_width: float = 612.0
    page_height: float = 792.0
    checkbox_side: float = 10.0
    comb_cell_width: float = 15.0
    comb_style: str = "boxes"          # "boxes" | "separators"
    box_label: str = "left"            # "left" | "above"

    @property
    def leading(self) -> float:
        """Baseline-to-baseline distance for body copy."""
        return self.font_size * LEADING_FACTOR


def build_style(
    rng: random.Random,
    *,
    font: str = "Helvetica",
    font_size: float = 10.0,
    line_width: float = 0.6,
    page_width: float = 612.0,
    page_height: float = 792.0,
    randomize_font: bool = True,
    randomize_size: bool = True,
    randomize_width: bool = True,
) -> Style:
    """Compose a :class:`Style` from ``rng`` plus the caller's explicit choices.

    A value the caller left at its default is drawn from ``rng`` (font family from the
    three base-14 text families, size 8-12, stroke 0.4-1.2); a value the caller actually
    set is honoured verbatim.  Either way the result depends only on the seed.
    """
    # Every draw happens unconditionally so that the random stream -- and therefore the
    # rest of the document -- does not shift when a caller pins one attribute.
    drawn_font = rng.choice(_TEXT_FAMILIES)
    drawn_size = rng.choice((8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0))
    drawn_width = round(rng.uniform(0.4, 1.2), 2)

    base = drawn_font if randomize_font else resolve_base_font(font)
    bold = _BOLD_OF.get(base, "Helvetica-Bold")
    size = min(12.0, max(8.0, drawn_size if randomize_size else float(font_size)))
    width = min(1.2, max(0.4, drawn_width if randomize_width else float(line_width)))

    return Style(
        base_font=base,
        bold_font=bold,
        font_size=size,
        line_width=width,
        row_gap=rng.choice((8.0, 10.0, 12.0, 14.0, 16.0)),
        margin=rng.choice((48.0, 54.0, 60.0, 66.0)),
        page_width=float(page_width),
        page_height=float(page_height),
        checkbox_side=rng.choice((8.0, 9.0, 10.0, 11.0, 12.0)),
        comb_cell_width=rng.choice((12.0, 14.0, 16.0, 18.0)),
        comb_style=rng.choice(("boxes", "separators")),
        box_label=rng.choice(("left", "above")),
    )


# --------------------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------------------


class Canvas:
    """A page under construction: a painter, a downward cursor and the ground truth."""

    def __init__(self, style: Style, rng: random.Random, page_index: int) -> None:
        self.style = style
        self.rng = rng
        self.page_index = int(page_index)
        self.b = ContentBuilder()
        self.marks: List[FieldMark] = []
        self.y: float = style.page_height - style.margin
        self._res: Dict[str, str] = {}

    # -- frame -------------------------------------------------------------------------
    @property
    def left(self) -> float:
        return self.style.margin

    @property
    def right(self) -> float:
        return self.style.page_width - self.style.margin

    @property
    def bottom(self) -> float:
        return self.style.margin

    @property
    def content_width(self) -> float:
        return self.right - self.left

    def room(self, needed: float) -> bool:
        """True when ``needed`` points of vertical space remain above the bottom margin."""
        return (self.y - needed) >= self.bottom

    def full(self, extra: int = 1) -> bool:
        """True when ``extra`` more fields would exceed the per-page cap."""
        return (len(self.marks) + extra) > MAX_FIELDS_PER_PAGE

    # -- text --------------------------------------------------------------------------
    def _font_res(self, base_font: str) -> str:
        name = self._res.get(base_font)
        if name is None:
            name = "F%d" % (len(self._res) + 1)
            self._res[base_font] = name
        return name

    def measure(self, s: str, *, bold: bool = False, size: Optional[float] = None) -> float:
        """Advance width of ``s`` in the style's body (or bold) face."""
        face = self.style.bold_font if bold else self.style.base_font
        return text_width(s, face, self.style.font_size if size is None else float(size))

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        bold: bool = False,
        size: Optional[float] = None,
    ) -> float:
        """Draw ``s`` with its baseline at ``(x, y)`` and return its advance width."""
        face = self.style.bold_font if bold else self.style.base_font
        pt = self.style.font_size if size is None else float(size)
        self.b.text(x, y, s, font_res=self._font_res(face), size=pt, base_font=face)
        return text_width(s, face, pt)

    # -- ground truth ------------------------------------------------------------------
    def mark(
        self,
        spec: LabelSpec,
        rect: Rect,
        *,
        field_type: Optional[FieldType] = None,
        option: str = "",
        value: Optional[str] = None,
    ) -> None:
        """Record one ground-truth field."""
        expected = spec.value if value is None else value
        self.marks.append(
            FieldMark(
                label=spec.label,
                canonical_key=spec.key,
                field_type=field_type or spec.field_type,
                rect=rect.normalized().rounded(4),
                expected_value=expected,
                option=option,
            )
        )

    # -- furniture ---------------------------------------------------------------------
    def title(self, text: str) -> None:
        """Draw the document title with a rule under it."""
        size = self.style.font_size + 4.0
        baseline = self.y - size
        self.text(self.left, baseline, text, bold=True, size=size)
        rule = baseline - 5.0
        self.b.line(self.left, rule, self.right, rule, max(self.style.line_width, 0.8))
        self.y = rule - 14.0

    def running_head(self, text: str, page_index: int) -> None:
        """Draw the continuation header used on pages after the first."""
        size = self.style.font_size + 1.0
        baseline = self.y - size
        self.text(self.left, baseline, text, bold=True, size=size)
        caption = "Page %d" % (page_index + 1)
        self.text(self.right - self.measure(caption, size=size), baseline, caption, size=size)
        rule = baseline - 4.0
        self.b.line(self.left, rule, self.right, rule, self.style.line_width)
        self.y = rule - 12.0

    def heading(self, text: str) -> None:
        """Draw a section heading."""
        size = self.style.font_size + 1.0
        baseline = self.y - size
        self.text(self.left, baseline, text, bold=True, size=size)
        self.y = baseline - 8.0

    def paragraph(self, text: str) -> None:
        """Draw wrapped body copy across the full content width."""
        size = self.style.font_size
        lines = wrap_text(text, self.style.base_font, size, self.content_width)
        leading = size * LEADING_FACTOR
        for line in lines:
            baseline = self.y - size
            self.text(self.left, baseline, line)
            self.y = baseline - (leading - size)
        self.y -= 8.0


# --------------------------------------------------------------------------------------
# Row emitters
# --------------------------------------------------------------------------------------


def _columns(cv: Canvas, count: int) -> Tuple[float, List[float]]:
    """Return ``(column_width, x positions)`` for a ``count``-column row."""
    count = max(1, int(count))
    width = (cv.content_width - COL_GAP * (count - 1)) / count
    return width, [cv.left + i * (width + COL_GAP) for i in range(count)]


def _row_fits(cv: Canvas, group: Sequence[LabelSpec], min_field: float) -> bool:
    """True when every label in ``group`` leaves at least ``min_field`` for its input."""
    width, _ = _columns(cv, len(group))
    return all(
        cv.measure(spec.label + ":") + LABEL_GAP + min_field <= width for spec in group
    )


def pack_rows(
    cv: Canvas,
    specs: Sequence[LabelSpec],
    max_cols: int,
    min_field: float,
) -> List[List[LabelSpec]]:
    """Greedily group ``specs`` into rows of at most ``max_cols`` columns that fit."""
    rows: List[List[LabelSpec]] = []
    index = 0
    total = len(specs)
    while index < total:
        placed = False
        for count in range(min(max_cols, total - index), 1, -1):
            group = list(specs[index : index + count])
            if _row_fits(cv, group, min_field):
                rows.append(group)
                index += count
                placed = True
                break
        if not placed:
            rows.append([specs[index]])
            index += 1
    return rows


def row_underline(cv: Canvas, group: Sequence[LabelSpec]) -> bool:
    """``Label: ______`` -- label left, rule right.  Returns False when out of space."""
    st = cv.style
    needed = st.font_size + RULE_DROP + st.row_gap
    if not cv.room(needed) or cv.full(len(group)):
        return False
    width, xs = _columns(cv, len(group))
    baseline = cv.y - st.font_size
    rule_y = baseline - RULE_DROP
    for spec, x in zip(group, xs):
        label_w = cv.text(x, baseline, spec.label + ":")
        x0 = x + label_w + LABEL_GAP
        x1 = x + width
        if x1 - x0 < DET.min_line_length_pt:
            x0 = x1 - DET.min_line_length_pt
        cv.b.line(x0, rule_y, x1, rule_y, st.line_width)
        y0 = rule_y + DET.underline_gap_pt
        cv.mark(spec, Rect(x0, y0, x1, y0 + DET.field_height_pt))
    cv.y = rule_y - st.row_gap
    return True


def row_boxed(cv: Canvas, group: Sequence[LabelSpec], label_above: bool) -> bool:
    """A stroked rectangle as the input area; ground truth is its inside."""
    st = cv.style
    box_h = max(DET.field_height_pt + 4.0, st.font_size * 1.5)
    lw = st.line_width
    needed = box_h + st.row_gap + (st.font_size + 4.0 if label_above else 0.0)
    if not cv.room(needed) or cv.full(len(group)):
        return False
    width, xs = _columns(cv, len(group))

    if label_above:
        baseline = cv.y - st.font_size
        top = baseline - 4.0
        for spec, x in zip(group, xs):
            cv.text(x, baseline, spec.label + ":")
            cv.b.rect(x, top - box_h, width, box_h, lw)
            cv.mark(spec, Rect(x, top - box_h, x + width, top).inflated(-lw))
        cv.y = top - box_h - st.row_gap
        return True

    top = cv.y
    bottom = top - box_h
    baseline = bottom + (box_h - st.font_size * 0.72) / 2.0
    for spec, x in zip(group, xs):
        label_w = cv.text(x, baseline, spec.label + ":")
        x0 = x + label_w + LABEL_GAP
        x1 = x + width
        if x1 - x0 < DET.blank_min_width_pt:
            x0 = x1 - DET.blank_min_width_pt
        cv.b.rect(x0, bottom, x1 - x0, box_h, lw)
        cv.mark(spec, Rect(x0, bottom, x1, top).inflated(-lw))
    cv.y = bottom - st.row_gap
    return True


def row_borderless(cv: Canvas, group: Sequence[LabelSpec]) -> bool:
    """Label on its own line and a blank region under it -- no ink at all in the field."""
    st = cv.style
    region_h = max(DET.blank_min_height_pt + 5.0, st.font_size * 1.5)
    needed = st.font_size + 4.0 + region_h + st.row_gap
    if not cv.room(needed) or cv.full(len(group)):
        return False
    width, xs = _columns(cv, len(group))
    baseline = cv.y - st.font_size
    top = baseline - 4.0
    for spec, x in zip(group, xs):
        cv.text(x, baseline, spec.label + ":")
        cv.mark(spec, Rect(x + 2.0, top - region_h, x + width, top))
    cv.y = top - region_h - st.row_gap
    return True


def group_checkbox(
    cv: Canvas,
    stem: LabelSpec,
    options: Sequence[str],
    radio: bool,
) -> bool:
    """``Stem:  [ ] A  [ ] B`` -- squares for checkboxes, circles for radio groups."""
    st = cv.style
    side = st.checkbox_side
    needed = st.font_size + side + 8.0 + st.row_gap
    if not cv.room(needed) or cv.full(len(options)):
        return False
    baseline = cv.y - st.font_size
    x = cv.left + cv.text(cv.left, baseline, stem.label + ":") + 12.0
    ftype = FieldType.RADIO if radio else FieldType.CHECKBOX
    for option in options:
        advance = side + 4.0 + cv.measure(option) + 16.0
        if x + advance > cv.right and x > cv.left + 12.0:
            baseline -= st.font_size + 8.0
            x = cv.left + 12.0
        y0 = baseline - 2.0
        if radio:
            cv.b.circle(x + side / 2.0, y0 + side / 2.0, side / 2.0, st.line_width)
        else:
            cv.b.rect(x, y0, side, side, st.line_width)
        cv.text(x + side + 4.0, baseline, option)
        cv.mark(
            stem,
            Rect(x, y0, x + side, y0 + side),
            field_type=ftype,
            option=option,
            value=option,
        )
        x += advance
    cv.y = baseline - 6.0 - st.row_gap
    return True


def block_table(cv: Canvas, columns: Sequence[LabelSpec], rows: int) -> bool:
    """A ruled grid: the header row is furniture, every empty data cell is a field."""
    st = cv.style
    lw = st.line_width
    row_h = max(st.font_size * 1.9, 16.0)
    total_h = row_h * (rows + 1)
    if not cv.room(total_h + st.row_gap + 4.0) or cv.full(rows * len(columns)):
        return False
    col_w = cv.content_width / len(columns)
    top = cv.y
    bottom = top - total_h

    for i in range(rows + 2):
        y = top - i * row_h
        cv.b.line(cv.left, y, cv.right, y, lw)
    for j in range(len(columns) + 1):
        x = cv.left + j * col_w
        cv.b.line(x, bottom, x, top, lw)

    caption_size = max(7.0, st.font_size - 0.5)
    caption_baseline = top - row_h + (row_h - caption_size * 0.72) / 2.0
    for j, spec in enumerate(columns):
        cv.text(cv.left + j * col_w + 4.0, caption_baseline, spec.label, bold=True, size=caption_size)

    for r in range(rows):
        y1 = top - row_h * (r + 1)
        y0 = y1 - row_h
        for j, spec in enumerate(columns):
            x0 = cv.left + j * col_w
            cv.mark(spec, Rect(x0, y0, x0 + col_w, y1).inflated(-lw), option="row%d" % (r + 1))
    cv.y = bottom - st.row_gap - 4.0
    return True


def row_comb(cv: Canvas, spec: LabelSpec, cells: int) -> bool:
    """One value spread over ``cells`` equal adjacent boxes; the field is their union."""
    st = cv.style
    lw = st.line_width
    cells = max(2, int(cells))
    height = max(DET.field_height_pt + 3.0, st.font_size * 1.55)
    if not cv.room(height + st.row_gap) or cv.full(1):
        return False
    top = cv.y
    bottom = top - height
    baseline = bottom + (height - st.font_size * 0.72) / 2.0
    x = cv.left + cv.text(cv.left, baseline, spec.label + ":") + LABEL_GAP
    available = cv.right - x

    gap = 2.5 if st.comb_style == "boxes" else 0.0
    cell_w = st.comb_cell_width
    span = cells * cell_w + gap * (cells - 1)
    if span > available:
        cell_w = max(8.0, (available - gap * (cells - 1)) / cells)
        span = cells * cell_w + gap * (cells - 1)

    if st.comb_style == "boxes":
        for i in range(cells):
            cv.b.rect(x + i * (cell_w + gap), bottom, cell_w, height, lw)
    else:
        cv.b.rect(x, bottom, span, height, lw)
        for i in range(1, cells):
            sx = x + i * cell_w
            cv.b.line(sx, bottom, sx, top, lw)

    cv.mark(spec, Rect(x, bottom, x + span, top).inflated(-lw), field_type=FieldType.COMB)
    cv.y = bottom - st.row_gap
    return True


def block_signature(cv: Canvas, specs: Sequence[LabelSpec]) -> bool:
    """Rules with their captions *underneath* -- the signature-block idiom."""
    st = cv.style
    drawn = False
    index = 0
    while index < len(specs):
        group = list(specs[index : index + 2])
        needed = 14.0 + st.font_size + 6.0 + st.row_gap
        if not cv.room(needed) or cv.full(len(group)):
            break
        width, xs = _columns(cv, len(group))
        rule_y = cv.y - 14.0
        for spec, x in zip(group, xs):
            x1 = x + width
            cv.b.line(x, rule_y, x1, rule_y, st.line_width)
            cv.text(x, rule_y - st.font_size - 2.0, spec.label)
            y0 = rule_y + DET.underline_gap_pt
            ftype = FieldType.SIGNATURE if spec.key == "person.signature" else spec.field_type
            cv.mark(spec, Rect(x, y0, x1, y0 + DET.field_height_pt), field_type=ftype)
        cv.y = rule_y - st.font_size - 6.0 - st.row_gap
        index += 2
        drawn = True
    return drawn


# --------------------------------------------------------------------------------------
# Section composition
# --------------------------------------------------------------------------------------


def _sample(rng: random.Random, population: Sequence, count: int) -> List:
    """``rng.sample`` that never asks for more than the population holds."""
    count = max(0, min(int(count), len(population)))
    return list(rng.sample(list(population), count))


def _pick_sections(rng: random.Random, count: int) -> List[Tuple[str, List[LabelSpec]]]:
    """Choose ``count`` distinct sections and resolve their labels."""
    names = [name for name, _ in SECTIONS]
    chosen = _sample(rng, names, count)
    return [(name, list(section_specs(name))) for name in chosen]


def _emit_section(
    cv: Canvas,
    heading: str,
    specs: Sequence[LabelSpec],
    style_name: str,
    include_sections: bool,
) -> None:
    """Draw one heading plus its fields in the requested field style."""
    if not specs:
        return
    if include_sections:
        if not cv.room(cv.style.font_size + 12.0):
            return
        cv.heading(heading)
    if style_name == "underline":
        for group in pack_rows(cv, specs, 3, MIN_RULE):
            if not row_underline(cv, group):
                break
    elif style_name == "boxed":
        above = cv.style.box_label == "above"
        max_cols = 3 if above else 2
        min_field = 60.0 if above else DET.blank_min_width_pt
        for group in pack_rows(cv, specs, max_cols, min_field):
            if not row_boxed(cv, group, above):
                break
    elif style_name == "borderless":
        for group in pack_rows(cv, specs, 2, DET.blank_min_width_pt):
            if not row_borderless(cv, group):
                break
    else:  # pragma: no cover - defensive
        raise ValidationError("unknown field style %r" % (style_name,))


def _emit_checkbox_section(cv: Canvas, rng: random.Random, groups: int, include_sections: bool) -> None:
    """Draw ``groups`` checkbox/radio groups under one heading."""
    if include_sections:
        cv.heading("Questionnaire")
    chosen = _sample(rng, CHECKBOX_GROUPS, groups)
    for stem_label, options in chosen:
        stem = spec_for(stem_label)
        if stem is None:
            continue
        radio = rng.random() < 0.5
        if not group_checkbox(cv, stem, options, radio):
            break


def _emit_comb_section(cv: Canvas, rng: random.Random, count: int, include_sections: bool) -> None:
    """Draw ``count`` comb rows under one heading."""
    if include_sections:
        cv.heading("Identification Numbers")
    for label, cells in _sample(rng, COMB_FIELDS, count):
        spec = spec_for(label)
        if spec is None:
            continue
        if not row_comb(cv, spec, cells):
            break


def _emit_table_section(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    """Draw one ruled table under one heading."""
    if include_sections:
        cv.heading("Schedule of Dependents")
    ncols = rng.choice((3, 4))
    columns = specs_for(_sample(rng, TABLE_COLUMNS, ncols))
    if len(columns) < 2:
        return
    block_table(cv, columns, rng.randint(3, 5))


def _emit_signature_section(cv: Canvas, include_sections: bool) -> None:
    """Draw the certification paragraph and the signature rules."""
    if include_sections:
        cv.heading("Certification")
    cv.paragraph(_CERTIFICATION_TEXT)
    block_signature(cv, specs_for(SIGNATURE_LABELS))


def _top_up(cv: Canvas, rng: random.Random) -> None:
    """Add plain underline rows until the page carries the documented minimum."""
    pool: List[LabelSpec] = []
    for _, labels in SECTIONS:
        pool.extend(specs_for(labels))
    used = {mark.canonical_key for mark in cv.marks}
    remaining = [spec for spec in pool if spec.key not in used]
    index = 0
    while len(cv.marks) < MIN_FIELDS_PER_PAGE and index < len(remaining):
        group = remaining[index : index + 2]
        index += 2
        if not row_underline(cv, group):
            break


# --------------------------------------------------------------------------------------
# Page templates
# --------------------------------------------------------------------------------------


def _page_underline(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    for heading, specs in _pick_sections(rng, rng.randint(2, 3)):
        _emit_section(cv, heading, specs, "underline", include_sections)


def _page_boxed(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    for heading, specs in _pick_sections(rng, rng.randint(2, 3)):
        _emit_section(cv, heading, specs, "boxed", include_sections)


def _page_borderless(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    for heading, specs in _pick_sections(rng, rng.randint(2, 3)):
        _emit_section(cv, heading, specs, "borderless", include_sections)


def _page_checkbox(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    _emit_checkbox_section(cv, rng, len(CHECKBOX_GROUPS), include_sections)


def _page_table(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    _emit_table_section(cv, rng, include_sections)
    if len(cv.marks) < MAX_FIELDS_PER_PAGE:
        _emit_table_section(cv, rng, include_sections)


def _page_comb(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    _emit_comb_section(cv, rng, len(COMB_FIELDS), include_sections)


def _page_mixed(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    styles = _sample(rng, ("underline", "boxed", "borderless", "checkbox", "comb", "table"), 3)
    sections = _pick_sections(rng, 3)
    for (heading, specs), style_name in zip(sections, styles):
        if style_name == "checkbox":
            _emit_checkbox_section(cv, rng, rng.randint(2, 3), include_sections)
        elif style_name == "comb":
            _emit_comb_section(cv, rng, rng.randint(3, 5), include_sections)
        elif style_name == "table":
            _emit_table_section(cv, rng, include_sections)
        else:
            _emit_section(cv, heading, specs[: rng.randint(4, 6)], style_name, include_sections)


def _page_signature(cv: Canvas, rng: random.Random, include_sections: bool) -> None:
    cv.paragraph(_INTRO_TEXT)
    heading, specs = _pick_sections(rng, 1)[0]
    _emit_section(cv, heading, specs[: rng.randint(4, 6)], "underline", include_sections)
    _emit_signature_section(cv, include_sections)


_TEMPLATES = {
    "underline": _page_underline,
    "boxed": _page_boxed,
    "checkbox": _page_checkbox,
    "table": _page_table,
    "comb": _page_comb,
    "borderless": _page_borderless,
    "mixed": _page_mixed,
    "signature": _page_signature,
}


def draw_page(
    kind: str,
    rng: random.Random,
    style: Style,
    page_index: int,
    *,
    title: str = "",
    include_sections: bool = True,
    first_page: bool = True,
    repeat: Sequence[LabelSpec] = (),
) -> PageDraw:
    """Draw one page of the named kind and return its content plus its ground truth.

    Args:
        kind: One of :data:`PAGE_KINDS`.
        rng: The document's seeded generator; consumed in a fixed order.
        style: The document's :class:`Style`.
        page_index: Zero-based page index, used for the running head.
        title: Document title.  Drawn in full on the first page, as a running head
            afterwards.
        include_sections: Draw section headings.
        first_page: Whether this is the first sheet.
        repeat: Specs to draw as a repeated header block on this page.

    Returns:
        A :class:`PageDraw`.

    Raises:
        ValidationError: ``kind`` is not a known page template.
    """
    template = _TEMPLATES.get(str(kind))
    if template is None:
        raise ValidationError(
            "unknown page kind %r; expected one of %s" % (kind, ", ".join(PAGE_KINDS))
        )
    cv = Canvas(style, rng, page_index)
    if title:
        if first_page:
            cv.title(title)
        else:
            cv.running_head(title, page_index)
    if repeat:
        for group in pack_rows(cv, list(repeat), 2, MIN_RULE):
            if not row_underline(cv, group):
                break
    template(cv, rng, include_sections)
    _top_up(cv, rng)
    return PageDraw(content=cv.b.build(), fonts=dict(cv.b.fonts_used), marks=list(cv.marks))
