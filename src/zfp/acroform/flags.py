"""``/Ff`` field-flag and ``/F`` annotation-flag constants.

Named here so no other module in :mod:`zfp.acroform` spells out a bit position, per the
PDF specification's Table 221 (field flags) and Table 165 (annotation flags).
"""

from __future__ import annotations

from ..core.types import FieldSpec, FieldType

# -- /Ff : common to all field types -----------------------------------------------------
READ_ONLY = 1 << 0
REQUIRED = 1 << 1
NO_EXPORT = 1 << 2

# -- /Ff : text fields (/FT /Tx) ----------------------------------------------------------
MULTILINE = 1 << 12
PASSWORD = 1 << 13
FILE_SELECT = 1 << 20
DO_NOT_SPELL_CHECK = 1 << 22
DO_NOT_SCROLL = 1 << 23
COMB = 1 << 24
RICH_TEXT = 1 << 25

# -- /Ff : button fields (/FT /Btn) --------------------------------------------------------
NO_TOGGLE_TO_OFF = 1 << 14
RADIO = 1 << 15
PUSHBUTTON = 1 << 16
RADIOS_IN_UNISON = 1 << 25

# -- /Ff : choice fields (/FT /Ch) ---------------------------------------------------------
COMBO = 1 << 17
EDIT = 1 << 18
SORT = 1 << 19
MULTI_SELECT = 1 << 21
COMMIT_ON_SEL_CHANGE = 1 << 26

# -- /F : annotation flags ------------------------------------------------------------------
ANNOT_INVISIBLE = 1 << 0
ANNOT_HIDDEN = 1 << 1
ANNOT_PRINT = 1 << 2
ANNOT_NOZOOM = 1 << 3
ANNOT_NOROTATE = 1 << 4
ANNOT_NOVIEW = 1 << 5
ANNOT_LOCKED = 1 << 7
ANNOT_TOGGLENOVIEW = 1 << 8
ANNOT_LOCKEDCONTENTS = 1 << 9


def field_flags(spec: FieldSpec) -> int:
    """The ``/Ff`` value for ``spec``, from its type and boolean attributes."""
    flags = 0
    if spec.read_only:
        flags |= READ_ONLY
    if spec.required:
        flags |= REQUIRED

    ft = spec.field_type
    if ft in (FieldType.MULTILINE_TEXT,) or (ft is FieldType.TEXT and spec.multiline):
        flags |= MULTILINE
    if ft is FieldType.COMB:
        flags |= COMB
    if ft is FieldType.RADIO:
        flags |= RADIO | NO_TOGGLE_TO_OFF
    if ft is FieldType.BUTTON:
        flags |= PUSHBUTTON
    if ft is FieldType.CHOICE:
        flags |= COMBO
    if ft is FieldType.LISTBOX and spec.choices and len(spec.choices) > 1 and spec.group:
        flags |= MULTI_SELECT
    return flags


def annotation_flags(spec: FieldSpec) -> int:
    """The ``/F`` value for a widget annotation: printable, and hidden if read-only+empty."""
    return ANNOT_PRINT


__all__ = [
    "READ_ONLY", "REQUIRED", "NO_EXPORT", "MULTILINE", "PASSWORD", "FILE_SELECT",
    "DO_NOT_SPELL_CHECK", "DO_NOT_SCROLL", "COMB", "RICH_TEXT",
    "NO_TOGGLE_TO_OFF", "RADIO", "PUSHBUTTON", "RADIOS_IN_UNISON",
    "COMBO", "EDIT", "SORT", "MULTI_SELECT", "COMMIT_ON_SEL_CHANGE",
    "ANNOT_INVISIBLE", "ANNOT_HIDDEN", "ANNOT_PRINT", "ANNOT_NOZOOM", "ANNOT_NOROTATE",
    "ANNOT_NOVIEW", "ANNOT_LOCKED", "ANNOT_TOGGLENOVIEW", "ANNOT_LOCKEDCONTENTS",
    "field_flags", "annotation_flags",
]
