"""Document triage -- the layer that decides which perception system a page needs.

``zfp.preflight`` is the cheapest and earliest stage of the pipeline, and the one that
keeps ZFP honest about the file it was given: a page is rasterized only once this layer
has established there is nothing left to read from the PDF itself.

* :mod:`zfp.preflight.classifier` -- per-page and per-document classification and
  routing (:func:`classify_page`, :func:`profile_document`, :func:`route`).
* :mod:`zfp.preflight.security` -- encryption permissions, signatures, and the
  :func:`can_add_form_fields` gate every writer must clear.
"""

from __future__ import annotations

from .classifier import (
    ContentScan,
    classify_page,
    describe,
    profile_document,
    profile_to_dict,
    route,
    scan_content,
)
from .security import (
    SecurityState,
    SignatureState,
    can_add_form_fields,
    has_permission,
    inspect,
    inspect_security,
    inspect_signatures,
)

__all__ = [
    "ContentScan",
    "classify_page",
    "describe",
    "profile_document",
    "profile_to_dict",
    "route",
    "scan_content",
    "SecurityState",
    "SignatureState",
    "can_add_form_fields",
    "has_permission",
    "inspect",
    "inspect_security",
    "inspect_signatures",
]
