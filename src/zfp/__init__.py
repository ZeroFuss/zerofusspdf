"""Zero PDF (ZFP).

A geometry-first PDF understanding engine that converts visual intent into native PDF
interactivity, resolves the semantics of each field against an authorized user-data graph,
fills everything it can deterministically, uses AI only to resolve ambiguity, and proves
afterward that nothing outside the intended regions changed.

Clean-room implementation. See ``docs/CLEANROOM.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
