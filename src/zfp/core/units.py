"""Unit conversions between PDF points, pixels, and millimetres."""

from __future__ import annotations

__all__ = [
    "PT_PER_INCH",
    "MM_PER_INCH",
    "pt_to_px",
    "px_to_pt",
    "mm_to_pt",
    "pt_to_mm",
    "inch_to_pt",
    "pt_to_inch",
    "dpi_to_scale",
    "scale_to_dpi",
]

PT_PER_INCH: float = 72.0
MM_PER_INCH: float = 25.4


def pt_to_px(pt: float, dpi: float) -> float:
    """Convert points to pixels at ``dpi``."""
    return float(pt) * float(dpi) / PT_PER_INCH


def px_to_pt(px: float, dpi: float) -> float:
    """Convert pixels at ``dpi`` back to points."""
    if dpi == 0:
        return 0.0
    return float(px) * PT_PER_INCH / float(dpi)


def mm_to_pt(mm: float) -> float:
    """Convert millimetres to points."""
    return float(mm) * PT_PER_INCH / MM_PER_INCH


def pt_to_mm(pt: float) -> float:
    """Convert points to millimetres."""
    return float(pt) * MM_PER_INCH / PT_PER_INCH


def inch_to_pt(inch: float) -> float:
    """Convert inches to points."""
    return float(inch) * PT_PER_INCH


def pt_to_inch(pt: float) -> float:
    """Convert points to inches."""
    return float(pt) / PT_PER_INCH


def dpi_to_scale(dpi: float) -> float:
    """Return the raster scale factor for ``dpi`` (``dpi / 72``)."""
    return float(dpi) / PT_PER_INCH


def scale_to_dpi(scale: float) -> float:
    """Return the dpi matching a raster scale factor."""
    return float(scale) * PT_PER_INCH
