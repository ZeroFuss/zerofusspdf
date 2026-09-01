"""Page rasterization and scan preprocessing.

Three modules, all pure CPython:

* :mod:`zfp.raster.render` turns a page into an 8-bit gray raster, preferring an
  installed renderer and falling back to compositing the page's own image XObjects.
* :mod:`zfp.raster.image` decodes those image XObjects -- Flate/LZW/RunLength raw
  samples, CCITT Group 3/4 fax, and baseline JPEG -- without any third-party library.
* :mod:`zfp.raster.preprocess` cleans a scan up: contrast, binarization, denoising,
  deskew and orientation.

Pixel space never leaves this package: convert with
:class:`~zfp.core.geometry.PageGeometry` and :attr:`RenderedPage.scale` first.
"""

from __future__ import annotations

from .image import DecodedImage, ccitt_decode, decode_image_xobject, decode_jpeg_gray
from .preprocess import (
    PreprocessReport,
    binarize,
    denoise,
    deskew,
    detect_orientation,
    estimate_skew,
    normalize_contrast,
    otsu_threshold,
    preprocess,
    rotate_quarter,
)
from .render import (
    AGPL_ENV_VAR,
    BACKEND_EMBEDDED,
    BACKEND_PDFTOPPM,
    BACKEND_PYMUPDF,
    BACKEND_PYPDFIUM2,
    RenderedPage,
    available_backends,
    embedded_page_images,
    parse_pgm,
    render_available,
    render_page,
)

__all__ = [
    "AGPL_ENV_VAR",
    "BACKEND_EMBEDDED",
    "BACKEND_PDFTOPPM",
    "BACKEND_PYMUPDF",
    "BACKEND_PYPDFIUM2",
    "DecodedImage",
    "PreprocessReport",
    "RenderedPage",
    "available_backends",
    "binarize",
    "ccitt_decode",
    "decode_image_xobject",
    "decode_jpeg_gray",
    "denoise",
    "deskew",
    "detect_orientation",
    "embedded_page_images",
    "estimate_skew",
    "normalize_contrast",
    "otsu_threshold",
    "parse_pgm",
    "preprocess",
    "render_available",
    "render_page",
    "rotate_quarter",
]
