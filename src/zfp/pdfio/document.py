"""The high-level document facade: :class:`Document` and :class:`Page`.

Everything above ``zfp.pdfio`` talks to a PDF through these two classes rather than
through the raw object graph.  They add the three things the object layer deliberately
leaves out:

* **Inheritance.**  ``/Resources``, ``/MediaBox``, ``/CropBox`` and ``/Rotate`` may live
  anywhere on the page-tree spine; :meth:`Page.inherited` walks it so callers never have
  to, and :attr:`Page.geometry` turns the result into a :class:`PageGeometry`.
* **A write overlay.**  :meth:`Document.resolve` consults the pending
  :class:`~zfp.pdfio.writer.PdfWriter` updates *before* the parsed file, so an object
  added or replaced in this session is visible immediately -- reading back an AcroForm
  you just created works without a save/reload cycle.
* **Form introspection.**  :meth:`Document.existing_fields` reads a real AcroForm back
  out as :class:`~zfp.core.types.FieldSpec` records, which is what lets ZFP treat an
  already-interactive document as ground truth instead of re-detecting it.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.errors import PdfWriteError, ValidationError
from ..core.geometry import PageGeometry, Rect
from ..core.ids import stable_id
from ..core.logging import get_logger
from ..core.types import FieldSpec, FieldType
from .objects import PdfArray, PdfDict, PdfName, PdfNull, PdfRef, PdfStream, PdfString
from .writer import PdfWriter, build_document, format_number

__all__ = [
    "Page",
    "Document",
    "INHERITABLE_PAGE_KEYS",
    "INHERITABLE_FIELD_KEYS",
    "US_LETTER",
]

_log = get_logger(__name__)

#: Page attributes the specification declares inheritable through ``/Parent``.
INHERITABLE_PAGE_KEYS: Tuple[str, ...] = ("Resources", "MediaBox", "CropBox", "Rotate")

#: Field attributes a child inherits from its parent in the AcroForm field tree.
INHERITABLE_FIELD_KEYS: Tuple[str, ...] = (
    "FT",
    "Ff",
    "V",
    "DV",
    "MaxLen",
    "Opt",
    "TU",
    "Q",
    "DA",
)

#: The default media box when a document declares none at all.
US_LETTER = Rect(0.0, 0.0, 612.0, 792.0)

#: How far up a ``/Parent`` chain (page tree or field tree) we are willing to walk.
_MAX_DEPTH = 64

# ``/Ff`` bit positions (PDF 32000-1 tables 227/228/229/230).
FF_READ_ONLY = 1 << 0
FF_REQUIRED = 1 << 1
FF_MULTILINE = 1 << 12
FF_NO_TOGGLE_TO_OFF = 1 << 14
FF_RADIO = 1 << 15
FF_PUSHBUTTON = 1 << 16
FF_COMBO = 1 << 17
FF_COMB = 1 << 24

#: ``/DA`` font selector, e.g. ``/Helv 11 Tf``.  The last match in the string wins.
_DA_FONT_RE = re.compile(rb"/([^\s/\[\]()<>{}%]+)\s+([0-9]*\.?[0-9]+)\s+Tf")


def _as_sequence(value: Any) -> List[Any]:
    """Return ``value`` as a list; scalars are wrapped, ``null``/``None`` yields ``[]``."""
    if value is None or isinstance(value, PdfNull):
        return []
    if isinstance(value, (PdfArray, list, tuple)):
        return list(value)
    return [value]


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


class Page:
    """One page of a :class:`Document`.

    Attributes:
        document: The owning :class:`Document` (used as the object resolver).
        index: Zero-based page index in reading order.
        dict: The page dictionary itself.
        ref: The page's own indirect reference, when it has one.  Required before a
            page dictionary can be modified, since the writer keys updates by number.
    """

    __slots__ = ("document", "index", "dict", "ref", "_geometry")

    def __init__(
        self,
        document: "Document",
        index: int,
        page_dict: PdfDict,
        ref: Optional[PdfRef] = None,
    ) -> None:
        self.document = document
        self.index = int(index)
        self.dict: PdfDict = page_dict if isinstance(page_dict, PdfDict) else PdfDict(page_dict or {})
        self.ref: Optional[PdfRef] = ref
        self._geometry: Optional[PageGeometry] = None

    # -- inheritance ------------------------------------------------------------------
    def inherited(self, key: str) -> Any:
        """Return ``key`` from this page or the nearest ancestor that defines it.

        Walks the ``/Parent`` chain up to the page tree root.  Used for the inheritable
        attributes ``/Resources``, ``/MediaBox``, ``/CropBox`` and ``/Rotate``, but works
        for any key.  A cyclic or absurdly deep tree stops at ``_MAX_DEPTH`` instead of
        recursing forever.

        Args:
            key: The dictionary key, with or without a leading ``/``.

        Returns:
            The resolved value, or ``None`` when no ancestor defines it.
        """
        node: Any = self.dict
        seen: set = set()
        for _ in range(_MAX_DEPTH):
            if not isinstance(node, PdfDict):
                return None
            if key in node:
                value = self.document.resolve(node.get(key))
                if not isinstance(value, PdfNull):
                    return value
            parent = node.get("Parent")
            if isinstance(parent, PdfRef):
                if parent.num in seen:
                    return None
                seen.add(parent.num)
            elif parent is None or isinstance(parent, PdfNull):
                return None
            node = self.document.resolve(parent)
        return None

    def _rect_from(self, key: str) -> Optional[Rect]:
        """Read an inheritable rectangle attribute, resolving every element."""
        value = self.inherited(key)
        items = _as_sequence(value)
        if len(items) < 4:
            return None
        numbers: List[float] = []
        for item in items[:4]:
            item = self.document.resolve(item)
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            numbers.append(float(item))
        return Rect.from_list(numbers)

    @property
    def geometry(self) -> PageGeometry:
        """The page's :class:`PageGeometry`, computed once and cached.

        ``/MediaBox`` defaults to US Letter when absent or unusable.  ``/CropBox``
        defaults to the media box and is always intersected with it, because a crop box
        larger than the media box has no meaning.  ``/Rotate`` is normalized onto
        ``0/90/180/270``, handling negative and out-of-range multiples.
        """
        if self._geometry is None:
            media = self._rect_from("MediaBox") or US_LETTER
            if media.width <= 0.0 or media.height <= 0.0:
                media = US_LETTER
            crop = self._rect_from("CropBox")
            if crop is None:
                crop = media
            else:
                clipped = crop.intersection(media)
                crop = clipped if clipped is not None and clipped.width > 0.0 and clipped.height > 0.0 else media
            rotate = self.inherited("Rotate")
            rotation = 0
            if isinstance(rotate, (int, float)) and not isinstance(rotate, bool):
                rotation = int(rotate)
            self._geometry = PageGeometry(
                index=self.index, media_box=media, crop_box=crop, rotation=rotation
            )
        return self._geometry

    # -- content ----------------------------------------------------------------------
    def content_bytes(self) -> bytes:
        """Return every ``/Contents`` stream decoded and concatenated.

        ``/Contents`` may be a single stream or an array of streams that together form
        one logical content stream; the parts are joined with a newline so a token never
        straddles a boundary.
        """
        contents = self.document.resolve(self.dict.get("Contents"))
        parts: List[bytes] = []
        for item in _as_sequence(contents):
            stream = self.document.resolve(item)
            if isinstance(stream, PdfStream):
                try:
                    parts.append(stream.decoded(self.document))
                except Exception:  # pragma: no cover - filters are already lenient
                    _log.debug("page %d: a content stream failed to decode", self.index)
        return b"\n".join(parts)

    def resources(self) -> PdfDict:
        """Return the (possibly inherited) ``/Resources`` dictionary, never ``None``."""
        value = self.inherited("Resources")
        if isinstance(value, PdfDict):
            return value
        if isinstance(value, dict):
            return PdfDict(value)
        return PdfDict()

    # -- annotations ------------------------------------------------------------------
    def annotations(self) -> List[PdfDict]:
        """Return the resolved ``/Annots`` entries, skipping anything not a dictionary."""
        annots = self.document.resolve(self.dict.get("Annots"))
        out: List[PdfDict] = []
        for item in _as_sequence(annots):
            value = self.document.resolve(item)
            if isinstance(value, PdfDict):
                out.append(value)
        return out

    def annotation_refs(self) -> List[PdfRef]:
        """Return the ``/Annots`` entries that are indirect references, unresolved."""
        annots = self.document.resolve(self.dict.get("Annots"))
        return [item for item in _as_sequence(annots) if isinstance(item, PdfRef)]

    def touch(self) -> None:
        """Stage this page's dictionary for writing and drop the cached geometry.

        :attr:`dict` is handed out live, so a caller can edit the page directly --
        ``page.dict["Rotate"] = 90``, adding a font to ``/Resources``, and so on.  The
        writer only serializes object numbers it has been told about, though, and
        :attr:`geometry` is computed once and cached, so an in-place edit would
        otherwise be dropped by the next write *and* read back stale in the same
        session.  Call this after any direct mutation of :attr:`dict`.

        Raises:
            PdfWriteError: The page dictionary has no object number, so there is
                nothing the writer could update.
        """
        if self.ref is None:
            raise PdfWriteError(
                "page %d has no object number; its changes cannot be saved" % self.index
            )
        self._geometry = None
        self.document.writer.set_object(self.ref.num, self.dict)

    def add_annotation(self, ref: PdfRef) -> None:
        """Append ``ref`` to this page's ``/Annots`` and stage the change for writing.

        Handles all three shapes ``/Annots`` takes in the wild: absent, a direct array,
        or an indirect reference to an array.  Whichever object actually changed -- the
        annotation array or the page dictionary -- is registered with the writer, so the
        next :meth:`Document.to_bytes` persists it.

        Args:
            ref: Reference to the annotation dictionary to attach.

        Raises:
            PdfWriteError: ``ref`` is not a :class:`PdfRef`, or the page dictionary has
                no object number and therefore cannot be updated.
        """
        if not isinstance(ref, PdfRef):
            raise PdfWriteError("add_annotation requires a PdfRef, got %r" % (ref,))
        writer = self.document.writer
        raw = self.dict.get("Annots")

        if isinstance(raw, PdfRef):
            array = self.document.resolve(raw)
            if not isinstance(array, (PdfArray, list)):
                array = PdfArray()
            elif not isinstance(array, PdfArray):
                array = PdfArray(array)
            if ref not in array:
                array.append(ref)
            # Re-register under its own object number: the parser may hand back a fresh
            # object on every resolve, so mutating in place is not enough on its own.
            writer.set_object(raw.num, array)
            return

        array = raw if isinstance(raw, PdfArray) else PdfArray(_as_sequence(raw))
        if ref not in array:
            array.append(ref)
        self.dict["Annots"] = array
        if self.ref is None:
            raise PdfWriteError(
                "page %d has no object number; its /Annots change cannot be saved" % self.index
            )
        writer.set_object(self.ref.num, self.dict)

    def __repr__(self) -> str:
        box = self.geometry.media_box
        return "Page(index=%d, media_box=%r)" % (self.index, box.as_list())


# --------------------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------------------


class Document:
    """A parsed PDF plus the writer that stages edits to it.

    Attributes:
        file: The parsed :class:`~zfp.pdfio.parser.PdfFile`.
        writer: The :class:`~zfp.pdfio.writer.PdfWriter` collecting pending updates.
        document_id: Deterministic ``doc_<hex>`` id derived from the source bytes.
        path: Where the document was loaded from, when it came from disk.
        password: The password supplied to :meth:`open`, kept for the crypt layer.
    """

    def __init__(
        self,
        file: Any,
        *,
        source_bytes: Optional[bytes] = None,
        path: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.file = file
        self.writer = PdfWriter(file)
        self.path: Optional[str] = path
        self.password: Optional[str] = password
        data = source_bytes if source_bytes is not None else getattr(file, "data", b"")
        self.source_bytes: bytes = bytes(data or b"")
        # stable_id already joins prefix and digest with "_", so the prefix is "doc"
        # rather than "doc_" -- otherwise every id came out as "doc__<hex>".
        self.document_id: str = stable_id(
            hashlib.sha256(self.source_bytes).hexdigest(), prefix="doc"
        )
        self._pages: Optional[List[Page]] = None
        self.acroform_ref: Optional[PdfRef] = None

    # -- construction -----------------------------------------------------------------
    @staticmethod
    def open(
        source: "str | os.PathLike | bytes | bytearray",
        password: Optional[str] = None,
    ) -> "Document":
        """Open a PDF from a path, a :class:`os.PathLike`, or raw bytes.

        Args:
            source: File system path or the PDF bytes themselves.
            password: Optional password forwarded to the optional decryption layer.

        Returns:
            A ready-to-use :class:`Document`.

        Raises:
            PdfParseError: The bytes are not a usable PDF.
        """
        from .parser import PdfFile  # imported lazily to keep ``zfp.pdfio`` cheap

        path: Optional[str] = None
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            path = os.fspath(source)
            with open(path, "rb") as handle:
                data = handle.read()
        pdf = PdfFile.load(data)
        doc = Document(pdf, source_bytes=data, path=path, password=password)
        doc._apply_decryption(password)
        return doc

    def _apply_decryption(self, password: Optional[str]) -> None:
        """Unlock an encrypted document with ``password``, if one was supplied.

        The parser already tries the empty user password while loading, which covers the
        common "encrypted but not actually locked" case; this only adds an explicit
        password on top.  A failure is *not* fatal: the document still opens so preflight
        can report ``encrypted=True`` and ``can_modify=False`` rather than crashing, and
        the write path refuses on its own terms.
        """
        if not password or not getattr(self.file, "is_encrypted", False):
            return
        authenticate = getattr(self.file, "authenticate", None)
        if not callable(authenticate):
            _log.debug("%s: parser exposes no authenticate()", self.document_id)
            return
        try:
            if not authenticate(password):
                _log.warning("%s: the supplied password was rejected", self.document_id)
        except Exception as exc:
            _log.warning("%s: authentication failed: %s", self.document_id, exc)

    @staticmethod
    def from_pages_blank(count: int, width: float = 612.0, height: float = 792.0) -> "Document":
        """Build a brand new document of ``count`` empty pages, in memory.

        The result is a real, parseable PDF produced by ZFP's own serializer: a catalog,
        a page tree, and one page plus one (empty) content stream per page.  The media
        box is written on both the page tree node and every page, so a consumer can read
        it either directly or through inheritance.

        Args:
            count: Number of pages; must be at least 1.
            width: Page width in points.
            height: Page height in points.

        Returns:
            A :class:`Document` backed by the freshly generated bytes.

        Raises:
            ValidationError: ``count`` is below 1 or the page size is not positive.
        """
        count = int(count)
        if count < 1:
            raise ValidationError("from_pages_blank needs at least one page, got %d" % count)
        if float(width) <= 0.0 or float(height) <= 0.0:
            raise ValidationError("page size must be positive, got %rx%r" % (width, height))

        media = PdfArray([0, 0, float(width), float(height)])
        objects: Dict[int, Any] = {}
        catalog_num, pages_num = 1, 2
        first_page_num = 3
        kids = PdfArray()
        for i in range(count):
            page_num = first_page_num + 2 * i
            content_num = page_num + 1
            objects[page_num] = PdfDict(
                {
                    "Type": PdfName("Page"),
                    "Parent": PdfRef(pages_num),
                    "MediaBox": PdfArray(media),
                    "Resources": PdfDict({"ProcSet": PdfArray([PdfName("PDF"), PdfName("Text")])}),
                    "Contents": PdfRef(content_num),
                }
            )
            objects[content_num] = PdfStream(PdfDict({"Length": 0}), b"")
            kids.append(PdfRef(page_num))

        objects[pages_num] = PdfDict(
            {
                "Type": PdfName("Pages"),
                "Kids": kids,
                "Count": count,
                "MediaBox": PdfArray(media),
            }
        )
        objects[catalog_num] = PdfDict(
            {"Type": PdfName("Catalog"), "Pages": PdfRef(pages_num)}
        )
        data = build_document(objects, PdfRef(catalog_num))
        return Document.open(data)

    # -- object access ----------------------------------------------------------------
    def resolve(self, obj: Any) -> Any:
        """Follow indirect references, preferring objects staged in the writer.

        This is what makes the class satisfy the ``Resolver`` protocol.  Objects created
        during this session (a new AcroForm, an updated ``/Annots`` array) live only in
        ``writer.updates``; consulting that first is what lets them be read straight back.

        Args:
            obj: Any PDF value, reference or not.

        Returns:
            The referenced object, or :data:`PdfNull.NULL` for a dangling or cyclic
            reference.
        """
        seen: set = set()
        current = obj
        while isinstance(current, PdfRef):
            if current.num in seen or len(seen) >= _MAX_DEPTH:
                return PdfNull.NULL
            seen.add(current.num)
            if current.num in self.writer.updates:
                current = self.writer.updates[current.num]
                continue
            try:
                current = self.file.get_object(current.num, current.gen)
            except Exception:
                return PdfNull.NULL
            if current is None:
                return PdfNull.NULL
        return current

    @property
    def catalog(self) -> PdfDict:
        """The document catalog, with pending updates applied."""
        root = None
        trailer = getattr(self.file, "trailer", None)
        if isinstance(trailer, dict):
            root = trailer.get("Root")
        value = self.resolve(root)
        if isinstance(value, PdfDict):
            return value
        value = getattr(self.file, "catalog", None)
        return value if isinstance(value, PdfDict) else PdfDict()

    def catalog_ref(self) -> Optional[PdfRef]:
        """The catalog's indirect reference, when the trailer names one."""
        trailer = getattr(self.file, "trailer", None)
        root = trailer.get("Root") if isinstance(trailer, dict) else None
        return root if isinstance(root, PdfRef) else None

    # -- pages ------------------------------------------------------------------------
    @property
    def pages(self) -> List[Page]:
        """Every page in reading order.  Built once and cached."""
        if self._pages is None:
            dicts = list(self.file.page_dicts() or ())
            try:
                refs = list(self.file.page_refs() or ())
            except Exception:  # pragma: no cover - defensive
                refs = []
            pages: List[Page] = []
            for index, page_dict in enumerate(dicts):
                ref = refs[index] if index < len(refs) and isinstance(refs[index], PdfRef) else None
                pages.append(Page(self, index, page_dict, ref))
            self._pages = pages
        return self._pages

    @property
    def page_count(self) -> int:
        """The number of pages."""
        return len(self.pages)

    def page(self, i: int) -> Page:
        """Return page ``i``, accepting Python-style negative indices.

        Raises:
            ValidationError: The index is out of range.
        """
        pages = self.pages
        index = int(i)
        if index < 0:
            index += len(pages)
        if not 0 <= index < len(pages):
            raise ValidationError("page index %r out of range (0..%d)" % (i, len(pages) - 1))
        return pages[index]

    # -- AcroForm ---------------------------------------------------------------------
    def acroform(self) -> Optional[PdfDict]:
        """Return the resolved ``/AcroForm`` dictionary, or ``None`` when there is none."""
        catalog = self.catalog
        raw = catalog.get("AcroForm")
        if raw is None or isinstance(raw, PdfNull):
            return None
        if isinstance(raw, PdfRef):
            self.acroform_ref = raw
        value = self.resolve(raw)
        return value if isinstance(value, PdfDict) else None

    def ensure_acroform(self) -> PdfDict:
        """Return the document's ``/AcroForm``, creating it if necessary.

        Idempotent: calling it twice returns the same dictionary and stages no second
        update.  A form that already exists but is stored *directly* in the catalog is
        promoted to an indirect object first, because the writer can only persist a
        change to an object that has a number of its own.

        Returns:
            The AcroForm dictionary, ready to be mutated.

        Raises:
            PdfWriteError: The catalog cannot be updated (no ``/Root`` reference and no
                way to synthesize one).
        """
        catalog = self.catalog
        raw = catalog.get("AcroForm")
        existing = self.resolve(raw) if raw is not None else None

        if isinstance(existing, PdfDict):
            if isinstance(raw, PdfRef):
                self.acroform_ref = raw
                # Stage it even though nothing has changed yet.  The dictionary is
                # handed back "ready to be mutated", and a caller that appends to
                # /Fields mutates this object in place -- if the writer has not been
                # told the object number, that edit is silently dropped by the next
                # incremental write.  Re-staging an already staged number is a no-op.
                self.writer.set_object(raw.num, existing)
                return existing
            # Direct dictionary: promote so later edits are persistable.
            ref = self.writer.add_object(existing)
            catalog["AcroForm"] = ref
            self._register_catalog(catalog)
            self.acroform_ref = ref
            return existing

        acroform = PdfDict(
            {
                "Fields": PdfArray(),
                "DA": PdfString(b"/Helv 0 Tf 0 g"),
                "DR": PdfDict({"Font": PdfDict()}),
            }
        )
        ref = self.writer.add_object(acroform)
        catalog["AcroForm"] = ref
        self._register_catalog(catalog)
        self.acroform_ref = ref
        _log.debug("%s: created /AcroForm as object %d", self.document_id, ref.num)
        return acroform

    def _register_catalog(self, catalog: PdfDict) -> None:
        """Stage the catalog for writing, synthesizing a ``/Root`` object if needed."""
        ref = self.catalog_ref()
        if ref is not None:
            self.writer.set_object(ref.num, catalog)
            return
        new_ref = self.writer.add_object(catalog)
        self.writer.update_trailer("Root", new_ref)
        _log.debug("%s: catalog had no /Root reference; wrote object %d", self.document_id, new_ref.num)

    # -- form flavour -----------------------------------------------------------------
    def has_xfa(self) -> bool:
        """True when the AcroForm carries an ``/XFA`` packet."""
        acroform = self.acroform()
        if acroform is None:
            return False
        value = acroform.get("XFA")
        return value is not None and not isinstance(value, PdfNull)

    def _xfa_bytes(self, limit: int = 1 << 20) -> bytes:
        """Concatenate the XFA packet's streams, capped at ``limit`` bytes."""
        acroform = self.acroform()
        if acroform is None:
            return b""
        packet = self.resolve(acroform.get("XFA"))
        parts: List[bytes] = []
        total = 0
        for item in _as_sequence(packet):
            value = self.resolve(item)
            if not isinstance(value, PdfStream):
                continue
            try:
                chunk = value.decoded(self)
            except Exception:  # pragma: no cover - decoding is already lenient
                continue
            parts.append(chunk)
            total += len(chunk)
            if total >= limit:
                break
        return b"".join(parts)[:limit]

    def xfa_is_dynamic(self) -> bool:
        """Best-effort test for a *dynamic* XFA form.

        Two independent signals are accepted, because producers are inconsistent:
        the catalog's ``/NeedsRendering`` flag, and a ``<dynamicRender>required``
        element inside the XFA ``config`` packet.  Static XFA (which behaves like a
        plain AcroForm) reports ``False``.
        """
        catalog = self.catalog
        if catalog.get_bool("NeedsRendering", False, self) is True:
            return True
        if not self.has_xfa():
            return False
        blob = self._xfa_bytes()
        if not blob:
            return False
        start = 0
        while True:
            found = blob.find(b"dynamicRender", start)
            if found < 0:
                return False
            window = blob[found : found + 96]
            if b"required" in window:
                return True
            start = found + len(b"dynamicRender")

    def is_signed(self) -> bool:
        """True when the document carries a signature or a document-modification lock.

        Checks all three places a signature shows up: a signed ``/Sig`` field in the
        AcroForm tree, the catalog's ``/Perms /DocMDP`` certification entry, and a
        signature widget attached directly to a page.
        """
        perms = self.resolve(self.catalog.get("Perms"))
        if isinstance(perms, PdfDict):
            docmdp = perms.get("DocMDP")
            if docmdp is not None and not isinstance(docmdp, PdfNull):
                return True

        for entry in self._iter_terminal_fields():
            if entry.field_kind == "Sig" and entry.has_value:
                return True

        for page in self.pages:
            for annot in page.annotations():
                if annot.get_name("Subtype", None, self) != "Widget":
                    continue
                if annot.get_name("FT", None, self) != "Sig":
                    continue
                value = annot.get("V")
                if value is not None and not isinstance(value, PdfNull):
                    return True
        return False

    # -- field tree walking -----------------------------------------------------------
    class _TerminalField:
        """A terminal node of the AcroForm field tree plus its resolved inheritance."""

        __slots__ = ("name", "node", "ref", "attrs", "widgets")

        def __init__(
            self,
            name: str,
            node: PdfDict,
            ref: Optional[PdfRef],
            attrs: Dict[str, Any],
            widgets: List[Tuple[Optional[PdfRef], PdfDict]],
        ) -> None:
            self.name = name
            self.node = node
            self.ref = ref
            self.attrs = attrs
            self.widgets = widgets

        @property
        def field_kind(self) -> Optional[str]:
            """The ``/FT`` value (``Tx``/``Btn``/``Ch``/``Sig``) after inheritance."""
            value = self.attrs.get("FT")
            if isinstance(value, PdfName):
                return value.value
            if isinstance(value, str):
                return value[1:] if value.startswith("/") else value
            return None

        @property
        def has_value(self) -> bool:
            """True when ``/V`` is present and not null."""
            value = self.attrs.get("V")
            return value is not None and not isinstance(value, PdfNull)

    def _kid_entries(self, node: PdfDict) -> List[Tuple[Optional[PdfRef], PdfDict]]:
        """Return ``/Kids`` as ``(reference, resolved dictionary)`` pairs."""
        kids = self.resolve(node.get("Kids"))
        out: List[Tuple[Optional[PdfRef], PdfDict]] = []
        for entry in _as_sequence(kids):
            ref = entry if isinstance(entry, PdfRef) else None
            value = self.resolve(entry)
            if isinstance(value, PdfDict):
                out.append((ref, value))
        return out

    def _iter_terminal_fields(self) -> List["Document._TerminalField"]:
        """Walk the AcroForm field tree and return every terminal field.

        A kid is a *field* when it has its own ``/T`` or its own ``/Kids``; anything else
        under a terminal field is one of that field's widget annotations.  Names are
        fully qualified by joining each level's ``/T`` with ``'.'``, and the inheritable
        attributes are merged top-down so a radio kid sees its parent's ``/FT`` and
        ``/Ff``.
        """
        acroform = self.acroform()
        if acroform is None:
            return []
        out: List[Document._TerminalField] = []
        visited: set = set()
        for ref, node in self._kid_entries(PdfDict({"Kids": acroform.get("Fields")})):
            self._walk_field(ref, node, "", {}, 0, visited, out)
        return out

    def _walk_field(
        self,
        ref: Optional[PdfRef],
        node: PdfDict,
        parent_name: str,
        inherited: Mapping[str, Any],
        depth: int,
        visited: set,
        out: List["Document._TerminalField"],
    ) -> None:
        """Recursive half of :meth:`_iter_terminal_fields`."""
        if depth > _MAX_DEPTH:
            return
        if ref is not None:
            if ref.num in visited:
                return
            visited.add(ref.num)

        attrs: Dict[str, Any] = dict(inherited)
        for key in INHERITABLE_FIELD_KEYS:
            if key in node:
                value = node.get(key)
                if value is not None and not isinstance(value, PdfNull):
                    attrs[key] = self.resolve(value)

        partial = self._text_of(node.get("T"))
        if partial:
            name = "%s.%s" % (parent_name, partial) if parent_name else partial
        else:
            name = parent_name

        kids = self._kid_entries(node)
        field_kids = [(r, k) for r, k in kids if "T" in k or "Kids" in k]
        if field_kids:
            for kid_ref, kid in field_kids:
                self._walk_field(kid_ref, kid, name, attrs, depth + 1, visited, out)
            return

        widgets = [(r, k) for r, k in kids]
        if not widgets and ("Rect" in node or "Subtype" in node):
            widgets = [(ref, node)]
        out.append(Document._TerminalField(name, node, ref, attrs, widgets))

    def _text_of(self, value: Any) -> str:
        """Decode a ``/T``-style value to text, tolerating raw ``str``/``bytes``."""
        value = self.resolve(value)
        if isinstance(value, PdfString):
            return value.text()
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return PdfString(bytes(value)).text()
        return ""

    def _value_text(self, value: Any) -> Optional[str]:
        """Render a ``/V``-style value as text, or ``None`` when it has no text form."""
        value = self.resolve(value)
        if value is None or isinstance(value, PdfNull):
            return None
        if isinstance(value, PdfName):
            return value.value
        if isinstance(value, PdfString):
            return value.text()
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format_number(value).decode("ascii")
        if isinstance(value, PdfStream):
            try:
                return value.decoded(self).decode("utf-8", "replace")
            except Exception:  # pragma: no cover
                return None
        if isinstance(value, (PdfArray, list, tuple)):
            parts = [self._value_text(item) for item in value]
            kept = [p for p in parts if p is not None]
            return ", ".join(kept) if kept else None
        return None

    # -- page lookup for widgets ------------------------------------------------------
    def _page_lookup(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Return ``(page object number -> index, annotation number -> page index)``."""
        by_page_num: Dict[int, int] = {}
        by_annot_num: Dict[int, int] = {}
        for page in self.pages:
            if page.ref is not None:
                by_page_num.setdefault(page.ref.num, page.index)
            for annot_ref in page.annotation_refs():
                by_annot_num.setdefault(annot_ref.num, page.index)
        return by_page_num, by_annot_num

    def _widget_page(
        self,
        widget_ref: Optional[PdfRef],
        widget: PdfDict,
        by_page_num: Mapping[int, int],
        by_annot_num: Mapping[int, int],
    ) -> int:
        """Resolve the page index a widget lives on, falling back to a page scan."""
        parent = widget.get("P")
        if isinstance(parent, PdfRef):
            index = by_page_num.get(parent.num)
            if index is not None:
                return index
        if widget_ref is not None:
            index = by_annot_num.get(widget_ref.num)
            if index is not None:
                return index
        for page in self.pages:
            for annot in page.annotations():
                if annot is widget:
                    return page.index
        return 0

    def _widget_rect(self, widget: PdfDict) -> Rect:
        """Read and normalize a widget's ``/Rect``; a missing box degenerates to zero."""
        items = _as_sequence(self.resolve(widget.get("Rect")))
        numbers: List[float] = []
        for item in items[:4]:
            item = self.resolve(item)
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return Rect(0.0, 0.0, 0.0, 0.0)
            numbers.append(float(item))
        if len(numbers) < 4:
            return Rect(0.0, 0.0, 0.0, 0.0)
        return Rect.from_list(numbers)

    def _export_value(self, widgets: Sequence[Tuple[Optional[PdfRef], PdfDict]]) -> Optional[str]:
        """The "on" state of a checkbox or radio widget, read from ``/AP /N`` or ``/AS``."""
        for _ref, widget in widgets:
            appearance = self.resolve(widget.get("AP"))
            if isinstance(appearance, PdfDict):
                normal = self.resolve(appearance.get("N"))
                if isinstance(normal, PdfDict):
                    for state in normal.keys():
                        if state != "Off":
                            return state
            state = widget.get_name("AS", None, self)
            if state and state != "Off":
                return state
        return None

    @staticmethod
    def _map_field_type(kind: Optional[str], flags: int) -> FieldType:
        """Map ``/FT`` plus ``/Ff`` onto the engine's :class:`FieldType` vocabulary."""
        if kind == "Btn":
            if flags & FF_RADIO:
                return FieldType.RADIO
            if flags & FF_PUSHBUTTON:
                return FieldType.BUTTON
            return FieldType.CHECKBOX
        if kind == "Ch":
            return FieldType.CHOICE if flags & FF_COMBO else FieldType.LISTBOX
        if kind == "Tx":
            if flags & FF_MULTILINE:
                return FieldType.MULTILINE_TEXT
            if flags & FF_COMB:
                return FieldType.COMB
            return FieldType.TEXT
        if kind == "Sig":
            return FieldType.SIGNATURE
        return FieldType.UNKNOWN

    @staticmethod
    def parse_default_appearance(da: Any) -> Tuple[str, float]:
        """Extract ``(font name, size)`` from a ``/DA`` string, e.g. ``/Helv 11 Tf``.

        Returns ``("Helv", 0.0)`` when the string carries no usable font operator; a
        size of zero is the PDF convention for "auto-fit", so it is a safe default.
        """
        if isinstance(da, PdfString):
            raw = da.raw
        elif isinstance(da, (bytes, bytearray)):
            raw = bytes(da)
        elif isinstance(da, str):
            raw = da.encode("latin-1", "replace")
        else:
            return ("Helv", 0.0)
        matches = _DA_FONT_RE.findall(raw)
        if not matches:
            return ("Helv", 0.0)
        name, size = matches[-1]
        try:
            value = float(size)
        except ValueError:  # pragma: no cover - regex guarantees a number
            value = 0.0
        return (PdfName.decode(name).value, value)

    def _choices(self, opt: Any) -> List[str]:
        """Flatten ``/Opt`` into display strings, honouring ``[export, display]`` pairs."""
        out: List[str] = []
        for item in _as_sequence(self.resolve(opt)):
            item = self.resolve(item)
            if isinstance(item, (PdfArray, list, tuple)) and item:
                text = self._value_text(item[-1])
            else:
                text = self._value_text(item)
            if text is not None:
                out.append(text)
        return out

    def existing_fields(self) -> List[FieldSpec]:
        """Read the document's real AcroForm back as :class:`FieldSpec` records.

        Walks the whole field tree, resolving fully-qualified names, inherited
        attributes, widget rectangles and the page each widget sits on.  A field owning
        several widgets reports the first in ``rect``/``page`` and the remainder in
        ``extra_widgets``, matching the writer's own input shape -- so a document can be
        read, edited and written back without an information-losing round trip.

        Returns:
            One :class:`FieldSpec` per terminal field, in field-tree order.
        """
        terminals = self._iter_terminal_fields()
        if not terminals:
            return []
        by_page_num, by_annot_num = self._page_lookup()
        specs: List[FieldSpec] = []

        for terminal in terminals:
            attrs = terminal.attrs
            kind = terminal.field_kind
            flags_value = attrs.get("Ff")
            flags = int(flags_value) if isinstance(flags_value, (int, float)) and not isinstance(flags_value, bool) else 0
            field_type = self._map_field_type(kind, flags)

            placed: List[Tuple[int, Rect]] = []
            for widget_ref, widget in terminal.widgets:
                placed.append(
                    (
                        self._widget_page(widget_ref, widget, by_page_num, by_annot_num),
                        self._widget_rect(widget),
                    )
                )
            if not placed:
                placed = [(0, Rect(0.0, 0.0, 0.0, 0.0))]

            max_length = attrs.get("MaxLen")
            max_len = (
                int(max_length)
                if isinstance(max_length, (int, float)) and not isinstance(max_length, bool)
                else None
            )
            alignment = attrs.get("Q")
            font_name, font_size = self.parse_default_appearance(attrs.get("DA"))

            spec = FieldSpec(
                name=terminal.name,
                field_type=field_type,
                page=placed[0][0],
                rect=placed[0][1],
                value=self._value_text(attrs.get("V")),
                default_value=self._value_text(attrs.get("DV")),
                tooltip=self._value_text(attrs.get("TU")),
                required=bool(flags & FF_REQUIRED),
                read_only=bool(flags & FF_READ_ONLY),
                max_length=max_len,
                multiline=bool(flags & FF_MULTILINE),
                comb_cells=max_len if (flags & FF_COMB and max_len) else None,
                choices=self._choices(attrs.get("Opt")),
                export_value=(
                    self._export_value(terminal.widgets)
                    if field_type in (FieldType.CHECKBOX, FieldType.RADIO)
                    else None
                ),
                font_name=font_name,
                font_size=font_size,
                alignment=int(alignment)
                if isinstance(alignment, (int, float)) and not isinstance(alignment, bool)
                else 0,
                group=terminal.name if field_type is FieldType.RADIO else None,
                tab_order=len(specs),
                extra_widgets=list(placed[1:]),
            )
            specs.append(spec)
        return specs

    # -- output -----------------------------------------------------------------------
    def to_bytes(self, incremental: bool = True) -> bytes:
        """Serialize the document with every staged change applied.

        Args:
            incremental: When true (the default) the original bytes are preserved
                verbatim and a new revision is appended.  When false the whole document
                is rewritten as a single revision.

        Returns:
            The complete PDF bytes.
        """
        if incremental:
            return self.writer.write_incremental()
        return self.writer.write_full()

    def save(self, path: "str | os.PathLike", incremental: bool = True) -> None:
        """Write the document to ``path``.  See :meth:`to_bytes` for ``incremental``."""
        data = self.to_bytes(incremental=incremental)
        target = os.fspath(path)
        with open(target, "wb") as handle:
            handle.write(data)
        _log.debug("%s: wrote %d bytes to %s", self.document_id, len(data), target)

    def __repr__(self) -> str:
        return "Document(id=%s, pages=%d)" % (self.document_id, self.page_count)
