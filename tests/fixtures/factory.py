"""Cached synthetic fixtures for the test suite.

Fixtures are *generated*, never committed: :mod:`zfp.synth` builds a real PDF plus its
ground truth from a seed, so the corpus is reproducible, reviewable in a diff and free of
binary blobs.  Generation is fast (a few milliseconds a document), but a test session
asks for the same ``(kind, seed)`` many times, so everything is memoized here.

    >>> from tests.fixtures.factory import build, corpus
    >>> form = build("underline", 0)
    >>> form.pdf_bytes is build("underline", 0).pdf_bytes      # same object, cached
    True
    >>> len(corpus("boxed", 3))
    3
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from zfp.synth import KINDS, SyntheticForm, SynthOptions, generate

__all__ = ["build", "corpus", "all_kinds", "clear_cache", "cache_size", "KINDS"]

_CACHE: Dict[Tuple[Any, ...], SyntheticForm] = {}


def _cache_key(kind: str, seed: int, options: Dict[str, Any]) -> Tuple[Any, ...]:
    """A hashable, order-independent key for one fixture request."""
    return (str(kind), int(seed)) + tuple(sorted((str(k), v) for k, v in options.items()))


def build(kind: str = "underline", seed: int = 0, **kw: Any) -> SyntheticForm:
    """Return the synthetic form for ``(kind, seed, **kw)``, generating it once.

    Args:
        kind: Any kind in :data:`zfp.synth.KINDS`.
        seed: The document seed.  Identical arguments always yield identical bytes.
        **kw: Any other :class:`~zfp.synth.SynthOptions` attribute (``pages``,
            ``rotation``, ``font``, ``include_sections``, ...).

    Returns:
        The cached :class:`~zfp.synth.SyntheticForm`.  It is shared between callers, so
        treat it as read-only.
    """
    key = _cache_key(kind, seed, kw)
    form = _CACHE.get(key)
    if form is None:
        form = generate(SynthOptions(kind=kind, seed=int(seed), **kw))
        _CACHE[key] = form
    return form


def corpus(kind: str = "underline", n: int = 5, *, seed: int = 0, **kw: Any) -> List[SyntheticForm]:
    """Return ``n`` forms of one kind with consecutive seeds, all cached.

    Args:
        kind: The kind to build.
        n: How many documents.
        seed: Seed of the first document; document ``i`` uses ``seed + i``.
        **kw: Forwarded to :func:`build`.

    Returns:
        The forms, in seed order.
    """
    return [build(kind, seed + i, **kw) for i in range(max(0, int(n)))]


def all_kinds() -> Tuple[str, ...]:
    """Every document kind the generator supports."""
    return tuple(KINDS)


def clear_cache() -> None:
    """Drop every memoized fixture (only useful for measuring generation itself)."""
    _CACHE.clear()


def cache_size() -> int:
    """How many distinct fixtures are currently memoized."""
    return len(_CACHE)
