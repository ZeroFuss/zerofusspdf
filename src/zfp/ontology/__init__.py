"""The canonical form ontology: key namespace, alias index, and placeholder patterns.

``zfp.ontology`` is what lets "ZIP", "Postal Code", "Mail ZIP", "Zipcode" and
"ZIP / Postal" all resolve to the single key ``person.address.postal_code``, and what
lets a blank printed ``##-#######`` be recognized as an EIN before any model is asked.

Three layers:

:mod:`zfp.ontology.keys`
    :data:`CANONICAL_KEYS` -- the declared key namespace, one :class:`KeySpec` per key.
:mod:`zfp.ontology.aliases`
    :data:`ALIAS_INDEX` plus :func:`normalize_label`, :func:`lookup`,
    :func:`context_lookup` and :func:`fuzzy_lookup`.
:mod:`zfp.ontology.patterns`
    :data:`PATTERNS` plus :func:`match_placeholder`, :func:`match_value` and
    :func:`infer_from_context`.

Nothing here touches a PDF, a model or the network; the whole module is a pure,
deterministic dictionary.
"""

from __future__ import annotations

from .aliases import (
    ALIAS_INDEX,
    alias_count,
    aliases_for,
    context_lookup,
    fuzzy_lookup,
    lookup,
    normalize_label,
)
from .keys import (
    CANONICAL_KEYS,
    KeySpec,
    all_keys,
    children,
    get,
    key_count,
    keys_with_parent,
    namespaces,
)
from .patterns import (
    PATTERNS,
    PATTERNS_BY_NAME,
    PatternRule,
    infer_from_context,
    match_all,
    match_placeholder,
    match_value,
    rules_for_key,
)

__all__ = [
    # keys
    "KeySpec",
    "CANONICAL_KEYS",
    "get",
    "all_keys",
    "children",
    "namespaces",
    "keys_with_parent",
    "key_count",
    # aliases
    "ALIAS_INDEX",
    "normalize_label",
    "lookup",
    "fuzzy_lookup",
    "context_lookup",
    "alias_count",
    "aliases_for",
    # patterns
    "PatternRule",
    "PATTERNS",
    "PATTERNS_BY_NAME",
    "match_placeholder",
    "match_value",
    "match_all",
    "infer_from_context",
    "rules_for_key",
]
