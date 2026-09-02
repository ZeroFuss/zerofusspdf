"""The profile vault: an authorized, provenance-bearing store of identity/company data.

"ZFP should not ask a generative model to invent answers." Every value the autofill
resolver can place into a field comes from here, carries a record of where it came from,
and can be re-derived (a full name from a first and last name) without ever inventing a
fact the vault does not actually hold.
"""

from __future__ import annotations

import binascii
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.errors import VaultError
from ..ontology import get as ontology_get
from . import cipher

_PLAIN_MAGIC = b"ZFPJ1\n"
_ENCRYPTED_MAGIC = b"ZFPV1\n"

_SECRET_SENTINEL = "•••"


@dataclass
class VaultEntry:
    """One stored value and everything needed to trust or explain it."""

    key: str
    value: str
    source: str = "manual"
    verified_at: Optional[str] = None
    confidence: float = 1.0
    sensitivity: str = "normal"
    labels_seen: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "verified_at": self.verified_at,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity,
            "labels_seen": list(self.labels_seen),
            "provenance": dict(self.provenance),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VaultEntry":
        return VaultEntry(
            key=d["key"], value=d.get("value", ""), source=d.get("source", "manual"),
            verified_at=d.get("verified_at"), confidence=float(d.get("confidence", 1.0)),
            sensitivity=d.get("sensitivity", "normal"),
            labels_seen=list(d.get("labels_seen", [])),
            provenance=dict(d.get("provenance", {})),
        )


def _sensitivity_for(key: str) -> str:
    spec = ontology_get(key)
    return spec.sensitivity if spec else "normal"


# --------------------------------------------------------------------------------------
# Derivation rules: how to synthesize a value from its components when it is missing.
# --------------------------------------------------------------------------------------

def _derive_full_name(get) -> Optional[Tuple[str, List[str]]]:
    first = get("person.name.first")
    last = get("person.name.last")
    if not first or not last:
        return None
    middle = get("person.name.middle_initial") or get("person.name.middle")
    parts = [first.value]
    if middle:
        parts.append((middle.value[:1] + ".") if len(middle.value) > 1 else middle.value)
    parts.append(last.value)
    used = ["person.name.first", "person.name.last"] + (
        ["person.name.middle"] if middle else [])
    return " ".join(parts), used


def _derive_initials(get) -> Optional[Tuple[str, List[str]]]:
    first = get("person.name.first")
    last = get("person.name.last")
    if not first or not last:
        return None
    value = (first.value[:1] + last.value[:1]).upper()
    return value, ["person.name.first", "person.name.last"]


def _derive_full_address(prefix: str):
    def _fn(get) -> Optional[Tuple[str, List[str]]]:
        street = get(f"{prefix}.address.street_1")
        city = get(f"{prefix}.address.city")
        region = get(f"{prefix}.address.region")
        postal = get(f"{prefix}.address.postal_code")
        if not (street and city and region):
            return None
        parts = [street.value]
        if get(f"{prefix}.address.street_2"):
            parts[0] = parts[0] + ", " + get(f"{prefix}.address.street_2").value
        tail = f"{city.value}, {region.value}"
        if postal:
            tail += f" {postal.value}"
        parts.append(tail)
        used = [f"{prefix}.address.street_1", f"{prefix}.address.city",
                f"{prefix}.address.region"]
        return ", ".join(parts), used
    return _fn


_DERIVATIONS = {
    "person.name.full": _derive_full_name,
    "person.name.initials": _derive_initials,
    "person.address.full": _derive_full_address("person"),
    "company.address.full": _derive_full_address("company"),
    "billing.address.full": _derive_full_address("billing"),
    "shipping.address.full": _derive_full_address("shipping"),
}

#: A parent-context word maps to the namespace prefix it should redirect resolution to.
_PARENT_PREFIXES = {"billing": "billing", "shipping": "shipping", "company": "company"}


def _rekey_for_prefix(key: str, prefix: str) -> Optional[str]:
    """``person.address.street_1`` under a ``"billing"`` context -> ``billing.address.street_1``."""
    for base in ("person", "company"):
        if key.startswith(base + "."):
            return prefix + key[len(base):]
    return None


class ProfileVault:
    """An in-memory, optionally encrypted store of :class:`VaultEntry` values."""

    def __init__(self, entries: Optional[Iterable[VaultEntry]] = None, *,
                profile_id: str = "default") -> None:
        self.profile_id = profile_id
        self._entries: Dict[str, VaultEntry] = {}
        for e in entries or ():
            self._entries[e.key] = e

    # -- basic access -------------------------------------------------------------------

    def put(self, key: str, value: str, **provenance: Any) -> VaultEntry:
        entry = VaultEntry(
            key=key, value=value,
            source=provenance.pop("source", "manual"),
            verified_at=provenance.pop("verified_at", None),
            confidence=float(provenance.pop("confidence", 1.0)),
            sensitivity=provenance.pop("sensitivity", _sensitivity_for(key)),
            labels_seen=list(provenance.pop("labels_seen", [])),
            provenance=dict(provenance),
        )
        self._entries[key] = entry
        return entry

    def get(self, key: str) -> Optional[VaultEntry]:
        return self._entries.get(key)

    def keys(self) -> List[str]:
        return sorted(self._entries.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- resolution -----------------------------------------------------------------------

    def resolve(self, key: str, parent_context: Sequence[str] = ()) -> Optional[VaultEntry]:
        """Resolve ``key``, honouring parent-context scoping and derivation.

        Order: (1) a parent-scoped variant when ``parent_context`` names one
        (``person.address.street_1`` under ``("Billing",)`` tries
        ``billing.address.street_1`` first); (2) the exact key; (3) a synthesized value
        built from the key's own components, when a derivation rule exists.
        """
        normalized_ctx = {c.strip().lower() for c in parent_context}
        for word in normalized_ctx:
            prefix = _PARENT_PREFIXES.get(word)
            if prefix:
                rekeyed = _rekey_for_prefix(key, prefix)
                if rekeyed and rekeyed in self._entries:
                    return self._entries[rekeyed]

        if key in self._entries:
            return self._entries[key]

        rule = _DERIVATIONS.get(key)
        if rule is not None:
            result = rule(self.get)
            if result is not None:
                value, used_keys = result
                components = [self._entries[k] for k in used_keys if k in self._entries]
                confidence = min((c.confidence for c in components), default=0.9) * 0.98
                entry = VaultEntry(
                    key=key, value=value, source="derived", confidence=round(confidence, 4),
                    sensitivity=_sensitivity_for(key),
                    provenance={"derived_from": used_keys},
                )
                return entry
        return None

    def observe_label(self, key: str, label: str) -> None:
        """Record that ``label`` was seen naming ``key`` -- the repeated-form learning loop."""
        entry = self._entries.get(key)
        if entry is None:
            return
        norm = label.strip()
        if not norm or norm in entry.labels_seen:
            return
        entry.labels_seen.append(norm)
        if len(entry.labels_seen) > 32:
            entry.labels_seen = entry.labels_seen[-32:]

    def suggest_key_for_label(self, label: str) -> Optional[str]:
        norm = label.strip().lower()
        if not norm:
            return None
        for entry in self._entries.values():
            if any(seen.strip().lower() == norm for seen in entry.labels_seen):
                return entry.key
        return None

    # -- bulk operations ------------------------------------------------------------------

    def merge(self, other: "ProfileVault", prefer: str = "higher_confidence") -> "ProfileVault":
        merged = ProfileVault(list(self._entries.values()), profile_id=self.profile_id)
        for key, entry in other._entries.items():
            existing = merged._entries.get(key)
            if existing is None:
                merged._entries[key] = entry
                continue
            if prefer == "other":
                merged._entries[key] = entry
            elif prefer == "self":
                continue
            else:  # higher_confidence
                if entry.confidence > existing.confidence:
                    merged._entries[key] = entry
        return merged

    def redact_secrets(self) -> "ProfileVault":
        """A copy safe to log or trace: secret-sensitivity values are replaced."""
        out = ProfileVault(profile_id=self.profile_id)
        for key, entry in self._entries.items():
            if entry.sensitivity == "secret":
                out._entries[key] = VaultEntry(
                    key=key, value=_SECRET_SENTINEL, source=entry.source,
                    verified_at=entry.verified_at, confidence=entry.confidence,
                    sensitivity=entry.sensitivity, labels_seen=list(entry.labels_seen),
                    provenance={"redacted": True},
                )
            else:
                out._entries[key] = entry
        return out

    def stats(self) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        by_sensitivity: Dict[str, int] = {}
        for entry in self._entries.values():
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
            by_sensitivity[entry.sensitivity] = by_sensitivity.get(entry.sensitivity, 0) + 1
        return {"count": len(self._entries), "by_source": by_source,
                "by_sensitivity": by_sensitivity}

    @staticmethod
    def from_form_results(fill_report: Any, *, profile_id: str = "default") -> "ProfileVault":
        """Learn a vault from a completed form -- "a database of prior user answers"."""
        vault = ProfileVault(profile_id=profile_id)
        for filled in getattr(fill_report, "values", []):
            key = getattr(filled, "canonical_key", None)
            value = getattr(filled, "value", None)
            status = getattr(filled, "status", "")
            if not key or value is None or status != "filled":
                continue
            vault.put(key, value, source="prior_form",
                     confidence=float(getattr(filled, "confidence", 0.8)))
        return vault

    # -- persistence ------------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "entries": [e.as_dict() for e in self._entries.values()],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProfileVault":
        entries = [VaultEntry.from_dict(e) for e in d.get("entries", [])]
        return ProfileVault(entries, profile_id=d.get("profile_id", "default"))

    def save(self, path: "os.PathLike[str] | str", password: Optional[str] = None) -> None:
        payload = json.dumps(self.as_dict(), ensure_ascii=True, sort_keys=True).encode("utf-8")
        with open(path, "wb") as fh:
            if password is None:
                fh.write(_PLAIN_MAGIC)
                fh.write(payload)
                return
            salt = cipher.random_bytes(16)
            nonce = cipher.random_bytes(12)
            key = cipher.derive_key(password, salt)
            aad = self.profile_id.encode("utf-8")
            ciphertext = cipher.chacha20_poly1305_encrypt(key, nonce, payload, aad)
            header = {
                "kdf": cipher.kdf_name(), "salt": binascii.hexlify(salt).decode("ascii"),
                "nonce": binascii.hexlify(nonce).decode("ascii"), "aad": self.profile_id,
            }
            fh.write(_ENCRYPTED_MAGIC)
            fh.write(json.dumps(header, sort_keys=True).encode("utf-8"))
            fh.write(b"\n")
            fh.write(ciphertext)

    @staticmethod
    def load(path: "os.PathLike[str] | str", password: Optional[str] = None) -> "ProfileVault":
        with open(path, "rb") as fh:
            data = fh.read()
        if data.startswith(_PLAIN_MAGIC):
            body = data[len(_PLAIN_MAGIC):]
            try:
                return ProfileVault.from_dict(json.loads(body.decode("utf-8")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise VaultError("vault file is not valid JSON: %s" % exc) from exc
        if data.startswith(_ENCRYPTED_MAGIC):
            if password is None:
                raise VaultError("vault is encrypted; a password is required")
            rest = data[len(_ENCRYPTED_MAGIC):]
            header_line, _, ciphertext = rest.partition(b"\n")
            try:
                header = json.loads(header_line.decode("utf-8"))
                salt = binascii.unhexlify(header["salt"])
                nonce = binascii.unhexlify(header["nonce"])
                aad = header.get("aad", "").encode("utf-8")
            except (ValueError, KeyError, binascii.Error) as exc:
                raise VaultError("vault header is corrupt: %s" % exc) from exc
            key = cipher.derive_key(password, salt)
            try:
                payload = cipher.chacha20_poly1305_decrypt(key, nonce, ciphertext, aad)
            except VaultError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VaultError("wrong password or corrupt vault") from exc
            try:
                return ProfileVault.from_dict(json.loads(payload.decode("utf-8")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise VaultError("decrypted vault is not valid JSON: %s" % exc) from exc
        raise VaultError("not a ZFP vault file (bad magic)")


__all__ = ["VaultEntry", "ProfileVault"]
