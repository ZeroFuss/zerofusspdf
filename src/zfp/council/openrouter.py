"""A minimal OpenRouter client over :mod:`urllib.request`.

This is the only module in ZFP that can open a socket, and it is written to make that
fact obvious and auditable:

* the transport is one module-level function, :func:`_transport`, so a test can replace
  it and prove no socket was opened;
* every policy check runs **before** the transport is called, never after;
* ``requests`` is not imported, here or anywhere else -- the whole core path runs on a
  bare CPython stdlib.

The request asks for strict structured output
(``response_format.json_schema`` with ``strict: true``) so the reply is machine-checked
on the provider's side as well as ours, and it pins provider routing to zero-data-retention
endpoints on the allow-list.  Nothing is retried: a retry would make the run
non-reproducible and would send the same context twice.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..core.config import PrivacyConfig
from ..core.errors import CouncilError, PolicyError

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "REFERER",
    "TITLE",
    "build_payload",
    "build_headers",
    "chat_json",
    "monotonic_ms",
]

#: OpenRouter's OpenAI-compatible endpoint root.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Deliberately provider-agnostic: OpenRouter's own router picks a model that satisfies
#: the structured-output requirement.  Override per deployment with ``ZFP_COUNCIL_MODEL``
#: (read by :func:`zfp.council.council.build_default_council`) or by constructing
#: :class:`~zfp.council.members.OpenRouterMember` with an explicit model id.
DEFAULT_MODEL = "openrouter/auto"

#: Seconds.  One request, no retries.
DEFAULT_TIMEOUT = 30.0

#: Attribution headers OpenRouter uses for rate-limit accounting.
REFERER = "https://github.com/zerofusspdf/zfp"
TITLE = "ZFP semantic council"


def monotonic_ms() -> float:
    """Return a monotonic millisecond reading, used only for remote latency.

    Never call this on the local path: a local council must be reproducible, and a
    clock reading is not.
    """
    return time.monotonic() * 1000.0


def _transport(url: str, data: bytes, headers: Mapping[str, str], timeout: float) -> bytes:
    """Perform the HTTP POST.  Replaced wholesale in tests.

    This is the single point in ZFP where bytes leave the machine.
    """
    request = urllib.request.Request(url, data=data, method="POST")
    for name, value in headers.items():
        request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def build_payload(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    zdr: bool = True,
    allow_providers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the JSON request body.

    ``temperature`` is pinned to 0 and ``seed`` to 0: the council's determinism
    requirement extends as far as the provider will honour it.
    """
    provider: Dict[str, Any] = {"require_parameters": True, "allow_fallbacks": False}
    if zdr:
        provider["zdr"] = True
        provider["data_collection"] = "deny"
    else:
        provider["data_collection"] = "allow"
    if allow_providers:
        provider["order"] = [str(p) for p in allow_providers]
        provider["only"] = [str(p) for p in allow_providers]
    return {
        "model": str(model),
        "messages": [dict(m) for m in messages],
        "temperature": 0,
        "seed": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "zfp_answer",
                "strict": True,
                "schema": dict(schema),
            },
        },
        "provider": provider,
    }


def build_headers(api_key: str) -> Dict[str, str]:
    """Build the request headers, including OpenRouter's attribution pair."""
    return {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": REFERER,
        "X-Title": TITLE,
    }


def _check_policy(
    api_key: Optional[str],
    base_url: str,
    zdr: bool,
    allow_providers: Optional[Sequence[str]],
    privacy: Optional[PrivacyConfig],
) -> List[str]:
    """Run every egress check.  Raises before any socket exists.

    Returns the effective provider allow-list.
    """
    if privacy is not None and not privacy.allow_external_inference:
        raise PolicyError(
            "external inference is disabled (PrivacyConfig.allow_external_inference is "
            "False); no request was made"
        )
    if not api_key:
        raise PolicyError("no OpenRouter API key: refusing to open a connection")
    if not str(base_url).lower().startswith("https://"):
        raise PolicyError("refusing plaintext egress to %r" % base_url)

    providers = [str(p) for p in (allow_providers or [])]
    if privacy is not None:
        if privacy.require_zero_data_retention and not zdr:
            raise PolicyError(
                "PrivacyConfig.require_zero_data_retention is set but zdr routing was "
                "not requested"
            )
        allowed = [str(p) for p in privacy.provider_allowlist]
        if allowed:
            if not providers:
                providers = allowed
            else:
                rejected = sorted(set(providers) - set(allowed))
                if rejected:
                    raise PolicyError(
                        "providers %s are not on PrivacyConfig.provider_allowlist"
                        % ", ".join(rejected)
                    )
    return providers


def _content_of(reply: Mapping[str, Any]) -> str:
    """Extract the assistant message text from a chat-completions reply."""
    choices = reply.get("choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        error = reply.get("error")
        if error:
            raise CouncilError("OpenRouter returned an error: %s" % json.dumps(error, sort_keys=True))
        raise CouncilError("OpenRouter reply carried no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise CouncilError("OpenRouter reply carried no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        # Some providers return content as a list of typed parts.
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") in (None, "text")
        ]
        if parts:
            return "".join(parts)
    raise CouncilError("OpenRouter reply carried no textual content")


def chat_json(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    api_key: Optional[str],
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    zdr: bool = True,
    allow_providers: Optional[Sequence[str]] = None,
    privacy: Optional[PrivacyConfig] = None,
) -> Dict[str, Any]:
    """Ask a model one schema-constrained question and return the parsed answer.

    Args:
        model: OpenRouter model id, e.g. ``"openrouter/auto"``.
        messages: Chat messages, already redacted by the caller.
        schema: The JSON Schema the answer must satisfy; sent as strict structured
            output so the provider enforces it too.
        api_key: OpenRouter key.  Missing or empty is a policy refusal, not an error.
        base_url: Endpoint root.  Must be ``https``.
        timeout: Socket timeout in seconds.
        zdr: Request zero-data-retention routing.
        allow_providers: Provider allow-list; defaults to
            ``privacy.provider_allowlist`` when one is configured.
        privacy: Egress policy.  When supplied and external inference is disabled, this
            raises **before** the transport is touched.

    Returns:
        The parsed JSON object from the first choice's message content.

    Raises:
        PolicyError: Egress is not permitted, no key is present, the endpoint is not
            https, ZDR was required but not requested, or a provider is off the
            allow-list.  No socket is opened in any of those cases.
        CouncilError: The transport failed or the reply was not the expected shape.
    """
    providers = _check_policy(api_key, base_url, zdr, allow_providers, privacy)

    payload = build_payload(model, messages, schema, zdr=zdr, allow_providers=providers)
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    url = str(base_url).rstrip("/") + "/chat/completions"

    try:
        raw = _transport(url, data, build_headers(str(api_key)), float(timeout))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise CouncilError("OpenRouter HTTP %s" % exc.code) from None
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise CouncilError("OpenRouter transport failed: %s" % exc.reason) from None
    except Exception as exc:  # noqa: BLE001 - never leak a key through a traceback
        raise CouncilError("OpenRouter transport failed: %s" % type(exc).__name__) from None

    try:
        reply = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CouncilError("OpenRouter reply was not JSON: %s" % exc) from None
    if not isinstance(reply, Mapping):
        raise CouncilError("OpenRouter reply was not a JSON object")

    content = _content_of(reply)
    try:
        answer = json.loads(content)
    except ValueError as exc:
        raise CouncilError("model content was not JSON: %s" % exc) from None
    if not isinstance(answer, Mapping):
        raise CouncilError("model content was not a JSON object")
    return dict(answer)
