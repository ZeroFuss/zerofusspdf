"""Logging for ZFP.

Every module obtains its logger with ``get_logger(__name__)``. The package root logger
``zfp`` carries a :class:`logging.NullHandler` so importing ZFP never emits output;
applications opt in with :func:`configure`. :class:`LogContext` attaches structured
``document_id`` / ``page`` / ``agent`` fields to every record emitted inside its block.
"""

from __future__ import annotations

import json as _json
import logging as _logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import IO, Any, Dict, Iterator, List, Optional

__all__ = [
    "ROOT_LOGGER_NAME",
    "get_logger",
    "configure",
    "LogContext",
    "JsonFormatter",
    "ContextFilter",
    "current_context",
    "reset",
]

ROOT_LOGGER_NAME = "zfp"

#: Structured fields injected on every record.
CONTEXT_FIELDS = ("document_id", "page", "agent")

_state = threading.local()
_lock = threading.RLock()
_installed_handlers: List[_logging.Handler] = []


def current_context() -> Dict[str, Any]:
    """Return a copy of the structured logging context for the calling thread."""
    stack = getattr(_state, "stack", None)
    if not stack:
        return {}
    return dict(stack[-1])


class ContextFilter(_logging.Filter):
    """Injects the active :class:`LogContext` fields onto every record."""

    def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003 - stdlib name
        ctx = current_context()
        for name in CONTEXT_FIELDS:
            if not hasattr(record, name):
                setattr(record, name, ctx.get(name))
            elif ctx.get(name) is not None:
                setattr(record, name, ctx[name])
        extra = {k: v for k, v in ctx.items() if k not in CONTEXT_FIELDS}
        if extra and not hasattr(record, "zfp_extra"):
            record.zfp_extra = extra  # type: ignore[attr-defined]
        return True


class JsonFormatter(_logging.Formatter):
    """Formats records as one compact JSON object per line."""

    def format(self, record: _logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        extra = getattr(record, "zfp_extra", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _json.dumps(payload, sort_keys=True, default=str)


def _root() -> _logging.Logger:
    """Return the ``zfp`` root logger, installing the default guards exactly once."""
    logger = _logging.getLogger(ROOT_LOGGER_NAME)
    with _lock:
        if not getattr(logger, "_zfp_initialized", False):
            logger.addHandler(_logging.NullHandler())
            logger.addFilter(ContextFilter())
            logger.setLevel(_logging.INFO)
            logger._zfp_initialized = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: Optional[str] = None) -> _logging.Logger:
    """Return a logger under the ``zfp`` root.

    ``get_logger("zfp.core.geometry")`` and ``get_logger("core.geometry")`` return the
    same logger; ``get_logger()`` returns the package root itself.
    """
    root = _root()
    if not name or name == ROOT_LOGGER_NAME:
        return root
    if name.startswith(ROOT_LOGGER_NAME + "."):
        return _logging.getLogger(name)
    return _logging.getLogger("%s.%s" % (ROOT_LOGGER_NAME, name))


def configure(
    level: str = "INFO",
    json: bool = False,
    stream: Optional[IO[str]] = None,
) -> _logging.Logger:
    """Attach a stream handler to the ``zfp`` root logger and return it.

    Args:
        level: Level name or number for the ZFP root logger.
        json: When true, emit one JSON object per record instead of plain text.
        stream: Target stream; defaults to ``sys.stderr``.

    Repeated calls replace the handler installed by the previous call, so configuration
    is idempotent and never duplicates output.
    """
    logger = _root()
    with _lock:
        for handler in list(_installed_handlers):
            logger.removeHandler(handler)
            _installed_handlers.remove(handler)
        handler = _logging.StreamHandler(stream)
        if json:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                _logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
            )
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
        _installed_handlers.append(handler)
        if isinstance(level, str):
            logger.setLevel(_logging.getLevelName(level.upper()))
        else:
            logger.setLevel(level)
    return logger


def reset() -> None:
    """Remove handlers installed by :func:`configure` (used by tests)."""
    logger = _root()
    with _lock:
        for handler in list(_installed_handlers):
            logger.removeHandler(handler)
            _installed_handlers.remove(handler)
        logger.setLevel(_logging.INFO)


class LogContext:
    """Context manager attaching structured fields to records emitted inside it.

    Contexts nest: inner values override outer ones, and the previous state is restored
    on exit even when the block raises. State is per-thread.

    Example:
        >>> with LogContext(document_id="abc", page=3, agent="OcrAgent"):
        ...     get_logger(__name__).info("recognized")
    """

    def __init__(
        self,
        document_id: Optional[str] = None,
        page: Optional[int] = None,
        agent: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.fields: Dict[str, Any] = {
            "document_id": document_id,
            "page": page,
            "agent": agent,
        }
        self.fields.update(extra)
        self._entered = False

    def __enter__(self) -> LogContext:
        _root()
        stack = getattr(_state, "stack", None)
        if stack is None:
            stack = []
            _state.stack = stack
        merged = dict(stack[-1]) if stack else {}
        for key, value in self.fields.items():
            if value is not None:
                merged[key] = value
        stack.append(merged)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._entered:
            stack = getattr(_state, "stack", None)
            if stack:
                stack.pop()
            self._entered = False
        return False


@contextmanager
def log_context(**fields: Any) -> Iterator[Dict[str, Any]]:
    """Functional form of :class:`LogContext`."""
    with LogContext(**fields) as ctx:
        yield ctx.fields
