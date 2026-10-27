"""Typed provider errors.

cubepi providers catch raw SDK exceptions and re-raise as one of these
subclasses, so downstream callers can ``except cubeloop.errors.X`` instead
of pattern-matching on SDK-specific strings or status codes.

Adding a new subclass: add it at the bottom of the file (never reorder)
and re-export from ``cubepi/__init__.py`` and ``__all__``.
"""

from __future__ import annotations

import math
import re
from typing import NoReturn, Protocol

from cubeloop.providers.base import Message


class _ClassifyTarget(Protocol):
    """Structural type for the `model` parameter of `classify_and_raise`.

    Both `cubeloop.providers.base.Model` and
    `cubeloop.providers.images.types.ImagesModel` satisfy it; `context_window`
    is read with a getattr fallback so image models (which don't have it)
    skip the token-budget heuristic cleanly.
    """

    id: str
    provider_id: str


class ProviderError(Exception):
    """Base class for typed cubepi provider errors.

    Always carries provider / model context. ``raw_exception`` is the
    original SDK exception (kept on ``__cause__`` via ``raise … from``).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message or self.__class__.__name__)


class ContextLengthExceeded(ProviderError):
    """The request exceeded the model's context window.

    ``tokens_in`` is an estimate (chars/4) of the prompt size at the time
    of the failure; ``context_window`` is the model's advertised window.
    Both may be None if the provider couldn't measure them.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        tokens_in: int | None = None,
        context_window: int | None = None,
        error_code: str | None = None,
    ) -> None:
        self.tokens_in = tokens_in
        self.context_window = context_window
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            error_code=error_code,
        )


class RateLimited(ProviderError):
    """Provider rate-limit / quota error.

    ``retry_after`` is the recommended retry delay in seconds, parsed from
    the SDK response when available.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        error_code: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            error_code=error_code,
        )


class ProviderAuthFailed(ProviderError):
    """API-key invalid, account suspended, or 401/403 with no quota wording."""


class ProviderUnavailable(ProviderError):
    """5xx, timeout, or connection failure.

    On chain exhaustion ``errors`` holds the per-leg failures (typed when
    available). ``__cause__`` is the last typed error when one exists.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
        errors: list[BaseException] | None = None,
    ) -> None:
        self.errors: list[BaseException] = list(errors) if errors else []
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            error_code=error_code,
        )


class ProviderBadRequest(ProviderError):
    """Residual 4xx provider error (schema rejection, uncertain vendor 400s)."""


class ModelNotFound(ProviderBadRequest):
    """The requested model id is unknown to this provider (typically 404)."""


class ContentFiltered(ProviderBadRequest):
    """Provider refused the request for safety / content-policy reasons."""


class EmptyCompletion(ProviderError):
    """Provider returned a natural stop with no text content and no tool calls.

    Typically caused by an upstream model generation glitch, transient safety
    filter block, or premature EOS token emission without producing tokens.
    """


EmptyCompletionError = EmptyCompletion


# ---------------------------------------------------------------------------
# Heuristics: turn a raw SDK exception into one of the typed errors above.
# ---------------------------------------------------------------------------

_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context.{0,10}length.{0,20}exceed", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"reduce.{0,10}messages", re.IGNORECASE),
)

_RATE_LIMIT_PATTERNS = (
    re.compile(r"rate ?limit", re.IGNORECASE),
    re.compile(r"quota (?:exceed|exhaust|limit|reach)", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
)

_MODEL_NOT_FOUND_CODES = frozenset(
    {
        "model_not_found",
        "model_not_available",
        "invalid_model",
    }
)

_MODEL_NOT_FOUND_PATTERNS = (
    re.compile(r"model[_ ]not[_ ]found", re.IGNORECASE),
    re.compile(r"unknown model", re.IGNORECASE),
    re.compile(r"invalid model", re.IGNORECASE),
    re.compile(r"model .+ does not exist", re.IGNORECASE),
    re.compile(r"does not exist.+model", re.IGNORECASE),
)

_CONTENT_FILTER_PATTERNS = (
    re.compile(r"content[_ ]policy", re.IGNORECASE),
    re.compile(r"content[_ ]filter", re.IGNORECASE),
    re.compile(r"responsible[_ ]ai", re.IGNORECASE),
    re.compile(r"safety[_ ]system", re.IGNORECASE),
    re.compile(r"refused to (?:answer|respond)", re.IGNORECASE),
)

_SPECIFIC_ERROR_CODES = frozenset(
    {
        "model_not_found",
        "model_not_available",
        "invalid_model",
        "content_filter",
        "content_policy_violation",
        "responsible_ai_policy_violation",
    }
)


def _status_of(exc: BaseException) -> int | None:
    """Best-effort status code extraction from an SDK exception."""

    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        rc = getattr(resp, "status_code", None)
        if isinstance(rc, int):
            return rc
    return None


def _estimate_input_tokens(messages: list[Message] | None) -> int | None:
    """Rough chars/4 estimate over message text, used only for diagnostics.

    Returns None when ``messages`` is empty or missing.
    """

    if not messages:
        return None
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    total += len(text)
                else:
                    text = getattr(block, "content", None)
                    if isinstance(text, str):
                        total += len(text)
        elif isinstance(content, str):
            total += len(content)
    if total == 0:
        return None
    return max(1, math.ceil(total / 4))


def _as_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    return None


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _vendor_error_parts(
    exc: BaseException,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(code, type, param)`` from a vendor SDK exception."""

    body = getattr(exc, "body", None)
    err = getattr(exc, "error", None)
    nested = None
    if isinstance(err, dict):
        nested = err.get("error") if isinstance(err.get("error"), dict) else err
    elif err is not None:
        nested = {
            "code": getattr(err, "code", None),
            "type": getattr(err, "type", None),
            "param": getattr(err, "param", None),
        }

    body_map = _as_mapping(body)
    body_error = _as_mapping(body_map.get("error")) if body_map else None
    err_map = _as_mapping(err)
    nested_map = _as_mapping(nested)

    code = _first_str(
        getattr(exc, "code", None),
        err_map.get("code") if err_map else None,
        nested_map.get("code") if nested_map else None,
        body_error.get("code") if body_error else None,
        body_map.get("code") if body_map else None,
    )
    typ = _first_str(
        err_map.get("type") if err_map else None,
        nested_map.get("type") if nested_map else None,
        body_error.get("type") if body_error else None,
        getattr(err, "type", None)
        if err is not None and not isinstance(err, dict)
        else None,
    )
    param = _first_str(
        err_map.get("param") if err_map else None,
        nested_map.get("param") if nested_map else None,
        body_error.get("param") if body_error else None,
    )
    return (
        code.lower() if code else None,
        typ.lower() if typ else None,
        param.lower() if param else None,
    )


def extract_error_code(exc: BaseException) -> str | None:
    """Best-effort vendor ``error_code`` from an SDK exception.

    Priority (OpenAI / OpenAI-compatible first, then Anthropic ``type``):

    1. ``error.code`` / nested ``error.error.code`` / ``body.error.code``
    2. ``error.type`` (Anthropic) / ``type``
    3. ``param`` only when it identifies the model
    4. Never the full human message
    """

    code, typ, param = _vendor_error_parts(exc)
    if code:
        return code
    if typ and typ in _SPECIFIC_ERROR_CODES:
        return typ
    if param == "model" and typ:
        return typ
    if typ and typ not in {
        "invalid_request_error",
        "api_error",
        "error",
        "not_found_error",
        "not_found",
    }:
        return typ
    return None


def _looks_like_model_not_found(
    *,
    status: int | None,
    code: str | None,
    msg: str,
    param: str | None = None,
) -> bool:
    if code in _MODEL_NOT_FOUND_CODES:
        return True
    if param == "model":
        return True
    modelish = any(pat.search(msg) for pat in _MODEL_NOT_FOUND_PATTERNS)
    if code in {"not_found_error", "not_found"} and modelish:
        return True
    if status in {400, 404} and modelish:
        return True
    return False


def _looks_like_content_filtered(*, code: str | None, msg: str) -> bool:
    if code in {
        "content_filter",
        "content_policy_violation",
        "responsible_ai_policy_violation",
    }:
        return True
    return any(pat.search(msg) for pat in _CONTENT_FILTER_PATTERNS)


def classify_string_error(
    message: str,
    *,
    model: _ClassifyTarget | None = None,
    status_code: int | None = None,
    error_code: str | None = None,
    retry_after: float | None = None,
    tokens_in: int | None = None,
    context_window: int | None = None,
) -> ProviderError:
    """Classify a string-only stream / generate error into a typed error."""

    provider = getattr(model, "provider_id", None) if model is not None else None
    model_id = getattr(model, "id", None) if model is not None else None
    msg = message or "provider error"
    code = error_code.lower() if isinstance(error_code, str) else None
    cw = context_window if context_window else getattr(model, "context_window", None)

    for pat in _CONTEXT_LENGTH_PATTERNS:
        if pat.search(msg):
            return ContextLengthExceeded(
                msg,
                provider=provider,
                model=model_id,
                status_code=status_code,
                tokens_in=tokens_in,
                context_window=cw if cw else None,
                error_code=code,
            )

    if status_code == 429 or any(pat.search(msg) for pat in _RATE_LIMIT_PATTERNS):
        return RateLimited(
            msg,
            provider=provider,
            model=model_id,
            status_code=status_code or 429,
            retry_after=retry_after,
            error_code=code,
        )

    if status_code in (401, 403) or (code and "auth" in code):
        return ProviderAuthFailed(
            msg,
            provider=provider,
            model=model_id,
            status_code=status_code,
            error_code=code,
        )

    if _looks_like_content_filtered(code=code, msg=msg):
        return ContentFiltered(
            msg,
            provider=provider,
            model=model_id,
            status_code=status_code,
            error_code=code,
        )

    if _looks_like_model_not_found(status=status_code, code=code, msg=msg):
        return ModelNotFound(
            msg,
            provider=provider,
            model=model_id,
            status_code=status_code,
            error_code=code,
        )

    if status_code is not None and 500 <= status_code < 600:
        return ProviderUnavailable(
            msg,
            provider=provider,
            model=model_id,
            status_code=status_code,
            error_code=code,
        )

    return ProviderBadRequest(
        msg,
        provider=provider,
        model=model_id,
        status_code=status_code,
        error_code=code,
    )


def error_from_stream_fields(
    *,
    error_message: str | None,
    error_type: str | None = None,
    error_code: str | None = None,
    status_code: int | None = None,
    retry_after: float | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
    tokens_in: int | None = None,
    context_window: int | None = None,
) -> ProviderError:
    """Rebuild a typed error from structured stream / message fields."""

    type_map: dict[str, type[ProviderError]] = {
        "RateLimited": RateLimited,
        "ProviderUnavailable": ProviderUnavailable,
        "ContextLengthExceeded": ContextLengthExceeded,
        "ProviderAuthFailed": ProviderAuthFailed,
        "ProviderBadRequest": ProviderBadRequest,
        "ModelNotFound": ModelNotFound,
        "ContentFiltered": ContentFiltered,
        "EmptyCompletion": EmptyCompletion,
        "EmptyCompletionError": EmptyCompletion,
    }
    cls = type_map.get(error_type or "")
    msg = error_message or (error_type or "provider error")
    kwargs: dict[str, object] = {
        "provider": provider_id,
        "model": model_id,
        "status_code": status_code,
        "error_code": error_code,
    }
    if cls is RateLimited:
        return RateLimited(msg, retry_after=retry_after, **kwargs)  # type: ignore[arg-type]
    if cls is ContextLengthExceeded:
        return ContextLengthExceeded(
            msg,
            tokens_in=tokens_in,
            context_window=context_window,
            **kwargs,  # type: ignore[arg-type]
        )
    if cls is not None:
        return cls(msg, **kwargs)  # type: ignore[arg-type]
    target = None
    if provider_id or model_id:
        target = type(
            "_T",
            (),
            {
                "id": model_id or "",
                "provider_id": provider_id or "",
                "context_window": context_window,
            },
        )()
    return classify_string_error(
        msg,
        model=target,
        status_code=status_code,
        error_code=error_code,
        retry_after=retry_after,
        tokens_in=tokens_in,
        context_window=context_window,
    )


def annotate_error_event(
    exc: BaseException,
    *,
    fallback_message: str | None = None,
) -> dict[str, object]:
    """Fields to stamp on ``StreamEvent`` / ``AssistantMessage`` for a failure."""

    if isinstance(exc, ProviderError):
        fields: dict[str, object] = {
            "error_message": fallback_message or str(exc),
            "error_type": type(exc).__name__,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "provider_id": exc.provider or "",
            "model_id": exc.model or "",
        }
        if isinstance(exc, RateLimited):
            fields["retry_after"] = exc.retry_after
        if isinstance(exc, ContextLengthExceeded):
            fields["tokens_in"] = exc.tokens_in
            fields["context_window"] = exc.context_window
        return fields
    return {
        "error_message": fallback_message or str(exc),
        "error_type": None,
        "error_code": extract_error_code(exc),
        "status_code": _status_of(exc),
        "retry_after": _retry_after_from(exc),
        "provider_id": "",
        "model_id": "",
    }


def _retry_after_from(exc: BaseException) -> float | None:
    """Pull retry-after seconds from an SDK exception's response headers."""

    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def classify_and_raise(
    exc: BaseException,
    *,
    model: _ClassifyTarget,
    messages: list[Message] | None = None,
) -> NoReturn:
    """Inspect a raw SDK exception and raise the typed cubepi error.

    Heuristics (first match wins):
      1. Explicit context-length wording → ContextLengthExceeded.
      2. status==400 + estimated tokens_in within 5% of model.context_window
         → ContextLengthExceeded (covers Volcano ARK's opaque
         ``InvalidParameter`` 400).
      3. status==429 OR quota/rate-limit wording → RateLimited.
      4. status==401 / 403 → ProviderAuthFailed.
      5. TimeoutError / ConnectionError / 5xx → ProviderUnavailable.
      6. Any other 4xx → ProviderBadRequest.
      7. Else: re-raise the original (caller should let it propagate).

    ``raise classify_and_raise(...) from exc`` is the idiomatic call site.
    """

    # Already typed — re-raise unchanged so callers that catch ProviderError
    # don't get double-wrapped.
    if isinstance(exc, ProviderError):
        raise exc

    msg = str(exc) or getattr(exc, "message", "")
    status = _status_of(exc)
    provider = model.provider_id
    model_id = model.id
    error_code, _err_type, err_param = _vendor_error_parts(exc)
    if error_code is None:
        error_code = extract_error_code(exc)

    tokens_in = _estimate_input_tokens(messages)
    cw_val = getattr(model, "context_window", None)
    context_window = cw_val if cw_val else None

    for pat in _CONTEXT_LENGTH_PATTERNS:
        if pat.search(msg):
            raise ContextLengthExceeded(
                msg,
                provider=provider,
                model=model_id,
                status_code=status,
                tokens_in=tokens_in,
                context_window=context_window,
                error_code=error_code,
            ) from exc

    if (
        status == 400
        and tokens_in is not None
        and context_window is not None
        and tokens_in >= int(context_window * 0.95)
    ):
        raise ContextLengthExceeded(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            tokens_in=tokens_in,
            context_window=context_window,
            error_code=error_code,
        ) from exc

    if status == 429 or any(pat.search(msg) for pat in _RATE_LIMIT_PATTERNS):
        raise RateLimited(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            retry_after=_retry_after_from(exc),
            error_code=error_code,
        ) from exc

    if status in (401, 403):
        raise ProviderAuthFailed(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    if _looks_like_content_filtered(code=error_code, msg=msg):
        raise ContentFiltered(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    if isinstance(exc, (TimeoutError, ConnectionError)):
        raise ProviderUnavailable(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    # SDK-specific connection/timeout errors (openai.APIConnectionError,
    # anthropic.APITimeoutError) don't inherit from Python's built-in
    # ConnectionError/TimeoutError, so the isinstance above misses them.
    cls_name = type(exc).__name__
    if "ConnectionError" in cls_name or "Timeout" in cls_name:
        raise ProviderUnavailable(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    if status is not None and 500 <= status < 600:
        raise ProviderUnavailable(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    if _looks_like_model_not_found(
        status=status, code=error_code, msg=msg, param=err_param
    ):
        raise ModelNotFound(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    if status is not None and 400 <= status < 500:
        raise ProviderBadRequest(
            msg,
            provider=provider,
            model=model_id,
            status_code=status,
            error_code=error_code,
        ) from exc

    # Unknown → let original propagate. Callers might still need to handle it.
    raise exc


__all__ = [
    "ProviderError",
    "ContextLengthExceeded",
    "RateLimited",
    "ProviderAuthFailed",
    "ProviderUnavailable",
    "ProviderBadRequest",
    "ModelNotFound",
    "ContentFiltered",
    "EmptyCompletion",
    "EmptyCompletionError",
    "classify_and_raise",
    "classify_string_error",
    "extract_error_code",
    "error_from_stream_fields",
    "annotate_error_event",
]
