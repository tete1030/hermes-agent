"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Uses the same model/quality contract as the ``openai`` plugin:
- primary model: ``gpt-image-2``
- optional quality: ``low`` / ``medium`` / ``high`` (gpt-image-2 only)

Selection precedence for model (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml``
4. :data:`DEFAULT_MODEL` — ``gpt-image-2``

Quality resolution (only when model is ``gpt-image-2``):

1. Tool arg ``quality``
2. ``image_gen.openai-codex.quality`` in ``config.yaml``
3. ``image_gen.quality`` in ``config.yaml``
4. :data:`DEFAULT_QUALITY` — ``medium``

Output is saved as PNG under ``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"
VALID_QUALITIES = {"low", "medium", "high"}
DEFAULT_MODEL = API_MODEL
DEFAULT_QUALITY = "medium"

_MODEL_META: Dict[str, Dict[str, Any]] = {
    API_MODEL: {
        "display": "GPT Image 2",
        "speed": "varies",
        "strengths": "General image generation with optional quality tuning",
    }
}

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.4"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Resolve model id with minimal validation.

    Any non-empty string is accepted so advanced users can try unreleased or
    proxy-specific model IDs.
    """
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if isinstance(env_override, str) and env_override.strip():
        return env_override.strip()

    cfg = cfg or _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value.strip():
            return value.strip()

    top = cfg.get("model")
    if isinstance(top, str) and top.strip():
        return top.strip()

    return DEFAULT_MODEL


def _resolve_quality(
    cfg: Dict[str, Any],
    *,
    model: str,
    override: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve quality for gpt-image-2; reject quality usage on other models."""
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}

    raw: Any = override
    if raw is None and isinstance(sub, dict):
        raw = sub.get("quality")
    if raw is None:
        raw = cfg.get("quality")

    if raw is None:
        return (DEFAULT_QUALITY if model == API_MODEL else None), None

    if not isinstance(raw, str) or not raw.strip():
        return None, "quality must be one of: low, medium, high"

    quality = raw.strip().lower()
    if quality not in VALID_QUALITIES:
        return None, "quality must be one of: low, medium, high"

    if model != API_MODEL:
        return None, "quality is only supported when model is 'gpt-image-2'"

    return quality, None


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _resolve_endpoint_auth() -> Dict[str, Any]:
    """Resolve endpoint/auth details for image generation requests.

    Default behavior remains Codex OAuth against ``_CODEX_BASE_URL``.
    If ``image_gen.openai-codex.base_url`` is set to a non-Codex URL, we treat
    it as a custom OpenAI-compatible endpoint (e.g. LiteLLM) and use
    ``image_gen.openai-codex.api_key`` (fallback: ``OPENAI_API_KEY``).
    """
    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}

    raw_base_url = ""
    raw_api_key = ""
    if isinstance(sub, dict):
        base_val = sub.get("base_url")
        if isinstance(base_val, str):
            raw_base_url = base_val.strip()
        key_val = sub.get("api_key")
        if isinstance(key_val, str):
            raw_api_key = key_val.strip()

    normalized_codex_base = _CODEX_BASE_URL.rstrip("/")
    base_url = (raw_base_url or _CODEX_BASE_URL).strip().rstrip("/")
    custom_endpoint = bool(raw_base_url) and base_url != normalized_codex_base

    if custom_endpoint:
        api_key = raw_api_key or os.environ.get("OPENAI_API_KEY", "").strip() or ""
        return {
            "mode": "custom_endpoint",
            "base_url": base_url,
            "api_key": api_key,
            "default_headers": None,
        }

    token = _read_codex_access_token() or ""
    headers = None
    if token:
        try:
            from agent.auxiliary_client import _codex_cloudflare_headers

            headers = _codex_cloudflare_headers(token)
        except Exception as exc:
            logger.debug("Could not build Codex cloudflare headers: %s", exc)

    return {
        "mode": "codex_oauth",
        "base_url": normalized_codex_base,
        "api_key": token,
        "default_headers": headers,
    }


def _build_codex_client():
    """Return an OpenAI client for the resolved endpoint/auth settings, or None."""
    settings = _resolve_endpoint_auth()
    api_key = settings.get("api_key")
    base_url = settings.get("base_url")
    if not api_key or not base_url:
        return None

    try:
        import openai

        kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        headers = settings.get("default_headers")
        if isinstance(headers, dict) and headers:
            kwargs["default_headers"] = headers
        return openai.OpenAI(**kwargs)
    except Exception as exc:
        logger.debug("Could not build Codex image client: %s", exc)
        return None


def _normalize_local_attachments(value: Any) -> Tuple[List[Path], Optional[str]]:
    """Validate Codex attachments and keep the local-path-only contract in one place."""
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], "attachments must be an array of strings"

    normalized: List[Path] = []
    for idx, raw in enumerate(value, start=1):
        if not isinstance(raw, str) or not raw.strip():
            return [], f"attachments[{idx}] must be a non-empty string"

        ref = raw.strip()
        lowered = ref.lower()
        if lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("data:image/"):
            return [], "OpenAI-Codex provider only accepts local file path attachments"

        path = Path(ref).expanduser()
        if not path.is_file():
            return [], f"Attachment not found: {ref}"

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            return [], f"Attachment is not an image file: {ref}"

        normalized.append(path)

    return normalized, None


def _attachment_to_input_image_url(path: Path) -> str:
    """Convert a validated local image path into a data URL for Responses input."""
    mime, _ = mimetypes.guess_type(str(path))
    # Attachment MIME was validated in _normalize_local_attachments.
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_responses_input_content(prompt: str, attachments: List[Path]) -> List[Dict[str, Any]]:
    """Build Responses input content with text + optional input_image parts."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for path in attachments:
        content.append({"type": "input_image", "image_url": _attachment_to_input_image_url(path)})
    return content


def _collect_image_b64(
    client: Any,
    *,
    prompt: str,
    model: str,
    size: str,
    quality: Optional[str],
    attachments: Optional[List[Path]] = None,
) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    image_b64: Optional[str] = None
    content = _build_responses_input_content(prompt, attachments or [])

    tool_payload: Dict[str, Any] = {
        "type": "image_generation",
        "model": model,
        "size": size,
        "output_format": "png",
        "background": "opaque",
        "partial_images": 1,
    }
    if quality is not None:
        tool_payload["quality"] = quality

    with client.responses.stream(
        model=_CODEX_CHAT_MODEL,
        store=False,
        instructions=_CODEX_INSTRUCTIONS,
        input=[{
            "type": "message",
            "role": "user",
            "content": content,
        }],
        tools=[tool_payload],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "image_generation_call":
                    result = getattr(item, "result", None)
                    if isinstance(result, str) and result:
                        image_b64 = result
            elif event_type == "response.image_generation_call.partial_image":
                partial = getattr(event, "partial_image_b64", None)
                if isinstance(partial, str) and partial:
                    image_b64 = partial
        final = stream.get_final_response()

    # Final-response sweep covers the case where the stream finished before
    # we observed the ``output_item.done`` event for the image call.
    for item in getattr(final, "output", None) or []:
        if getattr(item, "type", None) == "image_generation_call":
            result = getattr(item, "result", None)
            if isinstance(result, str) and result:
                image_b64 = result

    return image_b64


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        settings = _resolve_endpoint_auth()
        if not settings.get("api_key"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODEL_META.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 with optional low/medium/high quality via Codex auth",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        attachments, attachment_err = _normalize_local_attachments(kwargs.get("attachments"))
        if attachment_err:
            return error_response(
                error=attachment_err,
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        settings = _resolve_endpoint_auth()
        if not settings.get("api_key"):
            if settings.get("mode") == "custom_endpoint":
                msg = (
                    "image_gen.openai-codex.base_url is configured but no API key "
                    "was found. Set image_gen.openai-codex.api_key or "
                    "OPENAI_API_KEY."
                )
            else:
                msg = (
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                )
            return error_response(
                error=msg,
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import openai  # noqa: F401
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        cfg = _load_image_gen_config()
        model_id = _resolve_model(cfg)
        quality, quality_err = _resolve_quality(
            cfg,
            model=model_id,
            override=kwargs.get("quality"),
        )
        if quality_err:
            return error_response(
                error=quality_err,
                error_type="invalid_argument",
                provider="openai-codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        size = _SIZES.get(aspect, _SIZES["square"])

        client = _build_codex_client()
        if client is None:
            return error_response(
                error="Could not initialize Codex image client",
                error_type="auth_required",
                provider="openai-codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            b64 = _collect_image_b64(
                client,
                prompt=prompt,
                model=model_id,
                size=size,
                quality=quality,
                attachments=attachments,
            )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation request failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{model_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if quality is not None:
            extra["quality"] = quality

        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())
