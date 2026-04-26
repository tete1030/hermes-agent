"""OpenAI image generation backend.

Primary model is ``gpt-image-2`` with optional ``quality`` control
(``low``/``medium``/``high``).

Selection precedence for model (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml``
4. :data:`DEFAULT_MODEL` — ``gpt-image-2``

Quality resolution (only when model is ``gpt-image-2``):

1. Tool arg ``quality``
2. ``image_gen.openai.quality`` in ``config.yaml``
3. ``image_gen.quality`` in ``config.yaml``
4. :data:`DEFAULT_QUALITY` — ``medium``
"""

from __future__ import annotations

from contextlib import ExitStack
import logging
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


def _load_openai_config() -> Dict[str, Any]:
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

    cfg = cfg or _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
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
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}

    raw: Any = override
    if raw is None and isinstance(openai_cfg, dict):
        raw = openai_cfg.get("quality")
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


def _normalize_local_attachments(value: Any) -> Tuple[List[Path], Optional[str]]:
    """Validate attachment inputs for OpenAI images.edit().

    OpenAI's edit endpoint expects image files, so this provider accepts
    local file paths only.
    """
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], "attachments must be an array of strings"

    paths: List[Path] = []
    for idx, raw in enumerate(value, start=1):
        if not isinstance(raw, str) or not raw.strip():
            return [], f"attachments[{idx}] must be a non-empty string"
        ref = raw.strip()
        lowered = ref.lower()
        if lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("data:image/"):
            return [], "OpenAI provider only accepts local file path attachments"

        path = Path(ref).expanduser()
        if not path.is_file():
            return [], f"Attachment not found: {ref}"
        paths.append(path)

    return paths, None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 at low/medium/high."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
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
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 with optional low/medium/high quality",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
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
                provider="openai",
                aspect_ratio=aspect,
            )

        attachments, attachment_err = _normalize_local_attachments(kwargs.get("attachments"))
        if attachment_err:
            return error_response(
                error=attachment_err,
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        if not os.environ.get("OPENAI_API_KEY"):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        cfg = _load_openai_config()
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
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        size = _SIZES.get(aspect, _SIZES["square"])

        try:
            client = openai.OpenAI()
            if attachments:
                with ExitStack() as stack:
                    images = [stack.enter_context(path.open("rb")) for path in attachments]
                    payload: Dict[str, Any] = {
                        "model": model_id,
                        "image": images,
                        "prompt": prompt,
                        "size": size,
                        "n": 1,
                    }
                    if quality is not None:
                        payload["quality"] = quality
                    response = client.images.edit(**payload)
            else:
                # gpt-image-2 returns b64_json unconditionally and REJECTS
                # ``response_format`` as an unknown parameter. Don't send it.
                payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "size": size,
                    "n": 1,
                }
                if quality is not None:
                    payload["quality"] = quality
                response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_{model_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64 today, but fall back
            # gracefully if the API ever changes.
            image_ref = url
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if quality is not None:
            extra["quality"] = quality
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
