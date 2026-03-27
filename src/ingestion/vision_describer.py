"""
Vision LLM integration for describing slide/page images.

Used during ingestion of PDF and PPTX files.
The Vision LLM is called only at ingestion time — never at query time.

This module takes a rendered image (PNG) and returns a Japanese text
description of all visual elements: text, charts, diagrams, tables, photos.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.generation.llm_client import call_vision
from src.generation.prompts import VISION_DESCRIBE_SLIDE

logger = logging.getLogger(__name__)


def describe_image(image_path: str | Path, *, max_tokens: int | None = None) -> str:
    """Describe the visual content of a slide or page image.

    Calls the Vision LLM with a structured prompt asking it to describe
    all text, charts, diagrams, and images in the given slide/page.

    Args:
        image_path: Absolute path to a PNG or JPG image.
        max_tokens: Override the default max_tokens from config.

    Returns:
        Japanese text description of the visual content.
        Returns an empty string if the image file does not exist.

    Raises:
        openai.OpenAIError: On API or connection failure.
    """
    p = Path(image_path)
    if not p.exists():
        logger.warning("Image not found, skipping vision description: %s", p)
        return ""

    logger.debug("Calling Vision LLM for: %s", p.name)
    description = call_vision(str(p), VISION_DESCRIBE_SLIDE, max_tokens=max_tokens)
    logger.debug("Vision description length: %d chars", len(description))
    return description


def enrich_chunk_with_vision(
    text_content: str,
    image_path: str | Path | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """Combine extracted text with Vision LLM description of the image.

    If no image_path is given, the original text is returned unchanged.

    Args:
        text_content: Extracted text from the page/slide.
        image_path: Path to the rendered image for this page/slide.
        max_tokens: Override the default Vision LLM max_tokens.

    Returns:
        Combined content string:
            "<extracted text>\n\n[視覚要素]\n<vision description>"
        or just the original text if no image is provided.
    """
    if not image_path:
        return text_content

    description = describe_image(image_path, max_tokens=max_tokens)
    if not description:
        return text_content

    return f"{text_content}\n\n[視覚要素]\n{description}"
