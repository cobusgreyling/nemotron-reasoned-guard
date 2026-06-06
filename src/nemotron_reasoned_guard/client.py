"""Thin wrapper around the NVIDIA NIM OpenAI-compatible endpoint for Nemotron safety models."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

DEFAULT_MODEL = os.getenv("NEMOTRON_SAFETY_MODEL", "nvidia/nemotron-3.5-content-safety")
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Return a configured OpenAI client pointing at NVIDIA NIM."""
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY not found. Set it in environment or pass explicitly. "
            "Get a free key at https://build.nvidia.com/"
        )
    return OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=key,
    )


def encode_image_to_data_url(image_path: str | Path) -> str:
    """Convert a local image file to a data URL for the API."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Basic validation + resize if huge (safety models don't need 4K)
    with Image.open(path) as img:
        if max(img.size) > 2048:
            img.thumbnail((2048, 2048))
            # Re-save to a temp buffer would be ideal; for simplicity we just use original
            # In production you'd resize and re-encode.

    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_content_messages(
    text: str,
    policy_text: str,
    image: str | None = None,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Build the messages list for the safety model, supporting optional image."""
    system_prompt = (
        "You are Nemotron Content Safety, an expert multilingual multimodal safety reasoner. "
        "Your job is to carefully evaluate whether the provided content violates the given policy. "
        "You must think step by step, consider context, intent, severity, and edge cases. "
        "Be strict on clear violations but reasonable on ambiguous cases. "
        "Always produce a clear final verdict."
    )

    policy_block = f"POLICY:\n{policy_text.strip()}\n"

    user_parts: list[dict[str, Any]] = [{"type": "text", "text": policy_block}]

    if context:
        user_parts.append({"type": "text", "text": f"ADDITIONAL CONTEXT:\n{context.strip()}\n"})

    user_parts.append({"type": "text", "text": f"CONTENT TO EVALUATE:\n{text.strip()}"})

    if image:
        # Support both data URLs and raw http(s) URLs
        if image.startswith("data:") or image.startswith("http"):
            image_url = image
        else:
            # Assume local path
            image_url = encode_image_to_data_url(image)

        user_parts.append({"type": "image_url", "image_url": {"url": image_url}})

    user_parts.append({
        "type": "text",
        "text": (
            "\n\nThink carefully step by step about the policy and the content (including any image). "
            "After your full reasoning, output EXACTLY in this format (no extra commentary after):\n\n"
            "VERDICT: SAFE or UNSAFE\n"
            "CATEGORIES: comma-separated list of violated categories (or 'none')\n"
            "CONFIDENCE: 0.0 to 1.0\n"
            "REASONING: [your complete step-by-step reasoning here]"
        ),
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]
