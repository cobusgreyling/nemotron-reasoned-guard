"""Core ReasonedGuard implementation using NVIDIA Nemotron safety reasoning model."""

from __future__ import annotations

import re
from typing import Any

from .client import DEFAULT_MODEL, build_content_messages, get_client
from .models import GuardResult, Policy


class ReasonedGuard:
    """
    High-level guard that uses Nemotron 3.5 Content Safety (with reasoning)
    to evaluate content against natural-language or structured policies.

    The key advantage: you get not just a boolean, but a full, auditable
    reasoning trace explaining *why* the decision was made.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        default_temperature: float = 0.2,
    ):
        self.model = model
        self.client = get_client(api_key=api_key, base_url=base_url)
        self.default_temperature = default_temperature

    def check(
        self,
        text: str,
        policy: str | Policy,
        image: str | None = None,
        context: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> GuardResult:
        """
        Run a reasoned safety check.

        Args:
            text: The text content (user message, model output, tool result, etc.)
            policy: Either a raw policy string or a Policy object
            image: Optional path, URL, or data URL to an image to evaluate jointly
            context: Optional extra context (previous turns, user role, etc.)
            temperature: Lower is more deterministic (0.1-0.3 recommended for safety)
            max_tokens: Max completion length for the reasoning + verdict

        Returns:
            GuardResult with is_safe, categories, full reasoning trace, etc.
        """
        if isinstance(policy, Policy):
            policy_name = policy.name
            policy_text = f"{policy.description}\n\nRules:\n{policy.rules}"
        else:
            policy_name = "custom"
            policy_text = policy

        messages = build_content_messages(
            text=text,
            policy_text=policy_text,
            image=image,
            context=context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        raw = response.choices[0].message.content or ""
        parsed = self._parse_verdict(raw)

        return GuardResult(
            is_safe=parsed["is_safe"],
            confidence=parsed.get("confidence"),
            categories=parsed.get("categories", []),
            reasoning=parsed.get("reasoning", raw),
            policy_name=policy_name,
            input_text=text,
            input_image=image,
            model=self.model,
            raw_response=raw,
        )

    def check_many(
        self,
        items: list[dict[str, Any]],
        policy: str | Policy,
        **kwargs: Any,
    ) -> list[GuardResult]:
        """Convenience batch helper (sequential for now)."""
        return [self.check(**{**item, "policy": policy}, **kwargs) for item in items]

    def _parse_verdict(self, raw: str) -> dict[str, Any]:
        """Robustly extract structured fields from the model's reasoning output."""
        result: dict[str, Any] = {
            "is_safe": True,
            "categories": [],
            "confidence": None,
            "reasoning": raw.strip(),
        }

        # VERDICT
        verdict_match = re.search(r"VERDICT:\s*(SAFE|UNSAFE)", raw, re.IGNORECASE)
        if verdict_match:
            result["is_safe"] = verdict_match.group(1).upper() == "SAFE"

        # CATEGORIES
        cat_match = re.search(r"CATEGORIES:\s*(.+)", raw, re.IGNORECASE)
        if cat_match:
            cats = [c.strip().lower() for c in cat_match.group(1).split(",") if c.strip()]
            result["categories"] = [c for c in cats if c and c != "none"]

        # CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw, re.IGNORECASE)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        # REASONING section (everything after REASONING: or the full thing if not present)
        reasoning_match = re.search(
            r"REASONING:\s*(.+)", raw, re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Fallback: use everything after the structured fields as reasoning
            # Remove the verdict lines from the top if present
            cleaned = re.sub(
                r"(VERDICT:|CATEGORIES:|CONFIDENCE:).*\n?", "", raw, flags=re.IGNORECASE
            ).strip()
            result["reasoning"] = cleaned or raw.strip()

        return result
