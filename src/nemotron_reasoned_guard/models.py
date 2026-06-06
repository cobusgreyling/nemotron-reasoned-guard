"""Pydantic models for nemotron-reasoned-guard."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Policy(BaseModel):
    """A reusable policy definition."""

    name: str = Field(..., description="Short name for the policy (e.g. 'corporate', 'kids')")
    description: str = Field(..., description="Human-readable description of what the policy covers")
    rules: str = Field(
        ...,
        description="Detailed natural language rules. The Nemotron safety model will reason against these.",
    )
    version: str = Field(default="1.0", description="Policy version for auditability")


class GuardResult(BaseModel):
    """Structured result from a reasoned safety check."""

    is_safe: bool = Field(..., description="Final safety decision after policy reasoning")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Model-reported confidence if available"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Violation categories flagged (e.g. 'violence', 'hate', 'self-harm', 'custom:medical-advice')",
    )
    reasoning: str = Field(
        ...,
        description="Full reasoning trace from the Nemotron model explaining the decision. This is the key value for audit and trust.",
    )
    policy_name: str = Field(..., description="Name of the policy that was evaluated against")
    input_text: str = Field(..., description="The text content that was checked")
    input_image: str | None = Field(
        default=None, description="Optional image reference (path, url, or base64 prefix)"
    )
    model: str = Field(
        default="nvidia/nemotron-3.5-content-safety",
        description="NIM model used for the check",
    )
    raw_response: str | None = Field(
        default=None, description="Raw model output for debugging / advanced parsing"
    )

    def to_audit_dict(self) -> dict:
        """Return a clean dict suitable for logging / JSONL audit trails."""
        return {
            "is_safe": self.is_safe,
            "confidence": self.confidence,
            "categories": self.categories,
            "policy": self.policy_name,
            "model": self.model,
            "reasoning_preview": self.reasoning[:500] + "..." if len(self.reasoning) > 500 else self.reasoning,
            "input_text_preview": self.input_text[:200] + "..." if len(self.input_text) > 200 else self.input_text,
        }


class CheckRequest(BaseModel):
    """Internal request model (used by FastAPI example etc)."""

    text: str
    policy: str | Policy
    image: str | None = None  # path, url, or base64 data URL
    context: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
