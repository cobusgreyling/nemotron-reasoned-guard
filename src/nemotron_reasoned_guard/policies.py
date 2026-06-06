"""Policy loading and reusable policy pack utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Policy


def load_policy(path: str | Path) -> Policy:
    """Load a single policy from a YAML file."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return Policy(**data)
    raise ValueError(f"Policy file {p} must contain a mapping")


def load_policy_pack(directory: str | Path) -> dict[str, Policy]:
    """Load all .yaml / .yml files in a directory as a named policy pack."""
    pack: dict[str, Policy] = {}
    for f in Path(directory).glob("*.y*ml"):
        policy = load_policy(f)
        pack[policy.name] = policy
    return pack


# Built-in minimal default policies (can be overridden by files)
DEFAULT_POLICIES: dict[str, Policy] = {
    "corporate": Policy(
        name="corporate",
        description="Corporate communication and brand safety policy",
        rules=(
            "Do not discuss confidential company information, unreleased products, or internal strategy.\n"
            "Do not provide financial guidance, pricing promises, or legal advice.\n"
            "Reject requests for assistance with fraud, scams, or unethical business practices.\n"
            "Be helpful but stay within public, non-sensitive topics."
        ),
        version="1.0",
    ),
    "kids": Policy(
        name="kids",
        description="Child-safe content policy (strict)",
        rules=(
            "Block any sexual, violent, hateful, or self-harm content.\n"
            "Block any content that could be used to groom, exploit, or endanger minors.\n"
            "Block discussions of weapons, illegal drugs, or dangerous activities presented as fun or instructional.\n"
            "When in doubt, err on the side of blocking."
        ),
        version="1.0",
    ),
    "medical": Policy(
        name="medical",
        description="Medical and health advice policy",
        rules=(
            "Do not provide personalized medical, psychiatric, or diagnostic advice.\n"
            "Do not suggest treatments, medications, or dosages.\n"
            "General public information about conditions is acceptable, but always include a disclaimer "
            "to consult a qualified healthcare professional.\n"
            "Immediately flag any crisis language (suicidal ideation, self-harm) and direct to professional help resources."
        ),
        version="1.0",
    ),
}
