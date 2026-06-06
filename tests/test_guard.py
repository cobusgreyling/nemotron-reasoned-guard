"""Basic smoke tests for nemotron-reasoned-guard (no live API calls)."""

from nemotron_reasoned_guard.guard import ReasonedGuard
from nemotron_reasoned_guard.models import GuardResult, Policy
from nemotron_reasoned_guard.policies import DEFAULT_POLICIES


def test_parse_verdict_safe():
    guard = ReasonedGuard.__new__(ReasonedGuard)  # avoid __init__ / API key
    raw = """VERDICT: SAFE
CATEGORIES: none
CONFIDENCE: 0.92
REASONING: The content is a normal question about Python programming and does not violate any part of the policy."""

    parsed = guard._parse_verdict(raw)
    assert parsed["is_safe"] is True
    assert parsed["categories"] == []
    assert parsed["confidence"] == 0.92
    assert "Python programming" in parsed["reasoning"]


def test_parse_verdict_unsafe():
    guard = ReasonedGuard.__new__(ReasonedGuard)
    raw = """After careful analysis...

VERDICT: UNSAFE
CATEGORIES: fraud, social-engineering
CONFIDENCE: 0.85
REASONING: The user is explicitly asking for help crafting a phishing email. This directly violates the corporate policy against assisting with fraud and social engineering."""

    parsed = guard._parse_verdict(raw)
    assert parsed["is_safe"] is False
    assert "fraud" in parsed["categories"]
    assert "phishing" in parsed["reasoning"].lower()


def test_guard_result_model():
    res = GuardResult(
        is_safe=False,
        categories=["hate"],
        reasoning="This is hateful content according to the policy.",
        policy_name="kids",
        input_text="some bad text",
    )
    assert res.is_safe is False
    audit = res.to_audit_dict()
    assert audit["is_safe"] is False
    assert audit["policy"] == "kids"


def test_default_policies_exist():
    assert "corporate" in DEFAULT_POLICIES
    assert "kids" in DEFAULT_POLICIES
    assert "medical" in DEFAULT_POLICIES

    pol = DEFAULT_POLICIES["corporate"]
    assert isinstance(pol, Policy)
    assert len(pol.rules) > 20
