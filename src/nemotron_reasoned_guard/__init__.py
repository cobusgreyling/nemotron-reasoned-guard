"""nemotron-reasoned-guard

Policy-as-code guardrails powered by NVIDIA Nemotron 3.5 Content Safety
with full reasoning traces for auditability and trust.
"""

from .guard import ReasonedGuard
from .models import GuardResult, Policy

__all__ = ["ReasonedGuard", "GuardResult", "Policy"]
__version__ = "0.1.0"
