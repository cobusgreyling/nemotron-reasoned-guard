"""
FastAPI integration example for nemotron-reasoned-guard.

Shows two patterns:
1. A dedicated /guard endpoint you can call from anywhere.
2. A simple dependency / middleware helper you can drop into existing routes
   to automatically guard user input or model output before it is processed/sent.

Run:
    pip install -e ".[api]"
    uvicorn examples.fastapi_app:app --reload
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from nemotron_reasoned_guard import ReasonedGuard, Policy
from nemotron_reasoned_guard.policies import DEFAULT_POLICIES

app = FastAPI(title="Nemotron Reasoned Guard — Example API")

guard = ReasonedGuard()


class GuardRequest(BaseModel):
    text: str
    policy_name: str | None = None
    policy_rules: str | None = None
    image_url: str | None = None
    context: str | None = None


class GuardResponse(BaseModel):
    is_safe: bool
    categories: list[str]
    reasoning: str
    policy_name: str
    confidence: float | None = None


def get_policy(request: GuardRequest) -> str | Policy:
    if request.policy_rules:
        return request.policy_rules
    if request.policy_name:
        return DEFAULT_POLICIES.get(
            request.policy_name, DEFAULT_POLICIES["corporate"]
        )
    return DEFAULT_POLICIES["corporate"]


@app.post("/guard", response_model=GuardResponse)
async def guard_endpoint(req: GuardRequest, policy: Annotated[str | Policy, Depends(get_policy)]):
    """Direct guard endpoint. Call this from your frontend, agent, or another service."""
    try:
        result = guard.check(
            text=req.text,
            policy=policy,
            image=req.image_url,
            context=req.context,
        )
        return GuardResponse(
            is_safe=result.is_safe,
            categories=result.categories,
            reasoning=result.reasoning,
            policy_name=result.policy_name,
            confidence=result.confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guard check failed: {e}") from e


# Example of using the guard as a dependency inside your own routes
def guard_user_input(
    text: str,
    policy: str | Policy = DEFAULT_POLICIES["corporate"],
) -> None:
    """Raise 400 if the input violates policy."""
    result = guard.check(text=text, policy=policy)
    if not result.is_safe:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Input blocked by safety policy",
                "categories": result.categories,
                "reasoning": result.reasoning[:800],
            },
        )


@app.post("/chat")
async def protected_chat(user_message: str):
    """Your normal chat route, now protected."""
    guard_user_input(user_message)  # <-- one line protection

    # In a real app you would call your LLM here
    # For demo we just echo
    assistant_reply = f"Thanks for your message: {user_message[:100]}"

    # You can also guard the *output* before returning it
    guard.check(text=assistant_reply, policy=DEFAULT_POLICIES["corporate"])

    return {"reply": assistant_reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
