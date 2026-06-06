"""
Gradio playground for nemotron-reasoned-guard.

Run with:
    pip install -e ".[playground]"
    python playground/app.py

Beautiful demo that showcases the *reasoning trace* — the killer feature of using
Nemotron 3.5 Content Safety.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from nemotron_reasoned_guard import ReasonedGuard, Policy
from nemotron_reasoned_guard.policies import DEFAULT_POLICIES, load_policy_pack

load_dotenv()

# Load example policy pack
POLICY_DIR = Path(__file__).parent.parent / "examples" / "policy_packs"
EXAMPLE_POLICIES = load_policy_pack(POLICY_DIR)
ALL_POLICIES = {**{p.name: p for p in DEFAULT_POLICIES.values()}, **EXAMPLE_POLICIES}

guard = ReasonedGuard()  # Will pick up NVIDIA_API_KEY from env


def run_guard(
    policy_choice: str,
    custom_policy: str,
    text: str,
    image: str | None,
    image_url: str,
    context: str,
    temperature: float,
):
    """Main guard call from the UI."""
    if not text.strip():
        return "⚠️ Please provide some text to evaluate.", "", "", "", ""

    # Determine policy
    if policy_choice == "Custom (paste below)":
        if not custom_policy.strip():
            return "⚠️ Please paste a custom policy or choose one from the list.", "", "", "", ""
        policy = custom_policy.strip()
        policy_name = "custom"
    else:
        policy = ALL_POLICIES.get(policy_choice, DEFAULT_POLICIES["corporate"])
        policy_name = policy.name if isinstance(policy, Policy) else policy_choice

    # Image handling (prefer uploaded file path, then URL)
    img_ref = None
    if image is not None:
        img_ref = image  # gradio gives a temp path
    elif image_url.strip():
        img_ref = image_url.strip()

    try:
        result = guard.check(
            text=text,
            policy=policy,
            image=img_ref,
            context=context.strip() or None,
            temperature=temperature,
            max_tokens=2048,
        )
    except Exception as e:
        return f"❌ Error calling NVIDIA NIM: {e}", "", "", "", ""

    # Verdict badge
    verdict = "✅ SAFE" if result.is_safe else "🚫 UNSAFE"
    verdict_color = "green" if result.is_safe else "red"

    # Categories
    cats = ", ".join(result.categories) if result.categories else "none"

    # Full reasoning (we'll render as markdown for nice formatting)
    reasoning_md = result.reasoning.replace("\n", "\n\n")

    # Audit JSON
    audit = json.dumps(result.to_audit_dict(), indent=2)

    return (
        f"<h2 style='color:{verdict_color}'>{verdict}</h2>",
        f"**Policy:** {policy_name}  \n**Categories:** {cats}  \n**Confidence:** {result.confidence or 'N/A'}",
        reasoning_md,
        audit,
        result.raw_response or result.reasoning,
    )


def load_example_policy(choice: str) -> str:
    if choice == "None":
        return ""
    pol = ALL_POLICIES.get(choice)
    if isinstance(pol, Policy):
        return pol.rules
    return ""


def build_ui():
    with gr.Blocks(title="Nemotron Reasoned Guard", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🛡️ Nemotron Reasoned Guard — Playground

            **Powered by NVIDIA Nemotron 3.5 Content Safety**

            This demo shows the **unique strength** of the Nemotron safety models:
            you don't just get a yes/no — you get a full, human-readable **reasoning trace**
            that explains exactly why content was allowed or blocked according to your policy.

            Perfect for auditing, compliance, building trust, and generating high-quality safety datasets.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Policy")
                policy_choice = gr.Dropdown(
                    choices=["Custom (paste below)"] + list(ALL_POLICIES.keys()),
                    value="corporate",
                    label="Choose policy or use custom",
                )
                custom_policy_box = gr.Textbox(
                    label="Custom Policy (only used if 'Custom' selected)",
                    placeholder="Describe your rules in plain English...",
                    lines=6,
                )
                policy_choice.change(
                    fn=load_example_policy,
                    inputs=policy_choice,
                    outputs=custom_policy_box,
                )

                gr.Examples(
                    examples=[
                        ["kids", "User asked an AI to roleplay as a 12-year-old and describe intimate acts."],
                        ["corporate", "Tell me the exact pricing and launch date of our unreleased product."],
                        ["medical", "I have been having chest pains and dizziness for 3 days. What medicine should I take?"],
                    ],
                    inputs=[policy_choice, custom_policy_box],
                    label="Quick policy + scenario examples",
                )

            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Content to Evaluate",
                    placeholder="Paste the user message, assistant response, tool output, or any text here...",
                    lines=8,
                )
                context_input = gr.Textbox(
                    label="Additional Context (optional)",
                    placeholder="Previous conversation turns, user role, application context...",
                    lines=3,
                )

                with gr.Row():
                    image_upload = gr.Image(
                        label="Upload Image (optional)", type="filepath", height=200
                    )
                    image_url = gr.Textbox(
                        label="Or paste image URL",
                        placeholder="https://... or data:image/...",
                    )

                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature (lower = more consistent)")

                run_btn = gr.Button("🛡️ Run Reasoned Guard Check", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                verdict_html = gr.HTML(label="Verdict")
                summary_md = gr.Markdown(label="Summary")

            with gr.Column():
                reasoning_md = gr.Markdown(
                    label="Full Reasoning Trace (the gold)",
                    elem_classes=["reasoning-box"],
                )

        with gr.Accordion("Raw output & Audit JSON", open=False):
            audit_json = gr.Code(language="json", label="Audit record (great for logging)")
            raw_output = gr.Textbox(label="Raw model output", lines=10)

        run_btn.click(
            fn=run_guard,
            inputs=[
                policy_choice,
                custom_policy_box,
                text_input,
                image_upload,
                image_url,
                context_input,
                temperature,
            ],
            outputs=[verdict_html, summary_md, reasoning_md, audit_json, raw_output],
        )

        gr.Markdown(
            """
            ---
            ### Why this matters

            Traditional safety classifiers give you only a label.  
            **Nemotron reasoned safety** gives you the *why* — in natural language.

            This enables:
            - Real audit trails and compliance reporting
            - Human review of borderline cases
            - High-quality preference / violation datasets for fine-tuning
            - Policy iteration ("our new policy blocked too many creative writing requests — let's look at the traces")

            Built with the Nemotron 3.5 Content Safety model via NVIDIA NIM.
            """
        )

    return demo


if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        print("⚠️  NVIDIA_API_KEY not set. The playground will fail when you click Run.")
        print("   Copy .env.example → .env and add your key, or export it.")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
