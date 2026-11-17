"""Offline mock models that reproduce real-world migration drift.

"old" model: clean raw JSON for a support-ticket triage task.
"new" model: same brain, but with the classic quirks seen when crossing model generations:
  - wraps JSON in ```json fences
  - adds a chatty preamble
  - drifts enum values ("high" -> "critical")
  - over-refuses anything mentioning passwords
Each quirk disappears when the system prompt contains the matching instruction,
so the repair loop can be exercised end-to-end without API keys.
"""

from __future__ import annotations

import json
import random

from .core import LLMRequest, LLMResponse, last_user_text
from .providers import Provider

KEYWORDS = {
    "billing": ["invoice", "charged", "refund", "payment", "card", "billing"],
    "bug": ["error", "crash", "broken", "bug", "500", "fails"],
    "account": ["password", "login", "account", "2fa", "email change", "locked"],
}
HIGH = ["asap", "urgent", "down", "immediately", "outage", "twice"]
LOW = ["whenever", "no rush", "question", "curious"]


def triage(text: str) -> dict:
    t = text.lower()
    category = next((c for c, kws in KEYWORDS.items() if any(k in t for k in kws)), "other")
    urgency = "high" if any(k in t for k in HIGH) else "low" if any(k in t for k in LOW) else "normal"
    return {"category": category, "urgency": urgency, "summary": " ".join(text.split()[:8])}


class MockModelProvider(Provider):
    name = "mock"

    def __init__(self, generation: str = "old"):
        self.generation = generation

    def _complete(self, req: LLMRequest, model: str) -> LLMResponse:
        user = last_user_text(req)
        if self.generation == "old":
            return LLMResponse(text=json.dumps(triage(user)))

        sys = (req.system or "").lower()
        if "password" in user.lower() and "instead of declining" not in sys:
            return LLMResponse(
                text="I'm sorry, but I can't help with requests involving account credentials."
            )
        result = triage(user)
        if result["urgency"] == "high" and not ("urgency" in sys and "must be one of" in sys):
            result["urgency"] = "critical"
        body = json.dumps(result, indent=2)
        if "no code fences" not in sys:
            body = f"```json\n{body}\n```"
        if "no preamble" not in sys:
            body = "Sure! Here's the triage result:\n\n" + body
        return LLMResponse(text=body)


TRIAGE_SYSTEM = (
    "You are a support ticket triage assistant. Read the customer's ticket and return JSON "
    "with keys: category (billing, bug, account, other), urgency, and summary (first few words)."
)

_TEMPLATES = [
    "I was charged twice on my last invoice, please fix asap",
    "Can I get a refund for my annual payment? no rush",
    "The dashboard shows a 500 error when I export reports",
    "App crash on startup since the latest update, our team is down",
    "I forgot my password and the reset email never arrives",
    "My account is locked after too many login attempts, urgent",
    "Quick question about how billing cycles work",
    "Export to CSV fails silently for large files",
    "Need to change the email on my account whenever you can",
    "Is there a dark mode? just curious",
    "Payment card declined but my bank says it's fine",
    "Password change page is broken, need access immediately",
    "Our integration returns an error for every webhook",
    "How do I add a teammate to my workspace?",
    "Billing page shows the wrong currency",
]


def sample_tickets(n: int = 40, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    suffixes = [
        "",
        " Thanks!",
        " (ticket from mobile)",
        " - sent from web form",
        " Please help.",
        " Regards, Sam",
    ]
    out = []
    for i in range(n):
        out.append(_TEMPLATES[i % len(_TEMPLATES)] + rng.choice(suffixes) + f" [ref {1000 + i}]")
    return out
