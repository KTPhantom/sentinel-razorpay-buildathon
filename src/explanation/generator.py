"""
Explanation layer — generates a natural language explanation from triggered signals.

The LLM (Claude) is called when an API key is available. If not, a deterministic
fallback builds the explanation from the signal values directly. Either way the
output goes through the grounding validator before being returned.

The function signature only accepts triggered_signals — no raw transaction data,
no feature vectors. That boundary is what makes the explanations auditable.
"""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from src.explanation.validator import GroundingValidator

load_dotenv()

SYSTEM_PROMPT = (
    "You are a risk analyst assistant. You will be given ONLY the triggered signals for a transaction. "
    "Write a 1-2 sentence explanation using ONLY these signals. Do not invent or assume any information not present. "
    "Then state confidence (low/medium/high) with one sentence justifying it based on the number and strength of triggered signals. "
    "Respond ONLY with valid JSON in this exact structure:\n"
    "{\n"
    '  "explanation": "1-2 sentence explanation grounded strictly in the provided signals.",\n'
    '  "confidence": "low | medium | high",\n'
    '  "confidence_justification": "One sentence justifying confidence based on signals."\n'
    "}"
)


class ExplanationGenerator:

    def __init__(
        self,
        validator: GroundingValidator | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.validator = validator or GroundingValidator()
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate(self, triggered_signals: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a grounded explanation for the given triggered_signals list."""
        if not triggered_signals:
            explanation = "This transaction did not trigger any risk rules or anomalous model signals."
            val_result = self.validator.validate(explanation, triggered_signals)
            return {
                "explanation": explanation,
                "confidence": "high",
                "confidence_justification": "No anomaly signals detected.",
                "grounding_passed": val_result["is_grounded"],
                "grounding_violations": val_result["violations"],
            }

        if self.client:
            try:
                raw = self._call_llm(triggered_signals)
                parsed = self._parse_json(raw)
                if parsed:
                    explanation = parsed.get("explanation", "")
                    confidence = parsed.get("confidence", "medium").lower()
                    justification = parsed.get("confidence_justification", "")
                    val_result = self.validator.validate(explanation, triggered_signals)
                    return {
                        "explanation": explanation,
                        "confidence": confidence,
                        "confidence_justification": justification,
                        "grounding_passed": val_result["is_grounded"],
                        "grounding_violations": val_result["violations"],
                    }
            except Exception:
                pass

        return self._deterministic_fallback(triggered_signals)

    def _call_llm(self, triggered_signals: list[dict[str, Any]]) -> str:
        user_content = json.dumps({"triggered_signals": triggered_signals}, indent=2)
        message = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=250,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return None

    def _deterministic_fallback(self, triggered_signals: list[dict[str, Any]]) -> dict[str, Any]:
        reasons: list[str] = []
        rule_signals = [s for s in triggered_signals if s.get("type") == "rule"]
        model_signals = [s for s in triggered_signals if s.get("type") == "model_feature"]

        for sig in rule_signals:
            name = sig.get("name")
            val = sig.get("value", "")
            if name == "amount_spike":
                reasons.append(f"a significant amount spike ({val})")
            elif name == "geo_anomaly":
                reasons.append(f"a geographic mismatch ({val})")
            elif name == "high_velocity":
                reasons.append(f"a high transaction velocity spike ({val})")
            elif name == "new_device_high_value":
                reasons.append(f"a high-value transaction on an unrecognized device ({val})")
            elif name == "odd_hour_spend":
                reasons.append(f"unusual spending during off-hours ({val})")

        for sig in model_signals:
            feat = sig.get("name")
            contrib = sig.get("contribution", 0)
            reasons.append(f"elevated anomaly attribution on model feature {feat} (+{contrib:.2f} SHAP score)")

        if not reasons:
            explanation = "This transaction was flagged based on triggered risk signals."
        elif len(reasons) == 1:
            explanation = f"This transaction was flagged due to {reasons[0]}."
        else:
            explanation = (
                f"This transaction was flagged due to {', '.join(reasons[:-1])}, "
                f"combined with {reasons[-1]}."
            )

        n = len(triggered_signals)
        if n >= 3:
            confidence, justification = "high", f"{n} distinct risk signals triggered concurrently."
        elif n == 2:
            confidence, justification = "high", "Multiple independent risk signals triggered simultaneously."
        elif n == 1:
            confidence, justification = "medium", "A single high-severity risk signal was triggered."
        else:
            confidence, justification = "low", "Weak or ambiguous signals detected."

        val_result = self.validator.validate(explanation, triggered_signals)
        return {
            "explanation": explanation,
            "confidence": confidence,
            "confidence_justification": justification,
            "grounding_passed": val_result["is_grounded"],
            "grounding_violations": val_result["violations"],
        }
