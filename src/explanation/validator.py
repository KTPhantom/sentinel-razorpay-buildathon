"""
Checks that a generated explanation is actually grounded in its triggered signals.

Two checks run on every explanation:
- Negative: the text must not reference concepts from signal categories that didn't fire.
- Positive: the text must reference at least one keyword from each signal that did fire,
  so a generic "this looked suspicious" won't silently pass.
"""

import re
from typing import Any

# Keywords associated with each signal category. Used in both directions:
# to catch hallucinated references, and to verify triggered signals are covered.
# Note: amount_spike patterns deliberately require "higher" after the multiplier
# because other signals (odd_hour_spend, new_device_high_value) legitimately
# include amount ratios like "3.8x average amount" in their value strings.
SIGNAL_KEYWORD_MAP: dict[str, list[str]] = {
    "geo_anomaly": [
        "location",
        "distance",
        r"\d+\.?\d*\s*km",
        "abroad",
        "foreign",
        "geographic",
        "geo",
        "travel",
        "usual location",
        "city",
        "miles",
    ],
    "amount_spike": [
        "amount spike",
        r"higher than historical average",
        r"higher than.{0,20}average",
        "unusual amount",
        "outlier amount",
        "large amount spike",
        r"\d+\.?\d*x higher",
    ],
    "new_device_high_value": [
        "device",
        "unrecognized device",
        "new device",
        "unknown device",
        "hardware",
    ],
    "high_velocity": [
        "velocity",
        "rapid",
        "burst",
        r"\d+-minute",
        "quick succession",
        "multiple transactions",
        "frequency",
        "transactions within",
    ],
    "odd_hour_spend": [
        "odd hour",
        "off-hours",
        "off hours",
        "unusual time",
        "night",
        "midnight",
        "active hours",
        "outside typical",
        "early morning",
    ],
}


def _matches_any(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if any(c in pat for c in r"\.+*?[](){}^$|"):
            try:
                if re.search(pat, text, re.IGNORECASE):
                    return True
            except re.error:
                if pat.lower() in text:
                    return True
        else:
            boundary_pat = r"(?<!\w)" + re.escape(pat) + r"(?!\w)"
            if re.search(boundary_pat, text, re.IGNORECASE):
                return True
    return False


class GroundingValidator:
    """Validates that an explanation only references what was actually triggered."""

    @staticmethod
    def validate(
        explanation: str,
        triggered_signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        explanation_lower = explanation.lower().strip()

        present_rule_signals: set[str] = set()
        present_feature_names: set[str] = set()

        for sig in triggered_signals:
            name = sig.get("name", "")
            sig_type = sig.get("type", "")
            if sig_type == "rule":
                present_rule_signals.add(name)
            elif sig_type == "model_feature":
                present_feature_names.add(name)

        all_present_signals = present_rule_signals | present_feature_names
        violations: list[str] = []
        specificity_warnings: list[str] = []

        # Empty signals: only flag affirmative risk claims, not negations.
        # "did not trigger any anomalous signals" is fine; "flagged due to velocity spike" is not.
        if not triggered_signals:
            affirmative_risk_patterns = [
                r"(?<!not\s)(?<!no\s)(?<!zero\s)(?<!without\s)\bvelocity\s+spike\b",
                r"(?<!not\s)(?<!no\s)\bunrecognized\s+device\b",
                r"(?<!not\s)(?<!no\s)\bgeo\s+anomaly\b",
                r"(?<!not\s)(?<!no\s)\bamount\s+spike\b",
                r"(?<!not\s)(?<!no\s)\bodd\s+hour\b",
                r"flagged\s+due\s+to",
                r"flagged\s+because",
            ]
            for pat in affirmative_risk_patterns:
                if re.search(pat, explanation_lower, re.IGNORECASE):
                    violations.append(
                        f"Affirmative risk claim ('{pat}') but no signals triggered."
                    )
            return {
                "is_grounded": len(violations) == 0,
                "violations": violations,
                "specificity_warnings": specificity_warnings,
                "present_signals": [],
            }

        # Negative check: explanation must not reference non-triggered signal categories
        for signal_category, patterns in SIGNAL_KEYWORD_MAP.items():
            if signal_category not in present_rule_signals:
                if _matches_any(explanation_lower, patterns):
                    violations.append(
                        f"Reference to '{signal_category}' concepts but that signal wasn't triggered."
                    )

        # Negative check: reject any PCA feature names not in the triggered set
        pca_matches = re.findall(r"\bv\d{1,2}\b", explanation_lower)
        for match in pca_matches:
            feat_name = match.upper()
            if feat_name not in present_feature_names:
                violations.append(
                    f"Reference to model feature '{feat_name}' not in triggered SHAP signals."
                )

        # Positive check: each triggered rule signal should appear in the explanation
        for signal_name in present_rule_signals:
            patterns = SIGNAL_KEYWORD_MAP.get(signal_name, [])
            if patterns and not _matches_any(explanation_lower, patterns):
                specificity_warnings.append(
                    f"Signal '{signal_name}' triggered but no matching keyword in explanation."
                )

        # Positive check: triggered SHAP features should be named in the explanation
        for feat_name in present_feature_names:
            if feat_name.lower() not in explanation_lower:
                specificity_warnings.append(
                    f"SHAP feature '{feat_name}' triggered but not mentioned in explanation."
                )

        return {
            "is_grounded": len(violations) == 0,
            "violations": violations,
            "specificity_warnings": specificity_warnings,
            "present_signals": list(all_present_signals),
        }
