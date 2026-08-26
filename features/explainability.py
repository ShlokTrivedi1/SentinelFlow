"""
SentinelFlow: SHAP-Based Explainability Layer
=============================================
Translates raw SHAP values into plain-language explanations for each
flagged transaction. No raw numerical SHAP output is exposed to the end user.
Every explanation is rendered as a human-readable sentence.
"""

from typing import Optional
import numpy as np


# Human-readable names for model feature columns
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "amount":                           "transaction amount",
    "user_transaction_count_last_1h":   "transaction velocity (last 1 hour)",
    "user_transaction_count_last_24h":  "transaction velocity (last 24 hours)",
    "amount_zscore_vs_user_history":    "amount deviation from user history",
    "geo_distance_from_last_txn_km":    "geographic distance from last transaction",
    "is_new_device":                    "use of a new or unrecognized device",
    "payment_method_encoded":           "payment method",
    "merchant_category_encoded":        "merchant category",
}

# Directional descriptor templates for each feature
# Keys are feature names, values are (increase_text, decrease_text)
FEATURE_DIRECTION_TEMPLATES: dict[str, tuple[str, str]] = {
    "amount":                          ("unusually high transaction amount", "low transaction amount"),
    "user_transaction_count_last_1h":  ("velocity spike in the last hour", "low velocity in the last hour"),
    "user_transaction_count_last_24h": ("high number of transactions in the last 24 hours", "low activity in the last 24 hours"),
    "amount_zscore_vs_user_history":   ("amount far above the user's historical average", "amount below the user's historical average"),
    "geo_distance_from_last_txn_km":   ("large geographic jump from the last transaction", "transaction close to the previous location"),
    "is_new_device":                   ("transaction from a new and unrecognized device", "transaction from a known device"),
    "payment_method_encoded":          ("unusual payment method for this user", "typical payment method"),
    "merchant_category_encoded":       ("unusual merchant category for this user", "typical merchant category"),
}


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """
    Extract the top-N features with the largest absolute SHAP values.
    Returns them sorted from most influential to least influential.

    Args:
        shap_values:   1-D array of SHAP values for a single prediction.
        feature_names: List of feature name strings matching shap_values length.
        top_n:         Number of top features to return.

    Returns:
        List of (feature_name, shap_value) tuples ordered by descending
        absolute importance.
    """
    if len(shap_values) != len(feature_names):
        raise ValueError(
            f"shap_values length ({len(shap_values)}) does not match "
            f"feature_names length ({len(feature_names)})."
        )

    pairs = list(zip(feature_names, shap_values))
    # Sort by absolute SHAP value, descending
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def feature_to_plain_text(feature_name: str, shap_value: float) -> str:
    """
    Convert a single (feature_name, shap_value) pair into a plain English
    phrase describing how this feature contributed to the fraud score.

    A positive SHAP value means the feature pushed the score toward fraud.
    A negative SHAP value means it pushed the score away from fraud.

    Args:
        feature_name: The internal feature column name.
        shap_value:   The SHAP value for this feature in a specific prediction.

    Returns:
        A human-readable phrase such as "velocity spike in the last hour".
    """
    direction = "increase" if shap_value > 0 else "decrease"
    templates = FEATURE_DIRECTION_TEMPLATES.get(feature_name)

    if templates:
        text = templates[0] if shap_value > 0 else templates[1]
    else:
        display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
        text = f"elevated {display}" if shap_value > 0 else f"low {display}"

    return text


def build_explanation_sentence(
    shap_values: np.ndarray,
    feature_names: list[str],
    fraud_score: float,
    decision: str,
    top_n: int = 3,
) -> str:
    """
    Generate a complete plain-English explanation sentence for a scored
    transaction by combining the top-N SHAP feature contributions.

    The output is intended for a human reviewer and does not expose raw
    SHAP numbers.

    Args:
        shap_values:   1-D array of SHAP values for this prediction.
        feature_names: Feature names corresponding to shap_values.
        fraud_score:   Model output probability (0 to 1).
        decision:      One of 'approve', 'flag_for_review', or 'block'.
        top_n:         Number of features to include in the explanation.

    Returns:
        A human-readable explanation string.
    """
    top_features = get_top_shap_features(shap_values, feature_names, top_n=top_n)
    phrases = [feature_to_plain_text(name, val) for name, val in top_features]

    if len(phrases) == 0:
        return "No significant risk drivers were identified for this transaction."
    elif len(phrases) == 1:
        driver_text = phrases[0]
    elif len(phrases) == 2:
        driver_text = f"{phrases[0]} and {phrases[1]}"
    else:
        driver_text = f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"

    score_pct = round(fraud_score * 100, 1)

    if decision == "block":
        prefix = f"This transaction was blocked (risk score {score_pct}%) mainly due to"
    elif decision == "flag_for_review":
        prefix = f"This transaction was flagged for review (risk score {score_pct}%) mainly due to"
    else:
        prefix = f"This transaction was approved (risk score {score_pct}%), with minor signals noted including"

    return f"{prefix}: {driver_text}."


def build_low_confidence_explanation(reason: str) -> str:
    """
    Generate a standard explanation for transactions that were routed to
    manual review due to missing data, rather than a model prediction.

    Args:
        reason: A short description of what data was missing or unavailable.

    Returns:
        A plain-English explanation string suitable for the audit log.
    """
    return (
        f"This transaction was routed to manual review because the model "
        f"could not produce a reliable prediction. Reason: {reason}. "
        f"A conservative default score was applied and no automatic decision "
        f"was made."
    )
