"""
SentinelFlow: Unit Tests
========================
Tests cover:
  1. Feature engineering module (engineering.py)
  2. API scoring endpoint (POST /score) with a complete transaction
  3. Graceful failure case: transaction with missing features must not crash
     the API and must route to manual review with a low_confidence label

Run with:
    python -m pytest tests/ -v
"""

import os
import sys
import pytest

# Add project root to path so imports work correctly from the tests folder
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.engineering import (
    engineer_features_from_row,
    encode_payment_method,
    encode_merchant_category,
    compute_amount_zscore,
    compute_geo_distance,
    FEATURE_COLUMNS,
)
from features.explainability import (
    get_top_shap_features,
    feature_to_plain_text,
    build_explanation_sentence,
    build_low_confidence_explanation,
)


# -----------------------------------------------------------------------
# Helper: minimal valid transaction dict for testing
# -----------------------------------------------------------------------
def make_valid_transaction(overrides: dict = {}) -> dict:
    """
    Return a minimal valid transaction dict suitable for engineer_features_from_row.

    Args:
        overrides: Dict of fields to override in the default transaction.

    Returns:
        Transaction dict with all required fields.
    """
    base = {
        "amount":                          5000.0,
        "payment_method":                  "UPI",
        "merchant_category":               "grocery",
        "user_transaction_count_last_1h":  2,
        "user_transaction_count_last_24h": 8,
        "amount_zscore_vs_user_history":   0.5,
        "geo_distance_from_last_txn_km":   12.0,
        "is_new_device":                   0,
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------
# Section 1: Feature Engineering Tests
# -----------------------------------------------------------------------
class TestFeatureEngineering:
    """Tests for the feature engineering module."""

    def test_encode_payment_method_known(self):
        """Known payment methods should encode to their assigned integers."""
        assert encode_payment_method("UPI")        == 0
        assert encode_payment_method("card")       == 1
        assert encode_payment_method("netbanking") == 2

    def test_encode_payment_method_unknown(self):
        """Unknown payment methods should fall back to 0."""
        assert encode_payment_method("crypto") == 0
        assert encode_payment_method("")       == 0

    def test_encode_merchant_category_known(self):
        """Known merchant categories should encode correctly."""
        assert encode_merchant_category("grocery")     == 0
        assert encode_merchant_category("electronics") == 1
        assert encode_merchant_category("travel")      == 4

    def test_encode_merchant_category_unknown(self):
        """Unknown merchant categories should fall back to 0."""
        assert encode_merchant_category("alien_tech") == 0

    def test_compute_amount_zscore_zero_std(self):
        """A near-zero user_std should not cause a division error and returns 0."""
        z = compute_amount_zscore(5000.0, 4000.0, 0.0)
        assert z == 0.0

    def test_compute_amount_zscore_positive(self):
        """Amount above average should yield a positive z-score."""
        z = compute_amount_zscore(10000.0, 5000.0, 2000.0)
        assert z > 0

    def test_compute_amount_zscore_negative(self):
        """Amount below average should yield a negative z-score."""
        z = compute_amount_zscore(1000.0, 5000.0, 2000.0)
        assert z < 0

    def test_compute_geo_distance_known_cities(self):
        """Distance between two known Indian cities should be positive."""
        dist = compute_geo_distance("Mumbai", "Delhi")
        assert dist > 1000.0  # Mumbai to Delhi is approximately 1150 km

    def test_compute_geo_distance_same_city(self):
        """Distance from a city to itself should be 0."""
        dist = compute_geo_distance("Pune", "Pune")
        assert dist == 0.0

    def test_compute_geo_distance_unknown_city(self):
        """Unknown city should return 0.0 without raising an exception."""
        dist = compute_geo_distance("Atlantis", "Mumbai")
        assert dist == 0.0

    def test_engineer_features_from_row_complete(self):
        """A complete transaction row should produce all FEATURE_COLUMNS."""
        row      = make_valid_transaction()
        features = engineer_features_from_row(row)

        assert features["has_missing_data"] is False
        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_engineer_features_from_row_missing_velocity(self):
        """Missing velocity features should set has_missing_data to True."""
        row = make_valid_transaction({
            "user_transaction_count_last_1h":  None,
            "user_transaction_count_last_24h": None,
        })
        features = engineer_features_from_row(row)
        assert features["has_missing_data"] is True

    def test_engineer_features_from_row_missing_zscore(self):
        """Missing amount z-score should set has_missing_data to True."""
        row      = make_valid_transaction({"amount_zscore_vs_user_history": None})
        features = engineer_features_from_row(row)
        assert features["has_missing_data"] is True

    def test_engineer_features_from_row_default_velocity_values(self):
        """Missing velocity features should use the conservative defaults."""
        row = make_valid_transaction({
            "user_transaction_count_last_1h":  None,
            "user_transaction_count_last_24h": None,
        })
        features = engineer_features_from_row(row)
        assert features["user_transaction_count_last_1h"]  == 5
        assert features["user_transaction_count_last_24h"] == 10


# -----------------------------------------------------------------------
# Section 2: Explainability Tests
# -----------------------------------------------------------------------
class TestExplainability:
    """Tests for the explainability module."""

    def test_get_top_shap_features_ordering(self):
        """Top features should be ordered by descending absolute SHAP value."""
        import numpy as np
        feature_names = ["amount", "user_transaction_count_last_1h", "is_new_device"]
        shap_values   = np.array([0.1, 0.9, -0.5])
        top           = get_top_shap_features(shap_values, feature_names, top_n=2)

        assert top[0][0] == "user_transaction_count_last_1h"
        assert top[1][0] == "is_new_device"

    def test_get_top_shap_features_length_mismatch(self):
        """Mismatched shap_values and feature_names should raise ValueError."""
        import numpy as np
        with pytest.raises(ValueError):
            get_top_shap_features(np.array([0.1, 0.2]), ["only_one_feature"], top_n=1)

    def test_feature_to_plain_text_positive_shap(self):
        """Positive SHAP value for velocity should indicate a velocity spike."""
        text = feature_to_plain_text("user_transaction_count_last_1h", 0.8)
        assert "velocity spike" in text.lower() or "velocity" in text.lower()

    def test_feature_to_plain_text_negative_shap(self):
        """Negative SHAP value for new device should indicate a known device."""
        text = feature_to_plain_text("is_new_device", -0.5)
        assert "known device" in text.lower() or "known" in text.lower()

    def test_build_explanation_sentence_not_empty(self):
        """build_explanation_sentence should return a non-empty string."""
        import numpy as np
        shap_vals = np.array([0.3, 0.8, 0.1, 0.05, 0.6, 0.9, 0.0, 0.0])
        sentence  = build_explanation_sentence(
            shap_values   = shap_vals,
            feature_names = FEATURE_COLUMNS,
            fraud_score   = 0.75,
            decision      = "block",
        )
        assert isinstance(sentence, str)
        assert len(sentence) > 10

    def test_build_low_confidence_explanation_contains_reason(self):
        """Low-confidence explanation should include the provided reason."""
        reason = "velocity features were absent"
        text   = build_low_confidence_explanation(reason)
        assert reason in text


# -----------------------------------------------------------------------
# Section 3: API Graceful Failure Tests (without running the full server)
# -----------------------------------------------------------------------
class TestGracefulFailureHandling:
    """
    Tests that verify graceful failure handling for transactions with missing
    data, without requiring a live FastAPI server.

    These tests call engineer_features_from_row directly and verify that:
      - The function does not raise an exception
      - has_missing_data is True
      - Conservative default values are applied
    """

    def test_all_optional_fields_missing(self):
        """
        A transaction with all optional velocity and history fields set to None
        must not raise an exception and must flag as low confidence.
        """
        row = {
            "amount":                          999.0,
            "payment_method":                  "card",
            "merchant_category":               "electronics",
            "user_transaction_count_last_1h":  None,
            "user_transaction_count_last_24h": None,
            "amount_zscore_vs_user_history":   None,
            "geo_distance_from_last_txn_km":   None,
            "is_new_device":                   0,
        }
        # This must not raise
        features = engineer_features_from_row(row)

        assert features["has_missing_data"] is True
        # Conservative defaults should be applied
        assert features["user_transaction_count_last_1h"]  == 5
        assert features["user_transaction_count_last_24h"] == 10
        assert features["amount_zscore_vs_user_history"]   == 0.0
        assert features["geo_distance_from_last_txn_km"]   == 0.0

    def test_empty_transaction_dict_does_not_crash(self):
        """
        An almost-empty transaction dict (only amount present) must not crash
        and must produce a has_missing_data True result.
        """
        row = {"amount": 500.0}
        features = engineer_features_from_row(row)
        assert features["has_missing_data"] is True

    def test_complete_transaction_no_missing_flag(self):
        """
        A fully specified transaction must set has_missing_data to False.
        """
        row      = make_valid_transaction()
        features = engineer_features_from_row(row)
        assert features["has_missing_data"] is False

    def test_partial_missing_triggers_low_confidence(self):
        """
        A transaction missing only one velocity field should still trigger
        the low-confidence flag.
        """
        row = make_valid_transaction({"user_transaction_count_last_1h": None})
        features = engineer_features_from_row(row)
        assert features["has_missing_data"] is True


# -----------------------------------------------------------------------
# Section 4: Integration Tests (requires running API server)
# -----------------------------------------------------------------------
class TestAPIIntegration:
    """
    Integration tests for the FastAPI scoring API.
    These tests are skipped if the API server is not reachable.

    To run these tests manually:
      1. Start the API: uvicorn api.main:app --port 8000
      2. Run: python -m pytest tests/ -v -k "integration"
    """

    API_BASE_URL = "http://127.0.0.1:8000"

    def _is_api_running(self) -> bool:
        """Check if the API server is reachable."""
        try:
            import requests
            r = requests.get(f"{self.API_BASE_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def test_health_endpoint(self):
        """GET /health should return 200 with status 'ok' when model is loaded."""
        if not self._is_api_running():
            pytest.skip("API server not running. Start with: uvicorn api.main:app --port 8000")

        import requests
        r = requests.get(f"{self.API_BASE_URL}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")

    def test_score_valid_transaction(self):
        """POST /score with a complete transaction should return a valid response."""
        if not self._is_api_running():
            pytest.skip("API server not running.")

        import requests
        payload = {
            "transaction_id":                  "TEST_001",
            "user_id":                         "USR_TEST",
            "amount":                          5000.0,
            "payment_method":                  "UPI",
            "merchant_category":               "grocery",
            "user_transaction_count_last_1h":  2,
            "user_transaction_count_last_24h": 8,
            "amount_zscore_vs_user_history":   0.5,
            "geo_distance_from_last_txn_km":   10.0,
            "is_new_device":                   0,
        }
        r = requests.post(f"{self.API_BASE_URL}/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "fraud_score" in data
        assert "decision"    in data
        assert "explanation" in data
        assert data["decision"] in ("approve", "flag_for_review", "block")
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_score_missing_features_graceful_failure(self):
        """
        POST /score with missing velocity and history fields must not return 500.
        Must return a flag_for_review decision with low_confidence label.
        This test specifically validates the graceful failure requirement.
        """
        if not self._is_api_running():
            pytest.skip("API server not running.")

        import requests
        payload = {
            "transaction_id":                  "TEST_MISSING_DATA_001",
            "user_id":                         "USR_NEW",
            "amount":                          3500.0,
            "payment_method":                  "card",
            # All optional fields deliberately omitted to trigger low-confidence path
        }
        r = requests.post(f"{self.API_BASE_URL}/score", json=payload)
        # Must not crash (no 500 error)
        assert r.status_code == 200
        data = r.json()
        # Must route to manual review, not auto-approve or auto-block
        assert data["decision"]    == "flag_for_review"
        assert data["confidence"]  == "low_confidence"
        assert data["fraud_score"] == 0.5

    def test_audit_log_populated_after_scoring(self):
        """
        After scoring a transaction, it must appear in the audit log.
        """
        if not self._is_api_running():
            pytest.skip("API server not running.")

        import requests, time
        txn_id = f"TEST_AUDIT_{int(time.time())}"
        payload = {
            "transaction_id":                  txn_id,
            "amount":                          1000.0,
            "payment_method":                  "UPI",
            "user_transaction_count_last_1h":  1,
            "user_transaction_count_last_24h": 3,
            "amount_zscore_vs_user_history":   0.2,
            "geo_distance_from_last_txn_km":   5.0,
            "is_new_device":                   0,
        }
        requests.post(f"{self.API_BASE_URL}/score", json=payload)

        log_r = requests.get(f"{self.API_BASE_URL}/audit-log?limit=10")
        assert log_r.status_code == 200
        log_data = log_r.json()
        ids = [entry["transaction_id"] for entry in log_data]
        assert txn_id in ids, f"Transaction {txn_id} not found in audit log"
