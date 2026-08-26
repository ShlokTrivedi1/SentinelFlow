"""
SentinelFlow: FastAPI Scoring Service
=====================================
Exposes three endpoints:

  POST /score         Score a single transaction and return a risk decision.
  GET  /audit-log     Return recently scored transactions from the audit log.
  GET  /health        Return service health status.

Decision logic uses two configurable thresholds:
  - THRESHOLD_FLAG  (default 0.35): score >= this triggers 'flag_for_review'
  - THRESHOLD_BLOCK (default 0.70): score >= this triggers 'block'
  - Below THRESHOLD_FLAG: 'approve'

This service is strictly defense-oriented. It contains no logic that could
assist fraud execution or evasion.
"""

import os
import sys
import pickle
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Any, AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# Add project root to path so sibling packages (features, api) can be imported
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.engineering import FEATURE_COLUMNS, engineer_features_from_row
from features.explainability import (
    build_explanation_sentence,
    build_low_confidence_explanation,
)
from api.database import init_db, log_transaction, fetch_recent_logs, count_by_decision

# -----------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinelflow.api")

# -----------------------------------------------------------------------
# Model artifact paths
# -----------------------------------------------------------------------
MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "model.json")
EXPLAINER_PATH = os.path.join(PROJECT_ROOT, "models", "shap_explainer.pkl")
MODEL_VERSION  = "1.0.0"

# -----------------------------------------------------------------------
# Decision thresholds (configurable via environment variables)
# -----------------------------------------------------------------------
THRESHOLD_FLAG  = float(os.environ.get("SENTINEL_THRESHOLD_FLAG",  "0.35"))
THRESHOLD_BLOCK = float(os.environ.get("SENTINEL_THRESHOLD_BLOCK", "0.70"))

# Fallback score used when transaction data is incomplete
LOW_CONFIDENCE_FALLBACK_SCORE = 0.5

# -----------------------------------------------------------------------
# Global model and explainer (loaded once at startup)
# -----------------------------------------------------------------------
_model: Optional[XGBClassifier] = None
_explainer: Optional[Any]       = None


def load_artifacts() -> tuple[XGBClassifier, Any]:
    """
    Load the trained XGBoost model and SHAP explainer from disk.
    Called once during application startup to avoid per-request I/O.

    Returns:
        Tuple of (XGBClassifier model, SHAP TreeExplainer).

    Raises:
        FileNotFoundError: If the model or explainer files are not present.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run scripts/train_model.py first."
        )
    if not os.path.exists(EXPLAINER_PATH):
        raise FileNotFoundError(
            f"SHAP explainer not found at {EXPLAINER_PATH}. "
            f"Run scripts/train_model.py first."
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)

    logger.info("Model and SHAP explainer loaded successfully.")
    return model, explainer


# -----------------------------------------------------------------------
# Lifespan context manager (replaces deprecated @app.on_event)
# -----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler. Runs startup logic before yield and
    shutdown logic after yield. Replaces the deprecated @app.on_event.

    Startup: initialise the SQLite audit log and load the model and SHAP
    explainer into module-level globals so they are reused across requests.
    """
    global _model, _explainer
    init_db()
    logger.info("Audit log database initialised.")
    try:
        _model, _explainer = load_artifacts()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        # The API will still start, but /score will return 503 until models exist
    yield
    # Shutdown: nothing to clean up for this lightweight service


# -----------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------
app = FastAPI(
    title       = "SentinelFlow Scoring API",
    description = (
        "Real-time fraud spike detection for payment transactions. "
        "This API is strictly defense-oriented."
    ),
    version     = MODEL_VERSION,
    lifespan    = lifespan,
)

# Allow requests from the Streamlit dashboard running on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# -----------------------------------------------------------------------
# Pydantic request and response models
# -----------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """
    Payload for the POST /score endpoint.
    All fields except transaction_id have default values so the endpoint
    can handle transactions with partially missing data gracefully.
    """
    transaction_id:                  str   = Field(...,  description="Unique transaction identifier")
    user_id:                         Optional[str]   = Field(None, description="User identifier")
    amount:                          float = Field(...,  description="Transaction amount in INR", gt=0)
    payment_method:                  Optional[str]   = Field("UPI", description="UPI, card, or netbanking")
    merchant_category:               Optional[str]   = Field("grocery", description="Merchant category")
    device_id:                       Optional[str]   = Field(None, description="Device fingerprint")
    geo_location:                    Optional[str]   = Field(None, description="City name")
    user_transaction_count_last_1h:  Optional[int]   = Field(None, description="Transactions in last 1 hour")
    user_transaction_count_last_24h: Optional[int]   = Field(None, description="Transactions in last 24 hours")
    amount_zscore_vs_user_history:   Optional[float] = Field(None, description="Amount z-score vs user history")
    geo_distance_from_last_txn_km:   Optional[float] = Field(None, description="Distance from last transaction in km")
    is_new_device:                   Optional[int]   = Field(0,    description="1 if device is new, else 0")


class TransactionResponse(BaseModel):
    """
    Response body for the POST /score endpoint.
    Includes the fraud score, decision, explanation, and metadata.
    """
    transaction_id:   str
    fraud_score:      float
    decision:         str
    explanation:      str
    confidence:       str
    model_version:    str
    scored_at:        str
    thresholds_used:  dict


class AuditLogEntry(BaseModel):
    """
    A single row from the audit log table, returned by GET /audit-log.
    """
    id:              int
    transaction_id:  str
    scored_at:       str
    fraud_score:     float
    decision:        str
    explanation:     str
    model_version:   str
    confidence:      str
    amount:          Optional[float]
    payment_method:  Optional[str]
    user_id:         Optional[str]


# -----------------------------------------------------------------------
# Decision logic
# -----------------------------------------------------------------------
def make_decision(score: float) -> str:
    """
    Apply the two-threshold decision logic to a fraud probability score.

    Two thresholds are used to avoid a single brittle cutoff:
      - Below THRESHOLD_FLAG:  automatically approve the transaction.
      - Between the two:       route to human review (flag_for_review).
      - At or above THRESHOLD_BLOCK: block the transaction.

    This design ensures no transaction is auto-blocked solely based on a
    model score without a human review gate at intermediate risk levels.

    Args:
        score: Fraud probability between 0 and 1.

    Returns:
        Decision string: 'approve', 'flag_for_review', or 'block'.
    """
    if score >= THRESHOLD_BLOCK:
        return "block"
    if score >= THRESHOLD_FLAG:
        return "flag_for_review"
    return "approve"


# -----------------------------------------------------------------------
# Scoring endpoint
# -----------------------------------------------------------------------
@app.post("/score", response_model=TransactionResponse)
async def score_transaction(txn: TransactionRequest) -> TransactionResponse:
    """
    Score a single transaction and return a fraud risk decision.

    The scoring pipeline:
      1. Convert the request payload to a feature dict (engineer_features_from_row).
      2. If key features are missing, fall back to a conservative low-confidence score.
      3. Otherwise, run the XGBoost model to get a fraud probability.
      4. Compute SHAP values and translate to a plain-language explanation.
      5. Apply the two-threshold decision gate.
      6. Log the result to the SQLite audit trail.
      7. Return the full response to the caller.

    Graceful failure: if the feature dict signals missing data, the transaction
    is routed to manual review without crashing or making an unsafe decision.

    Args:
        txn: TransactionRequest payload.

    Returns:
        TransactionResponse with score, decision, and explanation.

    Raises:
        HTTPException 503 if the model has not been loaded yet.
    """
    if _model is None or _explainer is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Run scripts/train_model.py first, then restart the API.",
        )

    scored_at = datetime.utcnow().isoformat() + "Z"

    # Convert request to raw dict for feature engineering
    raw_row = txn.dict()

    # Engineer features, detects missing values and sets has_missing_data
    try:
        feature_dict = engineer_features_from_row(raw_row)
    except Exception as exc:
        logger.warning(f"Feature engineering failed for {txn.transaction_id}: {exc}")
        # Treat any unexpected engineering error as missing-data case
        feature_dict = {"has_missing_data": True}

    has_missing = feature_dict.pop("has_missing_data", False)

    if has_missing:
        # Graceful fallback for missing data, route to manual review
        fraud_score = LOW_CONFIDENCE_FALLBACK_SCORE
        decision    = "flag_for_review"
        confidence  = "low_confidence"
        explanation = build_low_confidence_explanation(
            "one or more required velocity or history features were absent from the request"
        )
        logger.info(
            f"Transaction {txn.transaction_id} routed to manual review (low confidence, missing data)."
        )
    else:
        # Build feature array in the correct column order
        feature_values = np.array(
            [[feature_dict[col] for col in FEATURE_COLUMNS]],
            dtype=np.float32,
        )
        feature_df = pd.DataFrame(feature_values, columns=FEATURE_COLUMNS)

        # Model inference
        fraud_score = float(_model.predict_proba(feature_df)[0, 1])

        # SHAP explanation
        shap_vals = _explainer.shap_values(feature_df)
        if isinstance(shap_vals, list):
            # Older SHAP returns a list of [neg_class_vals, pos_class_vals]
            shap_row = shap_vals[1][0]
        else:
            shap_row = shap_vals[0]

        decision   = make_decision(fraud_score)
        confidence = "normal"
        explanation = build_explanation_sentence(
            shap_values   = shap_row,
            feature_names = FEATURE_COLUMNS,
            fraud_score   = fraud_score,
            decision      = decision,
        )
        logger.info(
            f"Transaction {txn.transaction_id} scored {fraud_score:.4f} -> {decision}"
        )

    # Write every scored transaction to the audit log without exception
    log_transaction(
        transaction_id = txn.transaction_id,
        fraud_score    = round(fraud_score, 6),
        decision       = decision,
        explanation    = explanation,
        model_version  = MODEL_VERSION,
        confidence     = confidence,
        amount         = txn.amount,
        payment_method = txn.payment_method,
        user_id        = txn.user_id,
    )

    return TransactionResponse(
        transaction_id  = txn.transaction_id,
        fraud_score     = round(fraud_score, 6),
        decision        = decision,
        explanation     = explanation,
        confidence      = confidence,
        model_version   = MODEL_VERSION,
        scored_at       = scored_at,
        thresholds_used = {
            "flag_threshold":  THRESHOLD_FLAG,
            "block_threshold": THRESHOLD_BLOCK,
        },
    )


# -----------------------------------------------------------------------
# Audit log endpoint
# -----------------------------------------------------------------------
@app.get("/audit-log", response_model=list[AuditLogEntry])
async def get_audit_log(
    limit:           int            = Query(default=100, ge=1, le=1000),
    decision_filter: Optional[str]  = Query(default=None, description="Filter by decision: approve, flag_for_review, block"),
) -> list[AuditLogEntry]:
    """
    Return recent audit log entries, most recent first.

    Args:
        limit:           Maximum number of records to return (1 to 1000).
        decision_filter: Optional filter on the decision column.

    Returns:
        List of AuditLogEntry objects.
    """
    valid_decisions = {"approve", "flag_for_review", "block", None}
    if decision_filter not in valid_decisions:
        raise HTTPException(
            status_code = 400,
            detail      = f"decision_filter must be one of: approve, flag_for_review, block",
        )

    rows = fetch_recent_logs(limit=limit, decision_filter=decision_filter)
    return [AuditLogEntry(**row) for row in rows]


# -----------------------------------------------------------------------
# Health check endpoint
# -----------------------------------------------------------------------
@app.get("/health")
async def health_check() -> dict:
    """
    Return the current service health status.

    Returns:
        Dict with status, model loaded flag, model version, and timestamp.
    """
    model_loaded = _model is not None and _explainer is not None
    return {
        "status"        : "ok" if model_loaded else "degraded",
        "model_loaded"  : model_loaded,
        "model_version" : MODEL_VERSION,
        "thresholds"    : {
            "flag_threshold"  : THRESHOLD_FLAG,
            "block_threshold" : THRESHOLD_BLOCK,
        },
        "checked_at"    : datetime.utcnow().isoformat() + "Z",
    }
