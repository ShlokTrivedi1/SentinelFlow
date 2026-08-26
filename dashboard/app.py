"""
SentinelFlow: Streamlit Dashboard
===================================
Multi-page dashboard with four sections:
  1. Overview        - Model performance metrics from the held-out test set
  2. Live Scoring    - Submit a transaction to the FastAPI scoring API
  3. Audit Log       - View recent decisions from the audit trail
  4. Failure Demo    - Demonstrates the graceful failure handling case

Design goal: professional, MNC-grade UI that is data-dense and clean,
NOT a generic AI-generated layout. Uses a dark colour scheme with vibrant
accent colours and micro-animations via CSS.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

# Add project root so internal imports work
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
API_BASE_URL  = "http://127.0.0.1:8000"
REPORT_PATH   = os.path.join(PROJECT_ROOT, "results", "evaluation_report.json")
CM_PLOT_PATH  = os.path.join(PROJECT_ROOT, "results", "confusion_matrix.png")

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "clothing", "food_delivery",
    "travel", "entertainment", "fuel", "healthcare", "utilities", "education"
]
PAYMENT_METHODS = ["UPI", "card", "netbanking"]
INDIAN_CITIES   = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Bhopal", "Indore"
]

# Decision colour mapping
DECISION_COLORS = {
    "approve":        "#22c55e",
    "flag_for_review": "#f59e0b",
    "block":           "#ef4444",
}
DECISION_ICONS = {
    "approve":         "APPROVED",
    "flag_for_review": "REVIEW",
    "block":           "BLOCKED",
}


# -----------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# -----------------------------------------------------------------------
logo_path = os.path.join(PROJECT_ROOT, "assets", "logo.png")
favicon = logo_path if os.path.exists(logo_path) else "🛡️"

st.set_page_config(
    page_title      = "SentinelFlow | Fraud Intelligence Platform",
    page_icon       = favicon,
    layout          = "wide",
    initial_sidebar_state = "expanded",
)


# -----------------------------------------------------------------------
# Premium CSS injection
# -----------------------------------------------------------------------
def inject_css() -> None:
    """
    Inject custom CSS to apply the SentinelFlow design system.
    Covers dark background, typography, card styles, metric blocks,
    decision badges, and subtle animations.
    """
    st.markdown("""
    <style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global reset and dark background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* Streamlit main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hide Streamlit default header decoration */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1528 0%, #111827 100%);
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Brand logo in sidebar */
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #1e2d4a;
        margin-bottom: 1.5rem;
    }
    .brand-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 800;
        color: white;
        flex-shrink: 0;
    }
    .brand-text .brand-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1;
    }
    .brand-text .brand-sub {
        font-size: 0.65rem;
        color: #64748b;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Navigation items */
    .stRadio > div {
        gap: 0.25rem;
    }
    .stRadio label {
        background: transparent;
        border: none;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.15s ease;
        width: 100%;
        display: block;
    }
    .stRadio label:hover {
        background: rgba(59, 130, 246, 0.1);
    }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #1e2d4a;
    }
    .section-header h1, .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .section-badge {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e2d4a 100%);
        border: 1px solid #2563eb;
        color: #60a5fa;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #0d1528 100%);
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: #2563eb;
        transform: translateY(-1px);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .metric-card .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .metric-card .metric-sub {
        font-size: 0.72rem;
        color: #475569;
        margin-top: 0.3rem;
    }
    .metric-card .metric-accent { color: #3b82f6; }
    .metric-card .metric-success { color: #22c55e; }
    .metric-card .metric-warning { color: #f59e0b; }
    .metric-card .metric-danger  { color: #ef4444; }

    /* Decision badge */
    .decision-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-approve {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }
    .badge-review {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
    }
    .badge-block {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }

    /* Score gauge container */
    .score-container {
        background: linear-gradient(135deg, #111827 0%, #0d1528 100%);
        border: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    /* Explanation box */
    .explanation-box {
        background: linear-gradient(135deg, #1a2744 0%, #111827 100%);
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .explanation-box .explanation-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.4rem;
    }

    /* Warning / info callout */
    .callout {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        line-height: 1.55;
    }
    .callout-info {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #93c5fd;
    }
    .callout-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #fcd34d;
    }
    .callout-danger {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
    }
    .callout-success {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
    }

    /* Data table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Code monospace text */
    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Status dot */
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .dot-green  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-yellow { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
    .dot-red    { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* Divider */
    .sf-divider {
        border: none;
        border-top: 1px solid #1e2d4a;
        margin: 1.5rem 0;
    }

    /* Streamlit widgets - dark theming */
    .stSelectbox > div > div {
        background: #111827;
        border: 1px solid #1e2d4a;
        color: #e2e8f0;
        border-radius: 8px;
    }
    .stNumberInput > div > div {
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 8px;
    }
    .stTextInput > div > div {
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.02em;
        transition: opacity 0.15s ease, transform 0.1s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Helper: load evaluation report JSON
# -----------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_evaluation_report() -> Optional[dict]:
    """
    Load the evaluation report JSON produced by scripts/train_model.py.
    Returns None if the file does not exist yet.

    Returns:
        Dict with all held-out test metrics, or None if not found.
    """
    if not os.path.exists(REPORT_PATH):
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Helper: check API health
# -----------------------------------------------------------------------
def check_api_health() -> tuple[bool, dict]:
    """
    Ping the FastAPI health endpoint.

    Returns:
        Tuple of (is_healthy: bool, health_data: dict).
        health_data is empty if the API is unreachable.
    """
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, {}
    except Exception:
        return False, {}


# -----------------------------------------------------------------------
# Helper: post a transaction to the scoring API
# -----------------------------------------------------------------------
def score_transaction(payload: dict) -> Optional[dict]:
    """
    Send a transaction payload to POST /score and return the JSON response.

    Args:
        payload: Transaction dict matching the TransactionRequest schema.

    Returns:
        Response dict if successful, or None if the request failed.
    """
    try:
        r = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the scoring API. Make sure it is running at http://127.0.0.1:8000")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# -----------------------------------------------------------------------
# Helper: fetch audit log from API
# -----------------------------------------------------------------------
def fetch_audit_log(limit: int = 100, decision_filter: Optional[str] = None) -> list[dict]:
    """
    Fetch recent audit log entries from GET /audit-log.

    Args:
        limit:           Maximum number of entries to return.
        decision_filter: Optional decision type filter string.

    Returns:
        List of audit log entry dicts.
    """
    params: dict = {"limit": limit}
    if decision_filter:
        params["decision_filter"] = decision_filter
    try:
        r = requests.get(f"{API_BASE_URL}/audit-log", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


# -----------------------------------------------------------------------
# Helper: render a metric card via HTML
# -----------------------------------------------------------------------
def metric_card(label: str, value: str, sub: str = "", accent_class: str = "") -> str:
    """
    Generate HTML for a premium metric card.

    Args:
        label:        Uppercase label text above the number.
        value:        Large value string to display.
        sub:          Small subtext below the value.
        accent_class: CSS class for value colour: metric-success, metric-warning, etc.

    Returns:
        HTML string for the metric card.
    """
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {accent_class}">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


# -----------------------------------------------------------------------
# Helper: render a decision badge
# -----------------------------------------------------------------------
def decision_badge(decision: str) -> str:
    """
    Generate HTML for a styled decision badge.

    Args:
        decision: 'approve', 'flag_for_review', or 'block'.

    Returns:
        HTML string for the badge.
    """
    css_map = {
        "approve":         "badge-approve",
        "flag_for_review": "badge-review",
        "block":           "badge-block",
    }
    label_map = {
        "approve":         "Approved",
        "flag_for_review": "Review Required",
        "block":           "Blocked",
    }
    css   = css_map.get(decision, "badge-review")
    label = label_map.get(decision, decision)
    return f'<span class="decision-badge {css}">{label}</span>'


# -----------------------------------------------------------------------
# Page: Overview (Model Performance)
# -----------------------------------------------------------------------
def page_overview() -> None:
    """
    Render the Overview page showing held-out test set metrics, confusion
    matrix, and the false positive cost estimate.
    """
    st.markdown("""
    <div class="section-header">
        <div>
            <h1>Model Performance Overview</h1>
        </div>
        <span class="section-badge">Held-Out Test Set Only</span>
    </div>
    """, unsafe_allow_html=True)

    report = load_evaluation_report()

    if report is None:
        st.markdown("""
        <div class="callout callout-warning">
            <strong>No evaluation report found.</strong>
            Run the training pipeline first: <span class="mono">python scripts/train_model.py</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "ROC AUC",
            f"{report['roc_auc']:.4f}",
            "Area under ROC curve",
            "metric-accent"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Precision",
            f"{report['precision']:.4f}",
            "Of flagged, fraction are real fraud",
            "metric-success"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Recall",
            f"{report['recall']:.4f}",
            "Fraction of all fraud caught",
            "metric-warning"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "F1 Score",
            f"{report['f1_score']:.4f}",
            "Harmonic mean of precision and recall",
            "metric-success"
        ), unsafe_allow_html=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    # Confusion matrix + FP cost side by side
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("#### Confusion Matrix")
        cm = report["confusion_matrix"]
        cm_array = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        labels   = ["Legitimate", "Fraud"]

        fig = go.Figure(data=go.Heatmap(
            z          = cm_array,
            x          = ["Predicted: Legit", "Predicted: Fraud"],
            y          = ["Actual: Legit", "Actual: Fraud"],
            text       = cm_array,
            texttemplate = "%{text}",
            textfont   = {"size": 18, "color": "white"},
            colorscale  = [
                [0,   "#0d1528"],
                [0.4, "#1e3a5f"],
                [1,   "#2563eb"],
            ],
            showscale  = False,
            hovertemplate = "Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"family": "Inter", "color": "#e2e8f0"},
            margin        = {"t": 20, "b": 40, "l": 10, "r": 10},
            height        = 280,
            xaxis         = {"title": None, "tickfont": {"size": 12}},
            yaxis         = {"title": None, "tickfont": {"size": 12}},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("#### Decision Breakdown")
        tp = report["true_positives"]
        fp = report["false_positives"]
        fn = report["false_negatives"]
        tn = report["true_negatives"]

        fig2 = go.Figure(data=[
            go.Bar(
                x        = ["True Neg", "False Pos", "False Neg", "True Pos"],
                y        = [tn, fp, fn, tp],
                marker_color = ["#22c55e", "#ef4444", "#f59e0b", "#3b82f6"],
                text     = [tn, fp, fn, tp],
                textposition = "outside",
                textfont = {"color": "#e2e8f0", "size": 13},
                hovertemplate = "%{x}: %{y}<extra></extra>",
            )
        ])
        fig2.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"family": "Inter", "color": "#e2e8f0"},
            margin        = {"t": 20, "b": 10, "l": 10, "r": 10},
            height        = 280,
            showlegend    = False,
            xaxis         = {"tickfont": {"size": 11}, "gridcolor": "#1e2d4a"},
            yaxis         = {"tickfont": {"size": 11}, "gridcolor": "#1e2d4a"},
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    # False positive cost estimate
    st.markdown("#### False Positive Cost Estimate")
    fp_cost = report.get("fp_cost_total_inr", 0)
    fp_pp   = report.get("fp_cost_per_transaction_inr", 150)
    fp_note = report.get("fp_cost_note", "")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(metric_card(
            "Total FP Cost (Test Set)",
            f"INR {fp_cost:,.0f}",
            f"{fp} false positives on test set",
            "metric-danger"
        ), unsafe_allow_html=True)
    with cc2:
        st.markdown(metric_card(
            "Cost Per False Positive",
            f"INR {fp_pp:,.0f}",
            "Lost revenue + support overhead",
            "metric-warning"
        ), unsafe_allow_html=True)
    with cc3:
        st.markdown(metric_card(
            "Test Set Size",
            f"{report.get('test_set_size', 0):,}",
            f"{report.get('test_set_fraud_count', 0)} fraud cases",
            "metric-accent"
        ), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="callout callout-info" style="margin-top:1rem;">
        <strong>Cost Methodology:</strong> {fp_note}
    </div>
    """, unsafe_allow_html=True)

    # Model metadata footer
    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="callout callout-info">
        <strong>Note:</strong> All metrics above are from the <em>held-out test set</em> only.
        Training metrics are deliberately not reported to avoid inflated estimates.
        Model version: <span class="mono">{report.get('model_version', 'N/A')}</span>.
        Evaluated at: <span class="mono">{report.get('evaluated_at', 'N/A')}</span>.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Page: Live Scoring
# -----------------------------------------------------------------------
def page_live_scoring() -> None:
    """
    Render the Live Scoring page with a manual transaction input form.
    Submits the form data to the FastAPI /score endpoint and displays
    the risk score, decision badge, and plain-language explanation.
    """
    st.markdown("""
    <div class="section-header">
        <div><h1>Live Transaction Scoring</h1></div>
        <span class="section-badge">Real-Time</span>
    </div>
    """, unsafe_allow_html=True)

    # Check API status
    api_ok, health_data = check_api_health()
    if api_ok:
        model_ok = health_data.get("model_loaded", False)
        if model_ok:
            st.markdown('<div class="callout callout-success"><span class="status-dot dot-green"></span> Scoring API is online and model is loaded.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="callout callout-warning"><span class="status-dot dot-yellow"></span> API is running but model is not loaded. Run the training script first.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="callout callout-danger"><span class="status-dot dot-red"></span> Scoring API is offline. Run: <span class="mono">uvicorn api.main:app --port 8000</span></div>', unsafe_allow_html=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        st.markdown("#### Transaction Details")

        with st.form("score_form"):
            txn_id   = st.text_input("Transaction ID", value=f"TXN_LIVE_{int(time.time())}")
            user_id  = st.text_input("User ID", value="USR_00042")

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                amount = st.number_input("Amount (INR)", min_value=1.0, value=5000.0, step=100.0)
            with r1c2:
                payment_method = st.selectbox("Payment Method", PAYMENT_METHODS)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                merchant_cat = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            with r2c2:
                geo_loc = st.selectbox("City", INDIAN_CITIES)

            r3c1, r3c2 = st.columns(2)
            with r3c1:
                count_1h = st.slider("Transactions (last 1h)", 0, 50, 2)
            with r3c2:
                count_24h = st.slider("Transactions (last 24h)", 0, 100, 8)

            r4c1, r4c2 = st.columns(2)
            with r4c1:
                zscore = st.slider("Amount Z-Score vs History", -5.0, 10.0, 0.5, 0.1)
            with r4c2:
                geo_dist = st.slider("Geo Distance from Last Txn (km)", 0.0, 3000.0, 15.0, 5.0)

            is_new_dev = st.checkbox("New Device", value=False)

            submitted = st.form_submit_button("Score This Transaction", use_container_width=True)

    with result_col:
        st.markdown("#### Risk Assessment")

        if submitted:
            payload = {
                "transaction_id":                  txn_id,
                "user_id":                         user_id,
                "amount":                          amount,
                "payment_method":                  payment_method,
                "merchant_category":               merchant_cat,
                "geo_location":                    geo_loc,
                "user_transaction_count_last_1h":  count_1h,
                "user_transaction_count_last_24h": count_24h,
                "amount_zscore_vs_user_history":   zscore,
                "geo_distance_from_last_txn_km":   geo_dist,
                "is_new_device":                   1 if is_new_dev else 0,
            }

            with st.spinner("Scoring transaction..."):
                result = score_transaction(payload)

            if result:
                score    = result["fraud_score"]
                decision = result["decision"]
                expl     = result["explanation"]
                conf     = result["confidence"]
                scored   = result["scored_at"]

                # Gauge chart
                color = DECISION_COLORS.get(decision, "#94a3b8")
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = score * 100,
                    number = {"suffix": "%", "font": {"size": 36, "color": "#f1f5f9", "family": "Inter"}},
                    gauge = {
                        "axis"  : {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
                        "bar"   : {"color": color},
                        "bgcolor": "#0d1528",
                        "bordercolor": "#1e2d4a",
                        "steps": [
                            {"range": [0,   35], "color": "rgba(34,197,94,0.1)"},
                            {"range": [35,  70], "color": "rgba(245,158,11,0.1)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.1)"},
                        ],
                        "threshold": {
                            "line" : {"color": color, "width": 3},
                            "thickness": 0.8,
                            "value": score * 100,
                        },
                    },
                ))
                fig.update_layout(
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font  = {"family": "Inter", "color": "#e2e8f0"},
                    margin = {"t": 30, "b": 20, "l": 20, "r": 20},
                    height = 220,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Decision badge
                st.markdown(
                    f'<div style="text-align:center; margin: 0.5rem 0 1rem 0;">{decision_badge(decision)}</div>',
                    unsafe_allow_html=True
                )

                # Low confidence warning
                if conf == "low_confidence":
                    st.markdown('<div class="callout callout-warning"><strong>Low Confidence:</strong> Some features were missing. Score is a conservative default and transaction has been routed to manual review.</div>', unsafe_allow_html=True)

                # Explanation
                st.markdown(f"""
                <div class="explanation-box">
                    <div class="explanation-label">AI Explanation</div>
                    {expl}
                </div>
                """, unsafe_allow_html=True)

                # Thresholds reference
                thresh = result.get("thresholds_used", {})
                st.markdown(f"""
                <div class="callout callout-info" style="margin-top:0.75rem; font-size:0.8rem;">
                    Thresholds: Flag at {thresh.get('flag_threshold', 0.35):.0%} / Block at {thresh.get('block_threshold', 0.70):.0%}.
                    Scored at <span class="mono">{scored}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:280px; display:flex; align-items:center; justify-content:center; color:#475569; font-size:0.875rem; text-align:center;">
                Fill in the transaction details on the left and click<br><strong>Score This Transaction</strong> to see the risk assessment.
            </div>
            """, unsafe_allow_html=True)

    # Sample transactions section
    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Quick Load: Sample Transactions")
    s1, s2, s3 = st.columns(3)

    sample_txns = [
        {
            "label": "Low Risk Transaction",
            "color": "#22c55e",
            "data": {
                "transaction_id": "SAMPLE_LOW_RISK",
                "user_id": "USR_00001",
                "amount": 850.0,
                "payment_method": "UPI",
                "merchant_category": "grocery",
                "user_transaction_count_last_1h": 1,
                "user_transaction_count_last_24h": 4,
                "amount_zscore_vs_user_history": 0.2,
                "geo_distance_from_last_txn_km": 3.0,
                "is_new_device": 0,
            }
        },
        {
            "label": "Suspicious Transaction",
            "color": "#f59e0b",
            "data": {
                "transaction_id": "SAMPLE_SUSPICIOUS",
                "user_id": "USR_00099",
                "amount": 28000.0,
                "payment_method": "card",
                "merchant_category": "electronics",
                "user_transaction_count_last_1h": 12,
                "user_transaction_count_last_24h": 35,
                "amount_zscore_vs_user_history": 3.8,
                "geo_distance_from_last_txn_km": 1200.0,
                "is_new_device": 1,
            }
        },
        {
            "label": "High Risk Fraud Pattern",
            "color": "#ef4444",
            "data": {
                "transaction_id": "SAMPLE_HIGH_RISK",
                "user_id": "USR_00777",
                "amount": 75000.0,
                "payment_method": "netbanking",
                "merchant_category": "electronics",
                "user_transaction_count_last_1h": 28,
                "user_transaction_count_last_24h": 70,
                "amount_zscore_vs_user_history": 7.2,
                "geo_distance_from_last_txn_km": 2100.0,
                "is_new_device": 1,
            }
        },
    ]

    for col, txn_info in zip([s1, s2, s3], sample_txns):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: {txn_info['color']}; min-height:90px;">
                <div class="metric-label">{txn_info['label']}</div>
                <div class="metric-sub" style="margin-top:0.25rem;">
                    Amount: INR {txn_info['data']['amount']:,.0f}<br>
                    Velocity 1h: {txn_info['data']['user_transaction_count_last_1h']}<br>
                    Z-score: {txn_info['data']['amount_zscore_vs_user_history']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Score Sample", key=f"sample_{txn_info['label']}", use_container_width=True):
                with st.spinner("Scoring..."):
                    result = score_transaction(txn_info["data"])
                if result:
                    st.markdown(f'Decision: {decision_badge(result["decision"])}', unsafe_allow_html=True)
                    st.markdown(f'Score: **{result["fraud_score"]:.4f}**')
                    st.markdown(f'<div class="explanation-box"><div class="explanation-label">Explanation</div>{result["explanation"]}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Page: Audit Log
# -----------------------------------------------------------------------
def page_audit_log() -> None:
    """
    Render the Audit Log page showing recent scored transactions from
    the SQLite audit trail, with filtering by decision type.
    """
    st.markdown("""
    <div class="section-header">
        <div><h1>Audit Trail</h1></div>
        <span class="section-badge">SQLite Backed</span>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        decision_filter = st.selectbox(
            "Filter by Decision",
            options=["All", "approve", "flag_for_review", "block"],
        )
    with fc2:
        limit = st.slider("Max Rows to Show", 10, 500, 100, step=10)
    with fc3:
        refresh = st.button("Refresh", use_container_width=True)

    df_filter = None if decision_filter == "All" else decision_filter
    logs = fetch_audit_log(limit=limit, decision_filter=df_filter)

    if not logs:
        st.markdown("""
        <div class="callout callout-warning">
            No audit log entries found. Score some transactions via the Live Scoring page first.
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary stats row
    df = pd.DataFrame(logs)
    total  = len(df)
    n_app  = (df["decision"] == "approve").sum()
    n_rev  = (df["decision"] == "flag_for_review").sum()
    n_blk  = (df["decision"] == "block").sum()
    n_low  = (df["confidence"] == "low_confidence").sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(metric_card("Total Scored", f"{total:,}", "in this view", "metric-accent"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Approved", f"{n_app:,}", f"{n_app/max(total,1)*100:.1f}%", "metric-success"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Under Review", f"{n_rev:,}", f"{n_rev/max(total,1)*100:.1f}%", "metric-warning"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Blocked", f"{n_blk:,}", f"{n_blk/max(total,1)*100:.1f}%", "metric-danger"), unsafe_allow_html=True)
    with m5:
        st.markdown(metric_card("Low Confidence", f"{n_low:,}", "missing data cases", "metric-warning"), unsafe_allow_html=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    # Decision distribution donut chart
    chart_col, table_col = st.columns([1, 2], gap="large")

    with chart_col:
        st.markdown("#### Decision Distribution")
        decision_counts = df["decision"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels = decision_counts.index.tolist(),
            values = decision_counts.values.tolist(),
            hole   = 0.6,
            marker = {"colors": [
                DECISION_COLORS.get(d, "#94a3b8") for d in decision_counts.index
            ]},
            textinfo = "label+percent",
            textfont = {"size": 12, "color": "#e2e8f0"},
            hovertemplate = "%{label}: %{value}<extra></extra>",
        )])
        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"family": "Inter", "color": "#e2e8f0"},
            margin        = {"t": 20, "b": 20, "l": 20, "r": 20},
            height        = 260,
            showlegend    = False,
            annotations   = [{"text": f"{total}", "x": 0.5, "y": 0.5, "font_size": 22,
                               "showarrow": False, "font_color": "#f1f5f9"}],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with table_col:
        st.markdown("#### Recent Transactions")
        display_df = df[[
            "transaction_id", "scored_at", "fraud_score", "decision",
            "confidence", "amount", "payment_method", "explanation"
        ]].copy()

        # Format columns
        display_df["fraud_score"] = display_df["fraud_score"].map(lambda x: f"{x:.4f}")
        display_df["amount"]      = display_df["amount"].map(lambda x: f"INR {x:,.0f}" if x else "N/A")
        display_df = display_df.rename(columns={
            "transaction_id": "Txn ID",
            "scored_at":      "Scored At",
            "fraud_score":    "Score",
            "decision":       "Decision",
            "confidence":     "Confidence",
            "amount":         "Amount",
            "payment_method": "Method",
            "explanation":    "Explanation",
        })

        st.dataframe(
            display_df,
            use_container_width = True,
            hide_index          = True,
            height              = 260,
            column_config={
                "Explanation": st.column_config.TextColumn(width="large"),
            }
        )

    # Fraud score histogram
    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Fraud Score Distribution")
    scores = df["fraud_score"].astype(float)

    fig2 = go.Figure(data=[go.Histogram(
        x           = scores,
        nbinsx      = 30,
        marker_color = "#3b82f6",
        opacity     = 0.8,
        hovertemplate = "Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
    )])
    fig2.add_vline(x=0.35, line_dash="dash", line_color="#f59e0b",
                   annotation_text="Flag threshold (0.35)", annotation_font_color="#f59e0b")
    fig2.add_vline(x=0.70, line_dash="dash", line_color="#ef4444",
                   annotation_text="Block threshold (0.70)", annotation_font_color="#ef4444")
    fig2.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"family": "Inter", "color": "#e2e8f0"},
        margin        = {"t": 20, "b": 30, "l": 10, "r": 10},
        height        = 220,
        xaxis         = {"title": "Fraud Score", "gridcolor": "#1e2d4a"},
        yaxis         = {"title": "Count",       "gridcolor": "#1e2d4a"},
        bargap        = 0.05,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------
# Page: Graceful Failure Demo
# -----------------------------------------------------------------------
def page_failure_demo() -> None:
    """
    Render the Graceful Failure Demo page showing the system handling
    a transaction with missing data without crashing or making an unsafe
    automatic decision.
    """
    st.markdown("""
    <div class="section-header">
        <div><h1>Graceful Failure Demonstration</h1></div>
        <span class="section-badge">Safety Case</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-info">
        <strong>What this demonstrates:</strong> When a new user has no transaction history yet,
        the velocity and amount z-score features cannot be computed. Rather than crashing or
        making an unsafe automatic decision, SentinelFlow falls back to a conservative default
        score of 0.50 and routes the transaction to <em>manual review</em>. The response is
        clearly marked as <strong>low_confidence</strong> so reviewers know the model did not
        have full information.
    </div>
    """, unsafe_allow_html=True)

    # Show the missing-data transaction that will be sent
    st.markdown("#### Transaction With Missing History")

    missing_txn = {
        "transaction_id": f"DEMO_MISSING_{int(time.time())}",
        "user_id":        "USR_BRAND_NEW",
        "amount":         4200.0,
        "payment_method": "card",
        "merchant_category": "electronics",
        # All velocity and history features are None, simulating a brand-new user
    }

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(metric_card("Transaction ID", missing_txn["transaction_id"][-16:], "Brand new user", ""), unsafe_allow_html=True)
    with mc2:
        st.markdown(metric_card("Amount", f"INR {missing_txn['amount']:,.0f}", missing_txn["merchant_category"], "metric-warning"), unsafe_allow_html=True)
    with mc3:
        st.markdown(metric_card("User History", "NONE", "First transaction on record", "metric-danger"), unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-warning" style="margin-top:0.75rem;">
        <strong>Missing features:</strong> user_transaction_count_last_1h, user_transaction_count_last_24h,
        amount_zscore_vs_user_history, geo_distance_from_last_txn_km
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    demo_col, explanation_col = st.columns([1, 1], gap="large")

    with demo_col:
        st.markdown("#### What Should Happen")
        st.markdown("""
        <div class="metric-card" style="border-top-color: #f59e0b;">
            <div class="metric-label">Expected Behavior</div>
            <ul style="color: #cbd5e1; font-size: 0.875rem; margin: 0.75rem 0 0 0; padding-left: 1.25rem; line-height: 1.9;">
                <li>API must NOT return HTTP 500</li>
                <li>Decision must be <strong style="color:#f59e0b;">flag_for_review</strong></li>
                <li>Confidence must be <strong style="color:#f59e0b;">low_confidence</strong></li>
                <li>Score must be the conservative default (0.50)</li>
                <li>Transaction must still be logged to the audit trail</li>
                <li>No auto-approve or auto-block without full data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Failure Demo", use_container_width=True, key="run_demo"):
            api_ok, _ = check_api_health()
            if not api_ok:
                st.error("Scoring API is not running. Start it first: uvicorn api.main:app --port 8000")
            else:
                with st.spinner("Sending transaction with missing data to API..."):
                    result = score_transaction(missing_txn)

                st.session_state["demo_result"] = result

    with explanation_col:
        st.markdown("#### API Response")
        demo_result = st.session_state.get("demo_result")

        if demo_result is None:
            st.markdown("""
            <div style="height:250px; display:flex; align-items:center; justify-content:center; color:#475569; font-size:0.875rem; text-align:center;">
                Click <strong>Run Failure Demo</strong> to see the graceful<br>failure behavior in action.
            </div>
            """, unsafe_allow_html=True)
        else:
            decision = demo_result.get("decision", "")
            conf     = demo_result.get("confidence", "")
            score    = demo_result.get("fraud_score", 0)

            # Outcome verification
            passed = (
                decision == "flag_for_review" and
                conf     == "low_confidence"  and
                abs(score - 0.5) < 0.01
            )

            outcome_class = "callout-success" if passed else "callout-danger"
            outcome_text  = "All safety checks PASSED. Graceful failure working correctly." if passed else "One or more safety checks FAILED. Check implementation."
            st.markdown(f'<div class="callout {outcome_class}" style="margin-bottom:1rem;"><strong>{"PASS" if passed else "FAIL"}:</strong> {outcome_text}</div>', unsafe_allow_html=True)

            # Show verification checklist
            checks = [
                ("No HTTP 500 error",          True,                              "No crash"),
                ("Decision = flag_for_review", decision == "flag_for_review",     decision),
                ("Confidence = low_confidence",conf == "low_confidence",           conf),
                ("Score = 0.50 (default)",     abs(score - 0.5) < 0.01,           f"{score:.2f}"),
            ]
            for check_label, check_ok, actual_val in checks:
                icon  = "" if check_ok else ""
                color = "#22c55e" if check_ok else "#ef4444"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1e2d4a; font-size:0.85rem;">'
                    f'<span style="color:#94a3b8;">{check_label}</span>'
                    f'<span style="color:{color}; font-weight:600;">{icon} {actual_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"""
            <div class="explanation-box" style="margin-top:1rem;">
                <div class="explanation-label">Explanation Returned</div>
                {demo_result.get('explanation', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

    # Architecture explanation
    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### How Graceful Failure Works (Code Path)")
    st.code("""
# In features/engineering.py: engineer_features_from_row()
count_1h = row.get("user_transaction_count_last_1h")
if count_1h is None:
    count_1h = 5             # conservative default
    has_missing = True       # flag is set

# In api/main.py: score_transaction()
if has_missing:
    fraud_score = 0.5        # conservative fallback
    decision    = "flag_for_review"
    confidence  = "low_confidence"
    explanation = build_low_confidence_explanation(reason)
    # logged to audit trail just like any other transaction
    """, language="python")


# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
def render_sidebar() -> str:
    """
    Render the sidebar with the SentinelFlow branding and navigation menu.

    Returns:
        The selected page name as a string.
    """
    with st.sidebar:
        logo_path = os.path.join(PROJECT_ROOT, "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="brand-logo">
                <div class="brand-icon">🛡️</div>
                <div class="brand-text">
                    <div class="brand-name">SentinelFlow</div>
                    <div class="brand-sub">Fraud Intelligence Platform</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Live Scoring",
                "Audit Log",
                "Failure Demo",
            ],
            label_visibility = "collapsed",
        )

        st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

        # API status indicator in sidebar
        api_ok, health = check_api_health()
        dot_class = "dot-green" if api_ok else "dot-red"
        status_text = "API Online" if api_ok else "API Offline"
        st.markdown(
            f'<div style="font-size:0.78rem; color:#64748b; padding: 0.25rem 0;">'
            f'<span class="status-dot {dot_class}"></span>{status_text}'
            f'</div>',
            unsafe_allow_html=True
        )

        report = load_evaluation_report()
        if report:
            model_ver = report.get("model_version", "N/A")
            st.markdown(
                f'<div style="font-size:0.78rem; color:#64748b; padding:0.1rem 0;">'
                f'Model v{model_ver}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7rem; color:#334155; line-height:1.6;">
            Defense-only system. No functionality that could assist fraud execution or evasion.
        </div>
        """, unsafe_allow_html=True)

    return page


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
def main() -> None:
    """
    Dashboard entry point. Injects CSS, renders the sidebar, then routes
    to the appropriate page component based on the navigation selection.
    """
    inject_css()

    if "demo_result" not in st.session_state:
        st.session_state["demo_result"] = None

    page = render_sidebar()

    if page == "Overview":
        page_overview()
    elif page == "Live Scoring":
        page_live_scoring()
    elif page == "Audit Log":
        page_audit_log()
    elif page == "Failure Demo":
        page_failure_demo()


if __name__ == "__main__":
    main()
