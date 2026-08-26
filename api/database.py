"""
SentinelFlow: SQLite Audit Log Database Layer
=============================================
Handles all database operations for the transaction audit trail.
Every scored transaction is persisted here, with no exceptions.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional


# Database file is stored alongside the API package
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.db")
DB_PATH = os.path.normpath(DB_PATH)


def get_connection() -> sqlite3.Connection:
    """
    Open and return a SQLite database connection with row factory set to
    sqlite3.Row so rows can be accessed by column name.

    Returns:
        sqlite3.Connection with row_factory set.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the audit_log table if it does not already exist.
    This is safe to call multiple times (idempotent).

    The table stores every scored transaction with:
      - transaction_id: unique identifier for the transaction
      - scored_at:      ISO 8601 timestamp of when the score was produced
      - fraud_score:    model output probability (0 to 1)
      - decision:       one of 'approve', 'flag_for_review', or 'block'
      - explanation:    plain-language explanation of the decision
      - model_version:  version string of the model that produced the score
      - confidence:     'normal' or 'low_confidence' (for missing data cases)
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id  TEXT NOT NULL,
                scored_at       TEXT NOT NULL,
                fraud_score     REAL NOT NULL,
                decision        TEXT NOT NULL,
                explanation     TEXT NOT NULL,
                model_version   TEXT NOT NULL,
                confidence      TEXT NOT NULL DEFAULT 'normal',
                amount          REAL,
                payment_method  TEXT,
                user_id         TEXT
            )
        """)
    conn.close()


def log_transaction(
    transaction_id: str,
    fraud_score:    float,
    decision:       str,
    explanation:    str,
    model_version:  str,
    confidence:     str = "normal",
    amount:         Optional[float] = None,
    payment_method: Optional[str]   = None,
    user_id:        Optional[str]   = None,
) -> int:
    """
    Insert a scored transaction record into the audit_log table.
    This is called for every transaction that passes through the scoring API,
    including low-confidence fallback cases.

    Args:
        transaction_id: Unique identifier from the transaction payload.
        fraud_score:    Model fraud probability score between 0 and 1.
        decision:       'approve', 'flag_for_review', or 'block'.
        explanation:    Plain-language explanation text.
        model_version:  Model version string (e.g. '1.0.0').
        confidence:     'normal' or 'low_confidence'.
        amount:         Transaction amount in INR (optional).
        payment_method: Payment method string (optional).
        user_id:        User identifier (optional).

    Returns:
        The auto-generated row ID for the inserted record.
    """
    conn  = get_connection()
    now   = datetime.utcnow().isoformat() + "Z"
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_log
                (transaction_id, scored_at, fraud_score, decision,
                 explanation, model_version, confidence, amount,
                 payment_method, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (transaction_id, now, fraud_score, decision,
             explanation, model_version, confidence, amount,
             payment_method, user_id),
        )
        row_id = cursor.lastrowid
    conn.close()
    return row_id


def fetch_recent_logs(limit: int = 100, decision_filter: Optional[str] = None) -> list[dict]:
    """
    Retrieve the most recent audit log entries, optionally filtered by
    decision type.

    Args:
        limit:           Maximum number of rows to return.
        decision_filter: If provided, only return rows where decision matches
                         this value exactly ('approve', 'flag_for_review', or 'block').

    Returns:
        List of dicts, one per audit log row, most recent first.
    """
    conn = get_connection()

    if decision_filter:
        rows = conn.execute(
            """
            SELECT * FROM audit_log
            WHERE decision = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (decision_filter, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def count_by_decision() -> dict[str, int]:
    """
    Return a summary count of all decisions recorded in the audit log.
    Used by the dashboard overview section.

    Returns:
        Dict mapping decision string to count, e.g.
        {'approve': 120, 'flag_for_review': 15, 'block': 5}.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT decision, COUNT(*) AS cnt FROM audit_log GROUP BY decision"
    ).fetchall()
    conn.close()
    return {row["decision"]: row["cnt"] for row in rows}
