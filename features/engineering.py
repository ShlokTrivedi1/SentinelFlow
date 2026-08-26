"""
SentinelFlow: Reusable Feature Engineering Module
==================================================
Computes all ML features from raw transaction data.

This module is the single source of truth for feature logic.
It is imported by:
  - scripts/train_model.py (batch training)
  - api/main.py (live scoring per request)

No duplication of feature logic between training and serving.
"""

import math
from typing import Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Feature names used consistently across training and serving
# -----------------------------------------------------------------------
FEATURE_COLUMNS = [
    "amount",
    "user_transaction_count_last_1h",
    "user_transaction_count_last_24h",
    "amount_zscore_vs_user_history",
    "geo_distance_from_last_txn_km",
    "is_new_device",
    "payment_method_encoded",
    "merchant_category_encoded",
]

PAYMENT_METHOD_MAP: dict[str, int] = {
    "UPI":        0,
    "card":       1,
    "netbanking": 2,
}

MERCHANT_CATEGORY_MAP: dict[str, int] = {
    "grocery":       0,
    "electronics":   1,
    "clothing":      2,
    "food_delivery": 3,
    "travel":        4,
    "entertainment": 5,
    "fuel":          6,
    "healthcare":    7,
    "utilities":     8,
    "education":     9,
}

# Approximate lat/lon for Indian cities used in geo distance computation
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Mumbai":    (19.076, 72.877),
    "Delhi":     (28.613, 77.209),
    "Bangalore": (12.971, 77.594),
    "Hyderabad": (17.385, 78.486),
    "Chennai":   (13.083, 80.270),
    "Kolkata":   (22.572, 88.363),
    "Pune":      (18.520, 73.856),
    "Ahmedabad": (23.023, 72.572),
    "Jaipur":    (26.912, 75.787),
    "Surat":     (21.170, 72.831),
    "Lucknow":   (26.847, 80.947),
    "Kanpur":    (26.449, 80.331),
    "Nagpur":    (21.145, 79.088),
    "Bhopal":    (23.259, 77.413),
    "Indore":    (22.719, 75.857),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine great-circle distance in kilometers between two
    geographic points specified as decimal degree coordinates.

    Args:
        lat1: Latitude of the first point.
        lon1: Longitude of the first point.
        lat2: Latitude of the second point.
        lon2: Longitude of the second point.

    Returns:
        Distance in kilometers as a float.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(max(0.0, a))), 2)


def get_city_coords(city: str) -> Optional[tuple[float, float]]:
    """
    Look up the approximate geographic coordinates for a known Indian city.

    Args:
        city: City name string.

    Returns:
        A (latitude, longitude) tuple if the city is in the lookup table,
        otherwise None.
    """
    return CITY_COORDS.get(city, None)


def encode_payment_method(method: str) -> int:
    """
    Convert a payment method string to its integer code.
    Unknown values are mapped to 0 (UPI as the most common fallback).

    Args:
        method: Payment method string such as 'UPI', 'card', or 'netbanking'.

    Returns:
        Integer encoding for the payment method.
    """
    return PAYMENT_METHOD_MAP.get(method, 0)


def encode_merchant_category(category: str) -> int:
    """
    Convert a merchant category string to its integer code.
    Unknown categories fall back to 0 (grocery).

    Args:
        category: Merchant category string.

    Returns:
        Integer encoding for the merchant category.
    """
    return MERCHANT_CATEGORY_MAP.get(category, 0)


def compute_amount_zscore(
    amount: float,
    user_avg: float,
    user_std: float,
) -> float:
    """
    Compute how many standard deviations the current transaction amount
    deviates from the user's historical average.

    Args:
        amount:   Current transaction amount in INR.
        user_avg: User's historical average transaction amount.
        user_std: User's historical standard deviation of transaction amounts.

    Returns:
        Z-score as a float. Returns 0.0 if user_std is very small to avoid
        division by near-zero.
    """
    if user_std < 1.0:
        return 0.0
    return round((amount - user_avg) / user_std, 4)


def compute_geo_distance(city_current: str, city_previous: str) -> float:
    """
    Compute the geographic distance in kilometers between two city names
    by looking up their approximate coordinates in the city table.

    If either city is not in the lookup table, returns 0.0 as a safe default.

    Args:
        city_current:  The city of the current transaction.
        city_previous: The city of the immediately preceding transaction.

    Returns:
        Distance in km between the two cities (0.0 if either is unknown).
    """
    coords1 = get_city_coords(city_current)
    coords2 = get_city_coords(city_previous)
    if coords1 is None or coords2 is None:
        return 0.0
    return haversine_km(coords1[0], coords1[1], coords2[0], coords2[1])


def engineer_features_from_row(row: dict) -> dict:
    """
    Compute all ML features for a single transaction dictionary.
    This is the function called by the live scoring API for each request.

    The input dict must contain at least these keys:
      - amount (float)
      - payment_method (str)
      - merchant_category (str)
      - user_transaction_count_last_1h (int or None)
      - user_transaction_count_last_24h (int or None)
      - amount_zscore_vs_user_history (float or None)
      - geo_distance_from_last_txn_km (float or None)
      - is_new_device (int, 0 or 1)

    Returns a dict with keys matching FEATURE_COLUMNS, suitable for model
    inference. Missing values are filled with conservative defaults and a
    boolean flag 'has_missing_data' is set to True.

    Args:
        row: Dictionary of raw transaction fields.

    Returns:
        Dict of engineered features plus 'has_missing_data' boolean.
    """
    has_missing = False
    features: dict = {}

    # Core amount feature (required, non-nullable)
    features["amount"] = float(row.get("amount", 0.0))

    # Velocity features: use provided value or default to a conservative high value
    count_1h = row.get("user_transaction_count_last_1h")
    if count_1h is None:
        count_1h = 5          # conservative unknown baseline
        has_missing = True
    features["user_transaction_count_last_1h"] = int(count_1h)

    count_24h = row.get("user_transaction_count_last_24h")
    if count_24h is None:
        count_24h = 10        # conservative unknown baseline
        has_missing = True
    features["user_transaction_count_last_24h"] = int(count_24h)

    # Amount z-score vs user history
    z_score = row.get("amount_zscore_vs_user_history")
    if z_score is None:
        z_score = 0.0
        has_missing = True
    features["amount_zscore_vs_user_history"] = float(z_score)

    # Geo distance from last transaction
    geo_dist = row.get("geo_distance_from_last_txn_km")
    if geo_dist is None:
        geo_dist = 0.0
        has_missing = True
    features["geo_distance_from_last_txn_km"] = float(geo_dist)

    # New device flag
    features["is_new_device"] = int(row.get("is_new_device", 0))

    # Categorical encodings
    features["payment_method_encoded"]    = encode_payment_method(row.get("payment_method", "UPI"))
    features["merchant_category_encoded"] = encode_merchant_category(row.get("merchant_category", "grocery"))

    features["has_missing_data"] = has_missing
    return features


def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to an entire DataFrame in batch mode.
    Used during model training to convert the raw CSV into model-ready features.

    Encodes categorical columns and ensures all FEATURE_COLUMNS are present
    and in the correct order.

    Args:
        df: Raw transactions DataFrame loaded from data/transactions.csv.

    Returns:
        DataFrame with FEATURE_COLUMNS and the 'is_fraud' label column.
    """
    out = df.copy()

    # Encode categorical columns
    out["payment_method_encoded"]    = out["payment_method"].map(PAYMENT_METHOD_MAP).fillna(0).astype(int)
    out["merchant_category_encoded"] = out["merchant_category"].map(MERCHANT_CATEGORY_MAP).fillna(0).astype(int)

    # Ensure numeric columns are the right dtype
    out["user_transaction_count_last_1h"]  = out["user_transaction_count_last_1h"].fillna(5).astype(int)
    out["user_transaction_count_last_24h"] = out["user_transaction_count_last_24h"].fillna(10).astype(int)
    out["amount_zscore_vs_user_history"]   = out["amount_zscore_vs_user_history"].fillna(0.0).astype(float)
    out["geo_distance_from_last_txn_km"]   = out["geo_distance_from_last_txn_km"].fillna(0.0).astype(float)
    out["is_new_device"]                   = out["is_new_device"].fillna(0).astype(int)
    out["amount"]                          = out["amount"].astype(float)

    return out[FEATURE_COLUMNS + ["is_fraud"]]
