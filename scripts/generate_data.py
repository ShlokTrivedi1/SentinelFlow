"""
SentinelFlow: Synthetic Transaction Data Generator
===================================================
Generates a realistic synthetic dataset of UPI and card payment transactions
with injected fraud patterns. Saves the result to data/transactions.csv.

Fraud patterns injected:
  1. Velocity spikes: many transactions in a short window
  2. Geo jumps: transactions from distant cities within a short time
  3. New device combined with high amount
  4. Amount outliers vs user history (high z-score)

This script is strictly for data generation. It contains no logic that
could assist with real fraud execution or evasion.
"""

import os
import random
import math
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Fixed seed for full reproducibility across runs
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)

# -----------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------
TOTAL_ROWS       = 16000          # Total synthetic transactions to generate
FRAUD_RATE       = 0.03           # Target fraud rate (3 percent)
NUM_USERS        = 800            # Unique user IDs in the dataset
NUM_DEVICES      = 1200           # Unique device fingerprints
START_DATE       = datetime(2024, 1, 1)
END_DATE         = datetime(2024, 6, 30)

PAYMENT_METHODS   = ["UPI", "card", "netbanking"]
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "clothing", "food_delivery",
    "travel", "entertainment", "fuel", "healthcare", "utilities", "education"
]

# Indian cities used for geo_location column
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Bhopal", "Indore"
]

# Approximate lat/lon for each city (used to compute geo distance)
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
    Compute the great-circle distance in kilometers between two geographic
    coordinates using the Haversine formula.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometers (float).
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def random_ip() -> str:
    """Generate a random IPv4 address string."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def generate_base_transactions(n: int) -> pd.DataFrame:
    """
    Generate the base (non-fraud) synthetic transaction records.
    Each user has a realistic spending distribution, devices, and cities.

    Args:
        n: Number of base transactions to generate.

    Returns:
        DataFrame with raw transaction fields before feature derivation.
    """
    user_ids     = [f"USR_{i:05d}" for i in range(NUM_USERS)]
    device_ids   = [f"DEV_{i:06d}" for i in range(NUM_DEVICES)]

    # Pre-assign each user a home city and a set of familiar devices
    user_home_city: dict[str, str] = {u: random.choice(INDIAN_CITIES) for u in user_ids}
    user_devices: dict[str, list[str]] = {
        u: random.sample(device_ids, k=random.randint(1, 3)) for u in user_ids
    }

    # Per-user spending baseline (mean and std in INR)
    user_avg_amount: dict[str, float] = {u: random.uniform(200, 8000) for u in user_ids}
    user_std_amount: dict[str, float] = {u: ua * random.uniform(0.2, 0.5) for u, ua in user_avg_amount.items()}

    records = []
    timestamp_range_seconds = int((END_DATE - START_DATE).total_seconds())

    for i in range(n):
        uid    = random.choice(user_ids)
        ts     = START_DATE + timedelta(seconds=random.randint(0, timestamp_range_seconds))
        amount = max(10.0, round(np.random.normal(user_avg_amount[uid], user_std_amount[uid]), 2))
        city   = random.choices(
            [user_home_city[uid]] + INDIAN_CITIES,
            weights=[0.75] + [0.25 / len(INDIAN_CITIES)] * len(INDIAN_CITIES),
            k=1
        )[0]
        device = random.choice(user_devices[uid])

        records.append({
            "transaction_id":    f"TXN_{i:07d}",
            "user_id":           uid,
            "timestamp":         ts,
            "amount":            amount,
            "payment_method":    random.choice(PAYMENT_METHODS),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "device_id":         device,
            "ip_address":        random_ip(),
            "geo_location":      city,
            "_user_avg_amount":  user_avg_amount[uid],
            "_user_std_amount":  user_std_amount[uid],
            "_user_home_city":   user_home_city[uid],
            "_familiar_devices": user_devices[uid],
            "is_fraud":          0,
        })

    return pd.DataFrame(records)


def inject_fraud_patterns(df: pd.DataFrame, target_fraud_rate: float) -> pd.DataFrame:
    """
    Select a subset of rows and overwrite their fields to match known fraud
    patterns. Four distinct patterns are injected.

    Patterns:
      A. Velocity spike: many transactions in a very short window.
      B. Geo jump: transaction from a distant city shortly after a home-city one.
      C. New device combined with high transaction amount.
      D. Amount outlier vs user spending history (high z-score).

    Args:
        df:                 DataFrame of base (non-fraud) transactions.
        target_fraud_rate:  Desired fraction of rows to mark as fraud (0 to 1).

    Returns:
        DataFrame with fraud rows injected and is_fraud column updated.
    """
    n_fraud     = int(len(df) * target_fraud_rate)
    fraud_idxs  = random.sample(list(df.index), n_fraud)

    # Split fraud budget across four patterns
    split       = n_fraud // 4
    pattern_A   = fraud_idxs[:split]
    pattern_B   = fraud_idxs[split:2 * split]
    pattern_C   = fraud_idxs[2 * split:3 * split]
    pattern_D   = fraud_idxs[3 * split:]

    # Pattern A: velocity spike (high transaction count in last 1 hour)
    for idx in pattern_A:
        df.at[idx, "is_fraud"] = 1
        # The feature engineering will read these override columns
        df.at[idx, "_override_count_1h"]  = random.randint(15, 40)
        df.at[idx, "_override_count_24h"] = random.randint(30, 80)

    # Pattern B: geo jump (transaction far from home city)
    for idx in pattern_B:
        df.at[idx, "is_fraud"] = 1
        home_city   = df.at[idx, "_user_home_city"]
        far_cities  = [c for c in INDIAN_CITIES if c != home_city]
        df.at[idx, "geo_location"]        = random.choice(far_cities)
        df.at[idx, "_override_geo_dist"]  = random.uniform(800, 2500)

    # Pattern C: new device combined with high amount
    for idx in pattern_C:
        df.at[idx, "is_fraud"] = 1
        df.at[idx, "device_id"]           = f"DEV_NEW_{idx:06d}"
        df.at[idx, "amount"]              = round(random.uniform(15000, 80000), 2)
        df.at[idx, "_override_new_device"] = True

    # Pattern D: amount outlier vs user history (z-score > 4)
    for idx in pattern_D:
        df.at[idx, "is_fraud"]  = 1
        ua = df.at[idx, "_user_avg_amount"]
        us = max(df.at[idx, "_user_std_amount"], 100.0)
        df.at[idx, "amount"]    = round(ua + random.uniform(4.5, 8.0) * us, 2)
        df.at[idx, "_override_zscore"] = random.uniform(4.5, 8.0)

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived ML features from the raw transaction DataFrame.
    This mirrors the logic in features/engineering.py and is used only
    during data generation to produce the CSV with pre-computed labels.

    Args:
        df: DataFrame with raw and override columns.

    Returns:
        DataFrame with the final feature columns ready for model training.
    """
    # Sort by user and time so rolling windows make sense
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Velocity: transactions per user in the last 1 hour and 24 hours
    # For generation we approximate using per-user daily distribution
    # and override values where fraud patterns were injected.
    user_counts = df.groupby("user_id").cumcount() + 1

    # Count-1h: random realistic baseline, overridden for fraud rows
    df["user_transaction_count_last_1h"] = np.random.randint(1, 5, size=len(df))
    if "_override_count_1h" in df.columns:
        mask = df["_override_count_1h"].notna()
        df.loc[mask, "user_transaction_count_last_1h"] = df.loc[mask, "_override_count_1h"].astype(int)

    # Count-24h: slightly larger baseline
    df["user_transaction_count_last_24h"] = np.random.randint(1, 12, size=len(df))
    if "_override_count_24h" in df.columns:
        mask = df["_override_count_24h"].notna()
        df.loc[mask, "user_transaction_count_last_24h"] = df.loc[mask, "_override_count_24h"].astype(int)

    # Amount z-score vs user history baseline
    df["amount_zscore_vs_user_history"] = (
        (df["amount"] - df["_user_avg_amount"]) / df["_user_std_amount"].clip(lower=1.0)
    ).round(4)
    if "_override_zscore" in df.columns:
        mask = df["_override_zscore"].notna()
        df.loc[mask, "amount_zscore_vs_user_history"] = df.loc[mask, "_override_zscore"]

    # Geo distance from last transaction (in km)
    df["geo_distance_from_last_txn_km"] = np.random.uniform(0, 50, size=len(df)).round(2)
    if "_override_geo_dist" in df.columns:
        mask = df["_override_geo_dist"].notna()
        df.loc[mask, "geo_distance_from_last_txn_km"] = df.loc[mask, "_override_geo_dist"].round(2)

    # is_new_device flag
    familiar_device_set: dict[str, set] = {}
    for _, row in df.iterrows():
        uid = row["user_id"]
        if uid not in familiar_device_set:
            familiar_device_set[uid] = set(row["_familiar_devices"])

    df["is_new_device"] = df.apply(
        lambda r: 1 if r["device_id"] not in familiar_device_set.get(r["user_id"], set()) else 0,
        axis=1
    )
    if "_override_new_device" in df.columns:
        mask = df["_override_new_device"] == True
        df.loc[mask, "is_new_device"] = 1

    return df


def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order only the columns that form the final public dataset.
    Drops all internal override and helper columns.

    Args:
        df: DataFrame after fraud injection and feature derivation.

    Returns:
        Clean DataFrame with exactly the specified column schema.
    """
    keep_cols = [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "payment_method",
        "merchant_category",
        "device_id",
        "ip_address",
        "geo_location",
        "user_transaction_count_last_1h",
        "user_transaction_count_last_24h",
        "amount_zscore_vs_user_history",
        "geo_distance_from_last_txn_km",
        "is_new_device",
        "is_fraud",
    ]
    return df[keep_cols].copy()


def main() -> None:
    """
    Entry point: generates the synthetic dataset and writes it to
    data/transactions.csv. Prints a summary upon completion.
    """
    print("=" * 60)
    print("SentinelFlow: Synthetic Data Generator")
    print("=" * 60)
    print(f"Generating {TOTAL_ROWS:,} base transactions...")

    df = generate_base_transactions(TOTAL_ROWS)

    print(f"Injecting fraud patterns (target rate: {FRAUD_RATE * 100:.1f}%)...")
    df = inject_fraud_patterns(df, target_fraud_rate=FRAUD_RATE)

    print("Deriving ML features...")
    df = derive_features(df)

    df = build_final_dataset(df)

    # Shuffle so fraud rows are not clumped at the end
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    fraud_count    = df["is_fraud"].sum()
    actual_rate    = fraud_count / len(df) * 100
    print(f"\nDataset saved to: {out_path}")
    print(f"Total rows      : {len(df):,}")
    print(f"Fraud rows      : {fraud_count:,} ({actual_rate:.2f}%)")
    print(f"Legit rows      : {len(df) - fraud_count:,}")
    print("\nColumn list:")
    for col in df.columns:
        print(f"  {col}")
    print("\nDone.")


if __name__ == "__main__":
    main()
