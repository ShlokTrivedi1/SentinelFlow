"""
SentinelFlow: Model Training Script
====================================
Trains an XGBoost fraud classifier on the synthetic transactions dataset.

Key principles:
  - All metrics reported are from the HELD-OUT TEST SET only. Training set
    metrics are NOT reported and must not be confused with real performance.
  - A fixed random seed ensures all results are fully reproducible.
  - class_weight / scale_pos_weight handles the class imbalance.
  - A false positive cost estimate is computed to quantify business impact.

Outputs (written to results/):
  - model.json: Trained XGBoost model in JSON format
  - shap_explainer.pkl: SHAP TreeExplainer pickled for API reuse
  - evaluation_report.json: All held-out test metrics and cost estimate
  - confusion_matrix.png: Visualisation of held-out confusion matrix
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import shap

# Suppress warnings that are not actionable in this context
warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root to sys.path so 'features' can be imported
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.engineering import FEATURE_COLUMNS, engineer_features_batch

# -----------------------------------------------------------------------
# Reproducibility seed: changing this value will change split and results
# -----------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# Business cost parameters for the false positive cost estimate
# -----------------------------------------------------------------------
# Each false positive (legitimate transaction wrongly blocked) is estimated
# to cost the business INR 150 in lost revenue and support overhead.
# This is a configurable business assumption, not a model parameter.
FP_COST_PER_TRANSACTION_INR = 150.0

# -----------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "model.json")
EXPLAINER_PATH = os.path.join(PROJECT_ROOT, "models", "shap_explainer.pkl")
REPORT_PATH  = os.path.join(RESULTS_DIR, "evaluation_report.json")
CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
MODEL_VERSION = "1.0.0"


def load_and_prepare_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the transactions CSV, apply feature engineering in batch mode,
    and return the feature matrix X and label series y.

    Args:
        data_path: Absolute path to transactions.csv.

    Returns:
        Tuple of (X: DataFrame of features, y: Series of is_fraud labels).
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows. Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")

    engineered = engineer_features_batch(df)
    X = engineered[FEATURE_COLUMNS]
    y = engineered["is_fraud"]
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier with scale_pos_weight to handle the
    class imbalance between legitimate and fraudulent transactions.

    scale_pos_weight = (number of negative samples) / (number of positive samples)
    This tells XGBoost to penalise misclassifying the minority fraud class
    more heavily, effectively up-weighting fraud examples.

    Args:
        X_train:          Training feature DataFrame.
        y_train:          Training label Series.
        scale_pos_weight: Ratio of negatives to positives in the training set.

    Returns:
        Fitted XGBClassifier instance.
    """
    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        random_state      = RANDOM_SEED,
        eval_metric       = "logloss",
        use_label_encoder = False,
        verbosity         = 0,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the trained model ONLY on the held-out test set and return a
    dictionary of all performance metrics.

    IMPORTANT: These are held-out test metrics, NOT training metrics.
    Training metrics would be inflated and should never be reported as
    the model's real-world performance.

    Args:
        model:  Trained XGBClassifier.
        X_test: Held-out test feature DataFrame.
        y_test: Held-out test label Series.

    Returns:
        Dict containing precision, recall, f1, roc_auc, confusion_matrix,
        fp_count, fn_count, tp_count, tn_count, fp_cost_inr, and a
        threshold note.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Default XGBoost threshold is 0.5 for predict()
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_pred_proba)
    cm    = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # False positive cost estimate (business impact of wrongly blocking legit txns)
    fp_cost_total = fp * FP_COST_PER_TRANSACTION_INR

    return {
        "evaluation_set"             : "held_out_test_set",
        "note"                       : "All metrics are from the held-out test set. Training metrics are not reported.",
        "model_version"              : MODEL_VERSION,
        "evaluated_at"               : datetime.now().isoformat(),
        "precision"                  : round(prec, 4),
        "recall"                     : round(rec, 4),
        "f1_score"                   : round(f1, 4),
        "roc_auc"                    : round(auc, 4),
        "confusion_matrix"           : {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "true_positives"             : int(tp),
        "false_positives"            : int(fp),
        "false_negatives"            : int(fn),
        "true_negatives"             : int(tn),
        "fp_cost_per_transaction_inr": FP_COST_PER_TRANSACTION_INR,
        "fp_cost_total_inr"          : round(fp_cost_total, 2),
        "fp_cost_note"               : (
            f"Each false positive (legitimate transaction wrongly blocked) is estimated "
            f"to cost INR {FP_COST_PER_TRANSACTION_INR:.0f} in lost revenue and support "
            f"overhead. Total FP cost on the held-out test set: INR {fp_cost_total:,.2f}."
        ),
        "test_set_size"              : int(len(y_test)),
        "test_set_fraud_count"       : int(y_test.sum()),
    }


def plot_confusion_matrix(metrics: dict, save_path: str) -> None:
    """
    Render and save a styled confusion matrix heatmap as a PNG file.

    Args:
        metrics:   The metrics dictionary returned by compute_metrics.
        save_path: Absolute path where the PNG should be saved.
    """
    cm_dict = metrics["confusion_matrix"]
    cm_array = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit",    "Actual Fraud"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix (Held-Out Test Set)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def build_shap_explainer(model: XGBClassifier, X_sample: pd.DataFrame) -> shap.TreeExplainer:
    """
    Construct a SHAP TreeExplainer for the trained XGBoost model.
    The explainer is serialised to disk so the API can reuse it without
    retraining.

    Args:
        model:    Trained XGBClassifier.
        X_sample: A sample of training data used to initialise the explainer.

    Returns:
        SHAP TreeExplainer instance.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def print_summary(metrics: dict) -> None:
    """
    Print a clean, formatted summary of all held-out test metrics to stdout.

    Args:
        metrics: Dict returned by compute_metrics.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SentinelFlow: Model Evaluation Report")
    print(f"  {metrics['note']}")
    print(sep)
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1_score']:.4f}")
    print(f"  ROC AUC       : {metrics['roc_auc']:.4f}")
    print(f"{sep}")
    print("  Confusion Matrix (Held-Out Test Set):")
    cm = metrics["confusion_matrix"]
    print(f"    True Negatives  (Legit correctly approved) : {cm['tn']:,}")
    print(f"    False Positives (Legit wrongly blocked)    : {cm['fp']:,}")
    print(f"    False Negatives (Fraud missed)             : {cm['fn']:,}")
    print(f"    True Positives  (Fraud correctly caught)   : {cm['tp']:,}")
    print(sep)
    print("  False Positive Cost Estimate:")
    print(f"    {metrics['fp_cost_note']}")
    print(sep)
    print(f"  Model version     : {metrics['model_version']}")
    print(f"  Evaluated at      : {metrics['evaluated_at']}")
    print(sep)


def main() -> None:
    """
    Entry point: runs the full training pipeline end to end.
    Steps: load data, split, train, evaluate on test set, save artifacts.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Load and engineer features
    X, y = load_and_prepare_data(DATA_PATH)

    # 2. Stratified train/test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        stratify     = y,
        random_state = RANDOM_SEED,
    )
    print(f"Train set: {len(X_train):,} rows  |  Test set: {len(X_test):,} rows")
    print(f"Train fraud rate: {y_train.mean() * 100:.2f}%  |  Test fraud rate: {y_test.mean() * 100:.2f}%")

    # 3. Compute scale_pos_weight to handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = round(n_neg / max(n_pos, 1), 2)
    print(f"scale_pos_weight: {spw} (negatives / positives = {n_neg} / {n_pos})")

    # 4. Train the XGBoost model
    print("Training XGBoost model...")
    model = train_model(X_train, y_train, scale_pos_weight=spw)
    print("Training complete.")

    # 5. Evaluate on held-out test set only
    print("Evaluating on held-out test set...")
    metrics = compute_metrics(model, X_test, y_test)

    # 6. Print results
    print_summary(metrics)

    # 7. Build and save SHAP explainer
    print("Building SHAP explainer (this may take a moment)...")
    explainer = build_shap_explainer(model, X_train.head(500))

    # 8. Save model artifact
    model.save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # 9. Save SHAP explainer
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)
    print(f"SHAP explainer saved to: {EXPLAINER_PATH}")

    # 10. Save evaluation report as JSON
    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to: {REPORT_PATH}")

    # 11. Save confusion matrix plot
    plot_confusion_matrix(metrics, CM_PLOT_PATH)

    print("\nTraining pipeline complete. All artifacts saved.")


if __name__ == "__main__":
    main()
