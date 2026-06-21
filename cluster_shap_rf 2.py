"""
Cluster Feature Selector — Random Forest + TreeSHAP
=====================================================
Trains a binary Random Forest classifier:
    positive class  →  samples that belong to TARGET_CLUSTER_ID
    negative class  →  all other samples

Then uses TreeSHAP to rank every feature by its mean |SHAP| value
and returns the TOP_K most discriminating features for that cluster.

Dataset (place in the SAME folder as this script):
    DARPA_OpTC_50k_ProcessTree-2.csv

Install:
    pip install scikit-learn shap pandas numpy

Run with default cluster 0:
    python cluster_shap_rf.py

Run targeting a different cluster (e.g. cluster 3):
    python cluster_shap_rf.py --cluster 3
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← change TARGET_CLUSTER_ID here, or pass --cluster N at runtime
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR         = Path(__file__).resolve().parent
CSV_FILENAME       = "DARPA_OpTC_50k_ProcessTree-2.csv"
CSV_PATH           = SCRIPT_DIR / CSV_FILENAME

TARGET_CLUSTER_ID  = 0          # default cluster to explain  ← edit here
TOP_K              = 3          # number of top features to return
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 12,
    min_samples_leaf = 4,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    class_weight     = "balanced",   # handles cluster vs. rest imbalance
)

# Candidate names for the cluster column (first match wins)
CLUSTER_COL_CANDIDATES = [
    "cluster", "Cluster", "cluster_id", "ClusterID",
    "cluster_label", "label", "Label", "y",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: Path, target_cluster: int):
    """
    Returns
    -------
    X_train, X_test  : pd.DataFrame  — numeric feature matrices
    y_train, y_test  : pd.Series     — binary labels (1 = target cluster)
    feature_names    : list[str]
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  STEP 1 — Load & Preprocess")
    print(sep)
    print(f"  File    : {csv_path}")

    if not csv_path.exists():
        sys.exit(
            f"\n[ERROR] Dataset not found:\n  {csv_path}\n"
            f"Place '{CSV_FILENAME}' in the same folder as this script.\n"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Shape   : {df.shape}")

    # ── detect cluster column ─────────────────────────────────────────────
    cluster_col = next(
        (c for c in CLUSTER_COL_CANDIDATES if c in df.columns), None
    )
    if cluster_col is None:
        cluster_col = df.columns[-1]
        print(f"  [WARN] No cluster column found by name; "
              f"using last column: '{cluster_col}'")
    else:
        print(f"  Cluster column : '{cluster_col}'")

    # Cluster distribution
    dist = df[cluster_col].value_counts().sort_index()
    print(f"\n  Cluster distribution:\n{dist.to_string()}\n")

    available = sorted(df[cluster_col].unique())
    if target_cluster not in available:
        sys.exit(
            f"\n[ERROR] Cluster {target_cluster} not found in '{cluster_col}'.\n"
            f"Available cluster IDs: {available}\n"
        )

    # ── binary label ──────────────────────────────────────────────────────
    y_raw = df[cluster_col]
    y     = (y_raw == target_cluster).astype(int)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"  Target         : Cluster {target_cluster}")
    print(f"  In-cluster     : {n_pos:,} samples  (label = 1)")
    print(f"  Out-of-cluster : {n_neg:,} samples  (label = 0)")

    # ── feature matrix ────────────────────────────────────────────────────
    X = df.drop(columns=[cluster_col]).copy()

    # Drop columns with a single unique value (carry no information)
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        print(f"\n  Dropping {len(constant_cols)} constant column(s): {constant_cols}")
        X.drop(columns=constant_cols, inplace=True)

    # Label-encode any object/category columns
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    X.fillna(0, inplace=True)
    feature_names = list(X.columns)
    print(f"\n  Features after preprocessing : {len(feature_names)}")

    # ── stratified train/test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        target_cluster: int) -> RandomForestClassifier:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  STEP 2 — Train Random Forest  (Cluster {target_cluster} vs. Rest)")
    print(sep)
    for k, v in RF_PARAMS.items():
        print(f"  {k:<22}: {v}")

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print(f"\n  Training complete — {clf.n_estimators} trees built.")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf: RandomForestClassifier,
             X_test: pd.DataFrame, y_test: pd.Series,
             target_cluster: int) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  STEP 3 — Evaluation on test set  ({len(y_test):,} samples)")
    print(sep)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC : {auc:.4f}")
    except Exception:
        print("  ROC-AUC : N/A (only one class present in test set)")

    report = classification_report(
        y_test, y_pred,
        target_names=["Other Clusters", f"Cluster {target_cluster}"],
        digits=4,
        zero_division=0,
    )
    print(f"\n{report}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TreeSHAP → top-K features
# ─────────────────────────────────────────────────────────────────────────────

def _extract_class1_shap(shap_values) -> np.ndarray:
    """
    Normalise whatever shap.TreeExplainer.shap_values() returns into a
    strict 2-D array  (n_samples, n_features)  for the positive class.

    Handles all known SHAP output formats:
        list of 2 arrays  →  [class0(n,f), class1(n,f)]   older SHAP
        3-D ndarray       →  (n, f, 2)                    SHAP >= 0.42
        2-D ndarray       →  (n, f)                       edge / single-output
    """
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])      # positive-class matrix
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:                   # (n_samples, n_features, n_classes)
            sv = sv[:, :, 1]

    sv = sv.squeeze()
    if sv.ndim == 1:                       # single test sample
        sv = sv[np.newaxis, :]
    return sv                              # (n_samples, n_features)


def compute_top_features(clf: RandomForestClassifier,
                         X_test: pd.DataFrame,
                         feature_names: list,
                         target_cluster: int,
                         top_k: int = TOP_K):
    """
    Returns
    -------
    top_df   : pd.DataFrame  — top_k rows  [feature, mean_abs_shap]
    full_df  : pd.DataFrame  — all features ranked by mean_abs_shap
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  STEP 4 — TreeSHAP  (Cluster {target_cluster})")
    print(sep)
    print(f"  Computing SHAP values for {len(X_test):,} test samples ...")

    # tree_path_dependent — no background needed, robust across SHAP versions
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    sv = _extract_class1_shap(shap_values)
    print(f"  SHAP matrix shape : {sv.shape}  (samples × features)")

    # Mean absolute SHAP per feature — guaranteed 1-D
    mean_abs_shap = np.abs(sv).mean(axis=0).flatten()

    # Ranked DataFrame
    full_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    full_df.index += 1    # rank starts at 1

    # Console preview
    preview_n = min(10, len(full_df))
    print(f"\n  Feature ranking — top {preview_n} shown:")
    print(f"  {'Rank':<6} {'Feature':<40} {'Mean |SHAP|':>14}")
    print(f"  {'-'*62}")
    for rank, row in full_df.head(preview_n).iterrows():
        marker = "  ◄ TOP" if rank <= top_k else ""
        print(f"  {rank:<6} {row['feature']:<40} "
              f"{row['mean_abs_shap']:>14.6f}{marker}")

    top_df = full_df.head(top_k).copy()
    return top_df, full_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Report & save
# ─────────────────────────────────────────────────────────────────────────────

def report_and_save(top_df: pd.DataFrame, full_df: pd.DataFrame,
                    target_cluster: int, top_k: int) -> list:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  RESULT — Top {top_k} Features for Cluster {target_cluster}  (TreeSHAP)")
    print(sep)
    for rank, row in top_df.iterrows():
        print(f"  #{rank}  {row['feature']:<42}  "
              f"mean|SHAP| = {row['mean_abs_shap']:.6f}")
    print(sep)

    out_top  = SCRIPT_DIR / f"cluster{target_cluster}_top{top_k}_features.csv"
    out_full = SCRIPT_DIR / f"cluster{target_cluster}_full_shap_ranking.csv"
    top_df.to_csv(out_top)
    full_df.to_csv(out_full, index_label="rank")
    print(f"\n  Saved : {out_top}")
    print(f"  Saved : {out_full}\n")

    return list(top_df["feature"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> int:
    parser = argparse.ArgumentParser(
        description="Binary RF + TreeSHAP feature selector for a given cluster."
    )
    parser.add_argument(
        "--cluster", type=int, default=TARGET_CLUSTER_ID,
        help=f"Cluster ID to explain (default: {TARGET_CLUSTER_ID})"
    )
    args, _ = parser.parse_known_args()
    return args.cluster


def main() -> list:
    target_cluster = parse_args()

    print(f"\n  *** Targeting Cluster {target_cluster} vs. Rest ***")

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        CSV_PATH, target_cluster
    )
    clf = train_random_forest(X_train, y_train, target_cluster)
    evaluate(clf, X_test, y_test, target_cluster)
    top_df, full_df = compute_top_features(
        clf, X_test, feature_names, target_cluster, top_k=TOP_K
    )
    top3 = report_and_save(top_df, full_df, target_cluster, TOP_K)

    print(f"  → Top {TOP_K} features of Cluster {target_cluster} : {top3}\n")
    return top3


if __name__ == "__main__":
    main()
