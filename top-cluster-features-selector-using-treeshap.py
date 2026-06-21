"""
cluster_shap_rf.py
==================
Public API
----------
    from cluster_shap_rf import get_top_features

    top3 = get_top_features(dataset, cluster_id=0)
    # returns: ["feature_a", "feature_b", "feature_c"]

Parameters
----------
dataset    : str | Path | pd.DataFrame
                CSV file path  OR  an already-loaded DataFrame.
                Must contain a cluster column (auto-detected) plus numeric/
                categorical feature columns.
cluster_id : int  (default 0)
                The cluster whose top-3 discriminating features you want.
top_k      : int  (default 3)
                How many top features to return.
verbose    : bool (default True)
                Set False to suppress all console output.

Returns
-------
list[str]  — feature names ranked by mean absolute TreeSHAP value.

Standalone usage
----------------
    python cluster_shap_rf.py --dataset path/to/data.csv --cluster 2

Install
-------
    pip install scikit-learn shap pandas numpy
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Internal defaults (all overridable via get_top_features kwargs)
# ─────────────────────────────────────────────────────────────────────────────

_RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 12,
    min_samples_leaf = 4,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = 42,
    class_weight     = "balanced",
)

_TEST_SIZE    = 0.20
_RANDOM_STATE = 42

# Candidate names for the cluster column — first match wins
_CLUSTER_COL_CANDIDATES = [
    "cluster", "Cluster", "cluster_id", "ClusterID",
    "cluster_label", "label", "Label", "y",
]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def _load_dataframe(dataset: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Accept a file path or an already-loaded DataFrame."""
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()
    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _detect_cluster_col(df: pd.DataFrame) -> str:
    """Return the name of the cluster column, or raise ValueError."""
    match = next((c for c in _CLUSTER_COL_CANDIDATES if c in df.columns), None)
    if match:
        return match
    # last-resort: last column
    return df.columns[-1]


def _preprocess(
    df: pd.DataFrame,
    cluster_col: str,
    cluster_id: int,
    verbose: bool,
):
    """
    Build binary labels and a numeric feature matrix.

    Returns X_train, X_test, y_train, y_test, feature_names
    """
    available = sorted(df[cluster_col].unique())
    if cluster_id not in available:
        raise ValueError(
            f"Cluster {cluster_id} not found in column '{cluster_col}'. "
            f"Available IDs: {available}"
        )

    y = (df[cluster_col] == cluster_id).astype(int)
    X = df.drop(columns=[cluster_col]).copy()

    # Drop constant columns
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    if constant:
        _log(f"  [preprocess] Dropping {len(constant)} constant column(s).", verbose)
        X.drop(columns=constant, inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = _TEST_SIZE,
        random_state = _RANDOM_STATE,
        stratify     = y,
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def _train(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(**_RF_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def _evaluate(clf, X_test, y_test, cluster_id: int, verbose: bool) -> None:
    if not verbose:
        return
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC : {auc:.4f}")
    except Exception:
        print("  ROC-AUC : N/A")
    print(classification_report(
        y_test, y_pred,
        target_names=["Other Clusters", f"Cluster {cluster_id}"],
        digits=4, zero_division=0,
    ))


def _extract_class1_shap(shap_values) -> np.ndarray:
    """
    Normalise shap_values to (n_samples, n_features) for the positive class.

    Handles all known SHAP output formats:
        list of 2 arrays  →  [class0(n,f), class1(n,f)]   older SHAP
        3-D ndarray       →  (n, f, 2)                    SHAP >= 0.42
        2-D ndarray       →  (n, f)                       single-output
    """
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:          # (n_samples, n_features, n_classes)
            sv = sv[:, :, 1]

    sv = sv.squeeze()
    if sv.ndim == 1:              # single test sample edge-case
        sv = sv[np.newaxis, :]
    return sv                     # (n_samples, n_features)


def _treeshap_top_k(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
    top_k: int,
    verbose: bool,
) -> list[str]:
    """Run TreeSHAP and return the top_k feature names."""
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    sv            = _extract_class1_shap(shap_values)          # (n, f)
    mean_abs_shap = np.abs(sv).mean(axis=0).flatten()          # (f,)

    ranking = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1   # rank starts at 1

    if verbose:
        preview = min(10, len(ranking))
        print(f"\n  {'Rank':<6} {'Feature':<40} {'Mean |SHAP|':>14}")
        print(f"  {'-'*62}")
        for rank, row in ranking.head(preview).iterrows():
            marker = "  ◄" if rank <= top_k else ""
            print(f"  {rank:<6} {row['feature']:<40} "
                  f"{row['mean_abs_shap']:>14.6f}{marker}")

    return list(ranking.head(top_k)["feature"])


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_top_features(
    dataset: Union[str, Path, pd.DataFrame],
    cluster_id: int = 0,
    top_k: int = 3,
    verbose: bool = True,
) -> list[str]:
    """
    Train a one-vs-all Random Forest for *cluster_id* and return the
    top *top_k* most discriminating feature names ranked by TreeSHAP.

    Parameters
    ----------
    dataset    : file path (str / Path) or a pandas DataFrame
    cluster_id : which cluster to explain  (default: 0)
    top_k      : number of top features to return  (default: 3)
    verbose    : print progress and evaluation metrics  (default: True)

    Returns
    -------
    list[str]  —  top feature names, most discriminating first

    Examples
    --------
    # From a CSV file
    top3 = get_top_features("DARPA_OpTC_50k_ProcessTree-2.csv", cluster_id=0)

    # From an already-loaded DataFrame
    import pandas as pd
    df   = pd.read_csv("DARPA_OpTC_50k_ProcessTree-2.csv")
    top3 = get_top_features(df, cluster_id=2, verbose=False)

    print(top3)
    # ['feature_x', 'feature_y', 'feature_z']
    """
    sep = "=" * 65

    # ── 1. load ───────────────────────────────────────────────────────────
    _log(f"\n{sep}", verbose)
    _log(f"  get_top_features | cluster_id={cluster_id} | top_k={top_k}", verbose)
    _log(sep, verbose)

    df = _load_dataframe(dataset)
    _log(f"  Dataset shape  : {df.shape}", verbose)

    cluster_col = _detect_cluster_col(df)
    _log(f"  Cluster column : '{cluster_col}'", verbose)

    dist = df[cluster_col].value_counts().sort_index()
    _log(f"\n  Cluster distribution:\n{dist.to_string()}\n", verbose)

    # ── 2. preprocess ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = _preprocess(
        df, cluster_col, cluster_id, verbose
    )
    _log(f"  Features : {len(feature_names)}  |  "
         f"Train : {len(X_train):,}  |  Test : {len(X_test):,}", verbose)

    # ── 3. train ──────────────────────────────────────────────────────────
    _log(f"\n  Training Random Forest (Cluster {cluster_id} vs. Rest) ...", verbose)
    clf = _train(X_train, y_train)
    _log(f"  Done — {clf.n_estimators} trees built.", verbose)

    # ── 4. evaluate ───────────────────────────────────────────────────────
    _log(f"\n  Evaluation:", verbose)
    _evaluate(clf, X_test, y_test, cluster_id, verbose)

    # ── 5. TreeSHAP ───────────────────────────────────────────────────────
    _log(f"\n  TreeSHAP feature ranking:", verbose)
    top_features = _treeshap_top_k(clf, X_test, feature_names, top_k, verbose)

    # ── 6. result ─────────────────────────────────────────────────────────
    _log(f"\n{sep}", verbose)
    _log(f"  TOP {top_k} FEATURES — Cluster {cluster_id}", verbose)
    _log(sep, verbose)
    for i, feat in enumerate(top_features, 1):
        _log(f"  #{i}  {feat}", verbose)
    _log(sep, verbose)

    return top_features


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Binary RF + TreeSHAP — top-K features for a given cluster."
    )
    parser.add_argument(
        "--dataset", type=str,
        default="DARPA_OpTC_50k_ProcessTree-2.csv",
        help="Path to the CSV dataset (default: DARPA_OpTC_50k_ProcessTree-2.csv)"
    )
    parser.add_argument(
        "--cluster", type=int, default=0,
        help="Target cluster ID (default: 0)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top features to return (default: 3)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    top3    = get_top_features(
        dataset    = args.dataset,
        cluster_id = args.cluster,
        top_k      = args.top_k,
        verbose    = not args.quiet,
    )
    print(f"\nResult: {top3}")
