# -*- coding: utf-8 -*-
"""
Clustering pipeline using association-rule features + RFM.
Saves:
 - data/processed/customer_rule_features_binary.csv (if not present)
 - data/processed/customer_rule_features_weighted.csv
 - data/processed/customer_rule_features_rfm.csv
 - data/processed/customer_clusters.csv
 - data/processed/cluster_pca.png
 - data/processed/cluster_profile.csv

Usage:
    python src/cluster_pipeline.py

This re-uses `apriori_library.py` and `cluster_from_rules.py`.
"""
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ensure repo src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT)

from apriori_library import DataCleaner
try:
    from cluster_from_rules import load_and_clean, prepare_basket, mine_rules, build_customer_binary_features
except Exception:
    # If direct import fails, we'll call DataCleaner directly
    load_and_clean = None


PROC_DIR = os.path.join(ROOT, "data", "processed")
RAW_PATH = os.path.join(ROOT, "data", "raw", "online_retail.csv")


def ensure_binary_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Return customer binary features; generate if missing."""
    feat_path = os.path.join(PROC_DIR, "customer_rule_features_binary.csv")
    if os.path.exists(feat_path):
        return pd.read_csv(feat_path)

    # try to use helper from cluster_from_rules if available
    if load_and_clean is not None:
        # mine a small set of rules (top 20) and build features
        basket_bool = prepare_basket(df_clean)
        rules = mine_rules(basket_bool=basket_bool, min_support=0.01, min_lift=1.2, top_k=20)
        features = build_customer_binary_features(df_clean, basket_bool, rules)
        features.reset_index().to_csv(feat_path, index=False)
        return features.reset_index()

    raise FileNotFoundError(f"Binary features not found at {feat_path} and helper unavailable.")


def compute_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner(RAW_PATH)
    cleaner.df = df_clean  # DataCleaner expects df to be loaded
    # ensure df_uk is set
    cleaner.df_uk = df_clean
    rfm = cleaner.compute_rfm()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"] if "CustomerID" not in rfm.columns else rfm.columns
    return rfm


def build_weighted_features(binary_df: pd.DataFrame, rules_df: Optional[pd.DataFrame] = None, weight_type: str = "lift") -> pd.DataFrame:
    """Create weighted features by multiplying binary indicator by rule weight.
    weight_type: 'lift' or 'lift_conf' (lift * confidence)
    """
    # binary_df: columns CustomerID + rule_xxx
    features = binary_df.set_index("CustomerID")
    weights = None
    rules_path = os.path.join(PROC_DIR, "rules_apriori_filtered.csv")
    if os.path.exists(rules_path):
        rules_df = pd.read_csv(rules_path)
    if rules_df is None:
        # fallback: uniform weights
        weights = pd.Series(1.0, index=features.columns)
    else:
        # map rules order to columns by antecedents_str
        col_map = {}
        for i, row in rules_df.iterrows():
            key = f"rule_{i}__{row.get('antecedents_str','')}_to_{row.get('consequents_str','')}"
            if key in features.columns:
                if weight_type == "lift":
                    col_map[key] = float(row.get("lift", 1.0))
                else:
                    col_map[key] = float(row.get("lift", 1.0)) * float(row.get("confidence", 1.0))
        # any unmatched columns -> 1.0
        weights = pd.Series({c: col_map.get(c, 1.0) for c in features.columns})

    weighted = features.multiply(weights, axis=1)
    weighted = weighted.reset_index()
    outp = os.path.join(PROC_DIR, "customer_rule_features_weighted.csv")
    os.makedirs(PROC_DIR, exist_ok=True)
    weighted.to_csv(outp, index=False)
    return weighted


def combine_with_rfm(features_df: pd.DataFrame, rfm_df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
    df = features_df.merge(rfm_df, on="CustomerID", how="left")
    df = df.fillna(0)
    cols = [c for c in df.columns if c not in ("CustomerID",)]
    if scale:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    outp = os.path.join(PROC_DIR, "customer_rule_features_rfm.csv")
    df.to_csv(outp, index=False)
    return df


def choose_k_and_cluster(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[KMeans, int, dict]:
    results = {}
    best_k = k_min
    best_score = -1
    best_model = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            score = -1
        else:
            score = silhouette_score(X, labels)
        results[k] = {"inertia": float(km.inertia_), "silhouette": float(score)}
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km
    return best_model, best_k, results


def pca_plot(X: np.ndarray, labels: np.ndarray, outpath: str):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    plt.title("PCA 2D of customers (colored by cluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def profile_clusters(df_features_rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df_features_rfm.copy()
    df["cluster"] = labels
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    numeric = [c for c in numeric if c not in ("cluster")]
    profile = df.groupby("cluster")[numeric].mean()
    profile["n_customers"] = df.groupby("cluster").size()
    outp = os.path.join(PROC_DIR, "cluster_profile.csv")
    profile.reset_index().to_csv(outp, index=False)
    return profile


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    print("Loading and cleaning data...")
    if load_and_clean is not None:
        df_clean = load_and_clean(RAW_PATH)
    else:
        cleaner = DataCleaner(RAW_PATH)
        df_clean = cleaner.load_data()
        df_clean = cleaner.clean_data()

    print("Ensure binary rule features...")
    binary = ensure_binary_features(df_clean)

    print("Compute RFM...")
    rfm = compute_rfm(df_clean)

    print("Build weighted features...")
    weighted = build_weighted_features(binary, None, weight_type="lift_conf")

    print("Combine features with RFM and scale...")
    combined = combine_with_rfm(weighted, rfm, scale=True)

    X = combined.drop(columns=["CustomerID"]).values

    print("Choosing K (2..10) using silhouette score...")
    model, best_k, results = choose_k_and_cluster(X, 2, 10)
    print(f"Best K selected: {best_k}")

    print("Fitting final KMeans and saving labels...")
    labels = model.predict(X)
    out_labels = os.path.join(PROC_DIR, "customer_clusters.csv")
    df_labels = pd.DataFrame({"CustomerID": combined["CustomerID"], "cluster": labels})
    df_labels.to_csv(out_labels, index=False)

    print("PCA plot...")
    pca_out = os.path.join(PROC_DIR, "cluster_pca.png")
    pca_plot(X, labels, pca_out)

    print("Profiling clusters...")
    profile = profile_clusters(combined, labels)
    print(profile)

    # save results summary
    summary_path = os.path.join(PROC_DIR, "k_selection_results.csv")
    pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index":"k"}).to_csv(summary_path, index=False)
    print(f"Saved outputs into {PROC_DIR}")


if __name__ == "__main__":
    main()
