# -*- coding: utf-8 -*-
"""
Compare clustering configurations:
 - rule-only (binary) vs rule-only (weighted)
 - rule+RFM (binary) vs rule+RFM (weighted)
 - Top-K small vs Top-K large (by using subsets of rule columns)

Saves: data/processed/config_comparison.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC = os.path.join(ROOT, 'data', 'processed')

# Load files
bin_path = os.path.join(PROC, 'customer_rule_features_binary.csv')
wt_path = os.path.join(PROC, 'customer_rule_features_weighted.csv')
rfm_path = os.path.join(PROC, 'customer_rule_features_rfm.csv')

binary = pd.read_csv(bin_path)
weighted = pd.read_csv(wt_path)
rfm = pd.read_csv(rfm_path)

# ensure CustomerID aligned
binary = binary.set_index('CustomerID')
weighted = weighted.set_index('CustomerID')
rfm = rfm.set_index('CustomerID')

# rule columns
rule_cols_bin = [c for c in binary.columns if str(c).startswith('rule_')]
rule_cols_wt = [c for c in weighted.columns if str(c).startswith('rule_')]

# intersection of customers
ids = list(set(binary.index) & set(weighted.index) & set(rfm.index))
ids = sorted(ids)

binary = binary.loc[ids]
weighted = weighted.loc[ids]
rfm = rfm.loc[ids]

# K range for clusters
k_range = range(2, 11)

results = []

# helper to evaluate matrix X
def eval_X(X, label_prefix):
    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    best_k = None
    best_score = -1
    best_inertia = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        if len(set(labels)) <= 1:
            score = -1
        else:
            score = silhouette_score(Xs, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_inertia = float(km.inertia_)
    return best_k, best_score, best_inertia

# Top-K variants
avail = max(1, len(rule_cols_bin))
topk_list = sorted(set([min(5, avail), min(20, avail), avail]))
topk_list = [k for k in topk_list if k > 0]

for topk in topk_list:
    cols_bin_top = rule_cols_bin[:topk]
    cols_wt_top = rule_cols_wt[:topk]

    # 1) rule-only binary
    if len(cols_bin_top) > 0:
        X_bin = binary[cols_bin_top].fillna(0).values
        bk, bs, bi = eval_X(X_bin, f'rule-bin-k{topk}')
        # actionable metric: avg rule activations per customer
        actionable_bin = float(binary[cols_bin_top].sum(axis=1).mean())
        results.append({
            'config': 'rule-only-binary',
            'topk': topk,
            'best_k': bk,
            'silhouette': bs,
            'inertia': bi,
            'actionable_avg_rules_per_customer': actionable_bin,
        })

    # 2) rule-only weighted
    if len(cols_wt_top) > 0:
        X_wt = weighted[cols_wt_top].fillna(0).values
        bk, bs, bi = eval_X(X_wt, f'rule-wt-k{topk}')
        actionable_wt = float((weighted[cols_wt_top] > 0).sum(axis=1).mean())
        results.append({
            'config': 'rule-only-weighted',
            'topk': topk,
            'best_k': bk,
            'silhouette': bs,
            'inertia': bi,
            'actionable_avg_rules_per_customer': actionable_wt,
        })

    # 3) rule+RFM binary
    rfmc = rfm[['Recency','Frequency','Monetary']].fillna(0)
    if len(cols_bin_top) > 0:
        X_bin_rfm = pd.concat([binary[cols_bin_top].fillna(0), rfmc], axis=1).values
        bk, bs, bi = eval_X(X_bin_rfm, f'rule-bin-rfm-k{topk}')
        actionable = float(binary[cols_bin_top].sum(axis=1).mean())
        results.append({
            'config': 'rule+RFM-binary',
            'topk': topk,
            'best_k': bk,
            'silhouette': bs,
            'inertia': bi,
            'actionable_avg_rules_per_customer': actionable,
        })

    # 4) rule+RFM weighted
    if len(cols_wt_top) > 0:
        X_wt_rfm = pd.concat([weighted[cols_wt_top].fillna(0), rfmc], axis=1).values
        bk, bs, bi = eval_X(X_wt_rfm, f'rule-wt-rfm-k{topk}')
        actionable = float((weighted[cols_wt_top] > 0).sum(axis=1).mean())
        results.append({
            'config': 'rule+RFM-weighted',
            'topk': topk,
            'best_k': bk,
            'silhouette': bs,
            'inertia': bi,
            'actionable_avg_rules_per_customer': actionable,
        })

# save
    # Ensure CSV has headers even if results is empty
    out_path = os.path.join(PROC, 'config_comparison.csv')
    if not results:
        cols = ['config','topk','best_k','silhouette','inertia','actionable_avg_rules_per_customer']
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
    else:
        out = pd.DataFrame(results)
        out.to_csv(out_path, index=False)
    print('Wrote', out_path)
