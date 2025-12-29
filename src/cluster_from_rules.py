# -*- coding: utf-8 -*-
"""
Minimal pipeline to mine association rules and build customer-level features.
Produces:
 - data/processed/rules_apriori_filtered.csv
 - data/processed/customer_rule_features_binary.csv

This script re-uses classes in `apriori_library.py`.
"""
import argparse
import os
import sys
from typing import List

import pandas as pd


# make sure local src is importable when running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT)

from apriori_library import (
    DataCleaner,
    BasketPreparer,
    AssociationRulesMiner,
)


def load_and_clean(raw_path: str):
    cleaner = DataCleaner(raw_path)
    df = cleaner.load_data()
    df_uk = cleaner.clean_data()
    return df_uk


def prepare_basket(df: pd.DataFrame):
    preparer = BasketPreparer(df=df, invoice_col="InvoiceNo", item_col="Description")
    basket = preparer.create_basket()
    basket_bool = preparer.encode_basket(threshold=1)
    return basket_bool


def mine_rules(basket_bool: pd.DataFrame, min_support: float, min_lift: float, top_k: int):
    miner = AssociationRulesMiner(basket_bool=basket_bool)
    miner.mine_frequent_itemsets(min_support=min_support)
    # generate rules using lift as primary metric (min_lift used below to filter)
    miner.generate_rules(metric="lift", min_threshold=1.0)
    miner.add_readable_rule_str()
    rules = miner.filter_rules(min_support=min_support, min_lift=min_lift)
    if rules.empty:
        print("No rules found with given thresholds.")
        return rules

    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules_top = rules.head(top_k).reset_index(drop=True)
    return rules_top


def build_customer_binary_features(df: pd.DataFrame, basket_bool: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    # map invoice -> customer
    inv2cust = df[["InvoiceNo", "CustomerID"]].drop_duplicates().set_index("InvoiceNo")["CustomerID"]

    customers = sorted(df["CustomerID"].unique())
    features = pd.DataFrame(index=customers)

    for i, row in rules.iterrows():
        antecedent = row["antecedents"]
        col_name = f"rule_{i}__{row.get('antecedents_str','')}_to_{row.get('consequents_str','')}"
        try:
            items = list(antecedent)
            if not items:
                continue
            # check per-invoice whether all antecedent items present
            inv_has = basket_bool[items].all(axis=1)
        except Exception:
            # fallback for unexpected item names
            inv_has = pd.Series(False, index=basket_bool.index)

        inv_has = inv_has.astype(bool)
        inv_df = pd.DataFrame({"InvoiceNo": inv_has.index, "has": inv_has.values})
        inv_df = inv_df.set_index("InvoiceNo")
        # map to customer
        inv_df["CustomerID"] = inv_df.index.map(inv2cust)
        cust_has = inv_df.reset_index().dropna(subset=["CustomerID"]).groupby("CustomerID")["has"].any().astype(int)

        features[col_name] = cust_has

    features = features.fillna(0).astype(int)
    features.index.name = "CustomerID"
    return features


def main(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Mine rules and build customer features")
    p.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "online_retail.csv"))
    p.add_argument("--min_support", type=float, default=0.01)
    p.add_argument("--min_lift", type=float, default=1.2)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args(argv)

    print("Loading and cleaning data...")
    df_clean = load_and_clean(args.raw)

    print("Preparing basket...")
    basket_bool = prepare_basket(df_clean)

    print(f"Mining rules (min_support={args.min_support}, min_lift={args.min_lift})...")
    rules_top = mine_rules(basket_bool=basket_bool, min_support=args.min_support, min_lift=args.min_lift, top_k=args.top_k)

    out_rules_path = os.path.join(ROOT, "data", "processed", "rules_apriori_filtered.csv")
    os.makedirs(os.path.dirname(out_rules_path), exist_ok=True)
    rules_top.to_csv(out_rules_path, index=False)
    print(f"Saved top rules to {out_rules_path}")

    print("Building customer-level binary features from antecedents...")
    features = build_customer_binary_features(df_clean, basket_bool, rules_top)
    out_feat = os.path.join(ROOT, "data", "processed", "customer_rule_features_binary.csv")
    features.reset_index().to_csv(out_feat, index=False)
    print(f"Saved customer features to {out_feat}")


if __name__ == "__main__":
    main()
