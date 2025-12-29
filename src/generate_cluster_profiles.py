# -*- coding: utf-8 -*-
"""
Generate human-readable cluster profiles with English + Vietnamese names,
personas and marketing suggestions.
Reads:
 - data/processed/customer_clusters.csv
 - data/processed/customer_rule_features_binary.csv
 - data/processed/customer_rule_features_rfm.csv
 - data/processed/rules_apriori_filtered.csv
Writes:
 - data/processed/cluster_profiles_named.csv
"""
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC = os.path.join(ROOT, 'data', 'processed')

clusters_path = os.path.join(PROC, 'customer_clusters.csv')
binary_path = os.path.join(PROC, 'customer_rule_features_binary.csv')
rfm_path = os.path.join(PROC, 'customer_rule_features_rfm.csv')
rules_path = os.path.join(PROC, 'rules_apriori_filtered.csv')
out_path = os.path.join(PROC, 'cluster_profiles_named.csv')

# load
clusters = pd.read_csv(clusters_path)
binary = pd.read_csv(binary_path)
rfm = pd.read_csv(rfm_path)
rules = pd.read_csv(rules_path)

# merge features with clusters and rfm
df = clusters.merge(binary, on='CustomerID', how='left')
if 'Recency' in rfm.columns:
    df = df.merge(rfm[['CustomerID','Recency','Frequency','Monetary']], on='CustomerID', how='left')
else:
    df = df.merge(rfm, on='CustomerID', how='left')

# fillna
df[['Recency','Frequency','Monetary']] = df[['Recency','Frequency','Monetary']].fillna(0)

# determine cluster-level statistics
group = df.groupby('cluster')
# compute median RFM across all customers for heuristic naming
rfms_median = df[['Recency','Frequency','Monetary']].median()
profiles = []
for c, g in group:
    n_customers = len(g)
    recency = float(g['Recency'].mean())
    frequency = float(g['Frequency'].mean())
    monetary = float(g['Monetary'].mean())

    # top rule columns: columns that start with 'rule_'
    rule_cols = [col for col in g.columns if str(col).startswith('rule_')]
    top_rules = []
    if rule_cols:
        # sum across customers to get prevalence
        sums = g[rule_cols].sum().sort_values(ascending=False)
        top_cols = sums.head(5)
        for col in top_cols.index:
            # try to map back to rule string from rules file
            # rules file has antecedents_str and consequents_str and rule_str
            # our col names are like 'rule_{i}__antecedents_to_consequents'
            try:
                idx = int(col.split('__')[0].split('rule_')[1])
                rule_str = None
                if idx in rules.index:
                    rule_str = rules.loc[idx].get('rule_str')
                if not rule_str:
                    # fallback to cleaned column name
                    rule_str = col
            except Exception:
                rule_str = col
            top_rules.append(str(rule_str))

    # name clusters heuristically
    # score by recency low (recent), frequency high, monetary high
    score = (1.0/(1+recency)) * (1+frequency) * (1+monetary)

    if monetary > rfms_median['Monetary']*1.2 and frequency > rfms_median['Frequency']*1.2:
        name_en = 'High Value'
        name_vn = 'Khách giá trị cao'
        persona = 'High-spend frequent buyers; likely VIP.'
        marketing = 'VIP care, upsell premium bundles, loyalty rewards.'
    elif recency < rfms_median['Recency']*0.8 and frequency > rfms_median['Frequency']:
        name_en = 'Recent Loyal'
        name_vn = 'Khách mới trung thành'
        persona = 'Recently active with repeated purchases.'
        marketing = 'Cross-sell complementary items; encourage repeat purchases.'
    elif frequency < rfms_median['Frequency']*0.6 and monetary < rfms_median['Monetary']*0.6:
        name_en = 'Occasional Low Spender'
        name_vn = 'Khách ít mua, chi tiêu thấp'
        persona = 'Buys infrequently and spends little.'
        marketing = 'Promotional bundles and discounts to increase conversion.'
    elif recency > rfms_median['Recency']*1.5:
        name_en = 'Churn Risk'
        name_vn = 'Nguy cơ rời bỏ'
        persona = 'Hasn\'t purchased recently.'
        marketing = 'Reactivation campaigns, win-back offers.'
    else:
        name_en = 'Core Segment'
        name_vn = 'Tập khách chính'
        persona = 'Average customers; steady purchase behavior.'
        marketing = 'Cross-sell, personalized recommendations.'

    profiles.append({
        'cluster': int(c),
        'n_customers': int(n_customers),
        'recency_mean': recency,
        'frequency_mean': frequency,
        'monetary_mean': monetary,
        'top_rules': '; '.join(top_rules),
        'name_en': name_en,
        'name_vn': name_vn,
        'persona_one_line': persona,
        'marketing_recommendations': marketing,
    })

# write output
out = pd.DataFrame(profiles).sort_values('cluster')
out.to_csv(out_path, index=False)
print('Wrote', out_path)
