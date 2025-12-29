# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC = os.path.join(ROOT, 'data', 'processed')

@st.cache_data
def load_data():
    clusters = pd.read_csv(os.path.join(PROC, 'customer_clusters.csv'))
    binary = pd.read_csv(os.path.join(PROC, 'customer_rule_features_binary.csv'))
    profiles = pd.read_csv(os.path.join(PROC, 'cluster_profiles_named.csv'))
    rules = pd.read_csv(os.path.join(PROC, 'rules_apriori_filtered.csv'))

    weighted = None
    comp = None
    profile_summary = None

    wt_path = os.path.join(PROC, 'customer_rule_features_weighted.csv')
    comp_path = os.path.join(PROC, 'config_comparison.csv')
    profile_path = os.path.join(PROC, 'cluster_profile.csv')

    if os.path.exists(wt_path):
        try:
            weighted = pd.read_csv(wt_path)
        except Exception:
            weighted = None
    if os.path.exists(comp_path):
        try:
            comp = pd.read_csv(comp_path)
        except Exception:
            comp = None
    if os.path.exists(profile_path):
        try:
            profile_summary = pd.read_csv(profile_path)
        except Exception:
            profile_summary = None

    return clusters, binary, profiles, rules, weighted, comp, profile_summary

clusters, binary, profiles, rules, weighted, comp, profile_summary = load_data()

st.title('Khám phá Cụm Khách Hàng')

cluster_opts = ['Tất cả'] + sorted(clusters['cluster'].unique().tolist())
sel = st.sidebar.selectbox('Chọn cụm', cluster_opts)

if sel == 'Tất cả':
    sel_mask = clusters['cluster'].isin(clusters['cluster'].unique())
else:
    sel_mask = clusters['cluster'] == int(sel)

cust_ids = clusters[sel_mask]['CustomerID'].tolist()

st.sidebar.markdown(f"Số khách hàng trong lựa chọn: **{len(cust_ids)}**")

# Hiển thị chỉ số cụm nếu có
if profile_summary is not None:
    st.sidebar.markdown('**Chỉ số cụm**')
    try:
        st.sidebar.write(profile_summary.set_index('cluster'))
    except Exception:
        st.sidebar.write(profile_summary)

# Biểu đồ phân bố cụm
st.sidebar.markdown('**Phân bố cụm**')
try:
    counts = clusters['cluster'].value_counts().reset_index()
    counts.columns = ['cluster','n_customers']
    fig_counts = px.bar(counts.sort_values('cluster'), x='cluster', y='n_customers', title='Số khách theo cụm', text='n_customers')
    fig_counts.update_layout(xaxis_title='Cụm', yaxis_title='Số khách', height=300)
    st.sidebar.plotly_chart(fig_counts, use_container_width=True)
except Exception:
    st.sidebar.write('Không có dữ liệu phân bố cụm')

# Hàm tính PCA tương tác từ weighted features
@st.cache_data
def compute_pca_df(weighted_df, clusters_df):
    if weighted_df is None:
        return None
    if 'CustomerID' not in weighted_df.columns:
        return None
    merged = pd.merge(clusters_df[['CustomerID','cluster']], weighted_df, on='CustomerID', how='inner')
    rule_cols = [c for c in merged.columns if str(c).startswith('rule_')]
    if len(rule_cols) == 0:
        return None
    X = merged[rule_cols].fillna(0).values
    try:
        Xs = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=2).fit_transform(Xs)
    except Exception:
        return None
    out = pd.DataFrame({'CustomerID': merged['CustomerID'], 'pca1': pcs[:,0], 'pca2': pcs[:,1], 'cluster': merged['cluster']})
    return out

if st.sidebar.checkbox('Hiển thị PCA tương tác (tính toán)', value=False):
    pca_df = compute_pca_df(weighted, clusters)
    if pca_df is not None:
        fig = px.scatter(pca_df, x='pca1', y='pca2', color=pca_df['cluster'].astype(str), hover_data=['CustomerID'], title='PCA 2D (tương tác)')
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        pca_path = os.path.join(PROC, 'cluster_pca.png')
        if os.path.exists(pca_path):
            st.image(pca_path, caption='PCA 2D của khách hàng (ảnh tĩnh)')
        else:
            st.write('Không có dữ liệu hoặc ảnh PCA')

# Nội dung chính
st.header('Tóm tắt cụm')
if sel == 'Tất cả':
    st.write('Hiển thị tất cả cụm')
else:
    prof = profiles[profiles['cluster'] == int(sel)]
    if not prof.empty:
        st.write(prof.iloc[0].to_dict())
    else:
        st.write('Không có hồ sơ cho cụm này')

st.header('Luật hàng đầu (theo phổ biến trong cụm)')
if len(cust_ids) == 0:
    st.write('Không có khách trong cụm')
else:
    bin_sub = binary[binary['CustomerID'].isin(cust_ids)].set_index('CustomerID')
    rule_cols = [c for c in bin_sub.columns if str(c).startswith('rule_')]
    if not rule_cols:
        st.write('Không có đặc trưng theo luật')
    else:
        prevalences = bin_sub[rule_cols].sum().sort_values(ascending=False)
        topk = st.sidebar.slider('Top K luật', 1, min(20, len(prevalences)), 5)
        top_rules = prevalences.head(topk)

        # map to rule strings from rules df
        def col_to_rulestr(col):
            try:
                idx = int(col.split('__')[0].split('rule_')[1])
                if idx in rules.index:
                    return rules.loc[idx].get('rule_str', col)
            except Exception:
                pass
            return col

        df_show = pd.DataFrame({'rule_col': top_rules.index, 'count': top_rules.values})
        df_show['rule_str'] = df_show['rule_col'].apply(col_to_rulestr)
        st.dataframe(df_show[['rule_str', 'count']])
        csv_rules = df_show[['rule_str', 'count']].to_csv(index=False).encode('utf-8')
        st.download_button('Tải CSV luật hàng đầu', data=csv_rules, file_name='top_rules.csv', mime='text/csv')

        st.header('Gợi ý bundle / cross-sell')
        consequents = []
        for col in df_show['rule_col']:
            try:
                idx = int(col.split('__')[0].split('rule_')[1])
                if idx in rules.index:
                    cons = rules.loc[idx].get('consequents_str')
                    consequents.append(cons)
            except Exception:
                continue
        if consequents:
            st.write('Các consequents hàng đầu (mặt hàng đề xuất khuyến mãi):')
            st.write(pd.Series(consequents).value_counts().head(10))

st.header('Danh sách khách hàng')
if len(cust_ids) == 0:
    st.write('Không có khách để hiển thị')
else:
    show = st.sidebar.checkbox('Hiển thị mã khách hàng', value=True)
    if show:
        cust_df = clusters[sel_mask][['CustomerID','cluster']]
        st.write(cust_df.head(200))
        csv_cust = cust_df.to_csv(index=False).encode('utf-8')
        st.download_button('Tải CSV danh sách khách', data=csv_cust, file_name='customers_selection.csv', mime='text/csv')

st.markdown('---')
st.write('Các file sử dụng:')
st.write(os.listdir(PROC))

st.sidebar.markdown('---')
if comp is not None:
    if st.sidebar.checkbox('Hiển thị so sánh cấu hình', value=False):
        st.header('So sánh cấu hình')
        st.write('So sánh các cấu hình phân cụm (silhouette, chỉ số actionable)')
        st.dataframe(comp)
        csv_comp = comp.to_csv(index=False).encode('utf-8')
        st.download_button('Tải CSV so sánh', data=csv_comp, file_name='config_comparison.csv', mime='text/csv')
