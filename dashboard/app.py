"""
Streamlit Dashboard cho AIR GUARD
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cấu hình trang
st.set_page_config(
    page_title="AIR GUARD Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #028090;
    text-align: center;
    padding: 1rem 0;
}
.sub-header {
    font-size: 1.5rem;
    color: #00A896;
    font-weight: 600;
    margin-top: 1rem;
}
.metric-card {
    background-color: #F0F9FF;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #028090;
}
</style>
""", unsafe_allow_html=True)


def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def load_history(filepath):
    """Load training history from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    # Header
    st.markdown('<p class="main-header">🌍 AIR GUARD</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#64748B; font-size:1.2rem;">Dự báo Chất lượng Không khí với Học Bán giám sát</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Chọn trang",
        ["Tổng quan", "Baseline Model", "Self-Training", "Co-Training", "So sánh"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **AIR GUARD** sử dụng các thuật toán học bán giám sát để dự báo chất lượng không khí:
    - **Self-Training**: Tự huấn luyện với nhãn giả
    - **Co-Training**: Đồng huấn luyện 2 models
    """)
    
    # Main content
    results_dir = Path('../results')
    
    if page == "Tổng quan":
        show_overview(results_dir)
    elif page == "Baseline Model":
        show_baseline(results_dir)
    elif page == "Self-Training":
        show_self_training(results_dir)
    elif page == "Co-Training":
        show_co_training(results_dir)
    elif page == "So sánh":
        show_comparison(results_dir)


def show_overview(results_dir):
    """Trang tổng quan"""
    st.markdown('<p class="sub-header">📋 Tổng quan Dự án</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Mục tiêu
        Dự án AIR GUARD nhằm dự báo chất lượng không khí (AQI) dựa trên nồng độ PM2.5 
        với dữ liệu có nhãn khan hiếm.
        
        ### 📊 Phương pháp
        - **Baseline**: HistGradientBoosting truyền thống
        - **Self-Training**: Tự gán nhãn với độ tin cậy cao
        - **Co-Training**: 2 models với 2 views đặc trưng
        
        ### 📈 Dữ liệu
        - Nguồn: Beijing PM2.5 Dataset
        - Phân chia: Train (< 2017), Test (≥ 2017)
        - Labeled ratio: 10%
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 Tiêu chí AQI
        
        | Mức | PM2.5 (μg/m³) | Mô tả |
        |-----|---------------|-------|
        | Good | 0 - 12 | Tốt |
        | Moderate | 12.1 - 35.4 | Trung bình |
        | Unhealthy (Sensitive) | 35.5 - 55.4 | Không lành mạnh cho nhóm nhạy cảm |
        | Unhealthy | 55.5 - 150.4 | Không lành mạnh |
        | Very Unhealthy | 150.5 - 250.4 | Rất không lành mạnh |
        | Hazardous | > 250.4 | Nguy hại |
        """)
    
    st.markdown("---")
    
    # Load và hiển thị metrics nếu có
    metrics_baseline = load_metrics(results_dir / 'metrics_baseline.json')
    metrics_self = load_metrics(results_dir / 'metrics_self_training.json')
    metrics_co = load_metrics(results_dir / 'metrics_co_training.json')
    
    if metrics_baseline and metrics_self and metrics_co:
        st.markdown("### 🎖️ Kết quả Tổng quan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Baseline Accuracy",
                f"{metrics_baseline['accuracy']:.4f}",
                f"F1: {metrics_baseline['f1_macro']:.4f}"
            )
        
        with col2:
            delta_acc = metrics_self['accuracy'] - metrics_baseline['accuracy']
            st.metric(
                "Self-Training Accuracy",
                f"{metrics_self['accuracy']:.4f}",
                f"{delta_acc:+.4f} vs Baseline"
            )
        
        with col3:
            delta_acc = metrics_co['accuracy'] - metrics_baseline['accuracy']
            st.metric(
                "Co-Training Accuracy",
                f"{metrics_co['accuracy']:.4f}",
                f"{delta_acc:+.4f} vs Baseline"
            )
    else:
        st.warning("⚠️ Chưa có kết quả. Vui lòng chạy `python src/main.py` trước.")


def show_baseline(results_dir):
    """Hiển thị kết quả Baseline"""
    st.markdown('<p class="sub-header">📊 Baseline Model</p>', unsafe_allow_html=True)
    
    metrics = load_metrics(results_dir / 'metrics_baseline.json')
    
    if metrics:
        st.markdown("### Overall Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
        col3.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
        col4.metric("F1-Score (Macro)", f"{metrics['f1_macro']:.4f}")
        
        st.markdown("---")
        st.markdown("### Per-Class Performance")
        
        # Tạo DataFrame cho per-class metrics
        per_class_data = []
        for class_name, scores in metrics['per_class'].items():
            per_class_data.append({
                'Class': class_name,
                'Precision': scores['precision'],
                'Recall': scores['recall'],
                'F1-Score': scores['f1']
            })
        
        df_per_class = pd.DataFrame(per_class_data)
        st.dataframe(df_per_class, use_container_width=True)
        
        # Hiển thị confusion matrix nếu có
        cm_path = results_dir / 'cm_baseline.png'
        if cm_path.exists():
            st.markdown("### Confusion Matrix")
            st.image(str(cm_path), use_column_width=True)
    else:
        st.warning("⚠️ Không tìm thấy kết quả Baseline.")


def show_self_training(results_dir):
    """Hiển thị kết quả Self-Training"""
    st.markdown('<p class="sub-header">🔄 Self-Training Model</p>', unsafe_allow_html=True)
    
    metrics = load_metrics(results_dir / 'metrics_self_training.json')
    history = load_history(results_dir / 'history_self_training.json')
    
    if metrics:
        st.markdown("### Overall Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
        col3.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
        col4.metric("F1-Score (Macro)", f"{metrics['f1_macro']:.4f}")
        
        if history:
            st.markdown("---")
            st.markdown("### Training History")
            
            # Chuyển history thành DataFrame
            df_history = pd.DataFrame(history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(df_history.set_index('iteration')['train_size'])
                st.caption("Training Set Size")
            
            with col2:
                if 'val_accuracy' in df_history.columns:
                    st.line_chart(df_history.set_index('iteration')[['val_accuracy', 'val_f1_macro']])
                    st.caption("Validation Performance")
        
        # Hiển thị plot nếu có
        plot_path = results_dir / 'self_training_history.png'
        if plot_path.exists():
            st.markdown("### Detailed History")
            st.image(str(plot_path), use_column_width=True)
        
        # Confusion matrix
        cm_path = results_dir / 'cm_self_training.png'
        if cm_path.exists():
            st.markdown("### Confusion Matrix")
            st.image(str(cm_path), use_column_width=True)
    else:
        st.warning("⚠️ Không tìm thấy kết quả Self-Training.")


def show_co_training(results_dir):
    """Hiển thị kết quả Co-Training"""
    st.markdown('<p class="sub-header">🤝 Co-Training Model</p>', unsafe_allow_html=True)
    
    metrics = load_metrics(results_dir / 'metrics_co_training.json')
    history = load_history(results_dir / 'history_co_training.json')
    
    if metrics:
        st.markdown("### Overall Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
        col3.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
        col4.metric("F1-Score (Macro)", f"{metrics['f1_macro']:.4f}")
        
        if history:
            st.markdown("---")
            st.markdown("### Training History")
            
            df_history = pd.DataFrame(history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(df_history.set_index('iteration')[['model_a_train_size', 'model_b_train_size']])
                st.caption("Training Set Sizes (Both Models)")
            
            with col2:
                if 'ensemble_val_accuracy' in df_history.columns:
                    st.line_chart(df_history.set_index('iteration')['ensemble_val_accuracy'])
                    st.caption("Ensemble Validation Accuracy")
        
        # Confusion matrix
        cm_path = results_dir / 'cm_co_training.png'
        if cm_path.exists():
            st.markdown("### Confusion Matrix")
            st.image(str(cm_path), use_column_width=True)
    else:
        st.warning("⚠️ Không tìm thấy kết quả Co-Training.")


def show_comparison(results_dir):
    """So sánh các models"""
    st.markdown('<p class="sub-header">⚖️ So sánh các Models</p>', unsafe_allow_html=True)
    
    metrics_baseline = load_metrics(results_dir / 'metrics_baseline.json')
    metrics_self = load_metrics(results_dir / 'metrics_self_training.json')
    metrics_co = load_metrics(results_dir / 'metrics_co_training.json')
    
    if all([metrics_baseline, metrics_self, metrics_co]):
        # Tạo bảng so sánh
        comparison_data = {
            'Model': ['Baseline', 'Self-Training', 'Co-Training'],
            'Accuracy': [
                metrics_baseline['accuracy'],
                metrics_self['accuracy'],
                metrics_co['accuracy']
            ],
            'F1-Macro': [
                metrics_baseline['f1_macro'],
                metrics_self['f1_macro'],
                metrics_co['f1_macro']
            ],
            'Precision-Macro': [
                metrics_baseline['precision_macro'],
                metrics_self['precision_macro'],
                metrics_co['precision_macro']
            ],
            'Recall-Macro': [
                metrics_baseline['recall_macro'],
                metrics_self['recall_macro'],
                metrics_co['recall_macro']
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        st.markdown("### 📊 Bảng So sánh")
        st.dataframe(df_comparison.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Macro']), use_container_width=True)
        
        # Biểu đồ so sánh
        comp_path = results_dir / 'model_comparison.png'
        if comp_path.exists():
            st.markdown("### 📈 Biểu đồ So sánh")
            st.image(str(comp_path), use_column_width=True)
        
        # Per-class comparison
        perclass_path = results_dir / 'per_class_comparison.png'
        if perclass_path.exists():
            st.markdown("### 📊 So sánh Từng Lớp")
            st.image(str(perclass_path), use_column_width=True)
        
        # Insights
        st.markdown("---")
        st.markdown("### 💡 Nhận xét")
        
        best_model = df_comparison.loc[df_comparison['F1-Macro'].idxmax(), 'Model']
        best_f1 = df_comparison['F1-Macro'].max()
        baseline_f1 = df_comparison.loc[df_comparison['Model'] == 'Baseline', 'F1-Macro'].values[0]
        improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **Mô hình tốt nhất**: {best_model}
            - F1-Score (Macro): {best_f1:.4f}
            - Cải thiện: +{improvement:.2f}% so với Baseline
            """)
        
        with col2:
            if metrics_self['f1_macro'] > metrics_baseline['f1_macro']:
                st.info("✅ Self-Training đã cải thiện hiệu năng so với Baseline")
            else:
                st.warning("⚠️ Self-Training chưa cải thiện hiệu năng")
            
            if metrics_co['f1_macro'] > metrics_self['f1_macro']:
                st.info("✅ Co-Training tốt hơn Self-Training")
            else:
                st.info("ℹ️ Self-Training tốt hơn hoặc tương đương Co-Training")
    else:
        st.warning("⚠️ Chưa đủ kết quả để so sánh. Vui lòng chạy tất cả các models.")


if __name__ == "__main__":
    main()
