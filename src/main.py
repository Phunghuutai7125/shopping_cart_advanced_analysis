"""
Script chính để huấn luyện các mô hình AIR GUARD
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Import các module tự tạo
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from self_training import SelfTraining
from co_training import CoTraining
from evaluation import ModelEvaluator


def main():
    """
    Hàm chính để chạy toàn bộ pipeline
    """
    print("\n" + "="*80)
    print(" " * 20 + "AIR GUARD - DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ")
    print("="*80 + "\n")
    
    # ========== CẤU HÌNH ==========
    DATA_PATH = '../data/beijing_pm25.csv'  # Đường dẫn đến dữ liệu
    LABELED_RATIO = 0.1  # Tỷ lệ dữ liệu có nhãn
    CONFIDENCE_THRESHOLD = 0.9  # Ngưỡng tin cậy
    
    print("CẤU HÌNH:")
    print(f"  - Labeled ratio: {LABELED_RATIO*100}%")
    print(f"  - Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  - Data path: {DATA_PATH}\n")
    
    # ========== BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ==========
    print("\n" + "="*80)
    print("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU")
    print("="*80)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: File not found: {DATA_PATH}")
        print("Vui lòng tải dữ liệu Beijing PM2.5 và đặt vào thư mục data/")
        print("\nSử dụng dữ liệu mẫu để demo...")
        
        # Tạo dữ liệu mẫu
        create_sample_data('../data/beijing_pm25_sample.csv')
        DATA_PATH = '../data/beijing_pm25_sample.csv'
    
    preprocessor = DataPreprocessor(cutoff_date='2015-07-01')
    data_dict = preprocessor.preprocess_pipeline(
        DATA_PATH,
        labeled_ratio=LABELED_RATIO,
        random_state=42
    )
    
    labeled_df = data_dict['labeled']
    unlabeled_df = data_dict['unlabeled']
    test_df = data_dict['test']
    
    # ========== BƯỚC 2: FEATURE ENGINEERING ==========
    print("\n" + "="*80)
    print("BƯỚC 2: FEATURE ENGINEERING")
    print("="*80)
    
    engineer = FeatureEngineer()
    
    # Tạo features cho các tập
    labeled_df = engineer.feature_engineering_pipeline(labeled_df, create_lags=True)
    unlabeled_df = engineer.feature_engineering_pipeline(unlabeled_df, create_lags=True)
    test_df = engineer.feature_engineering_pipeline(test_df, create_lags=True)
    
    # Chuẩn bị X, y
    X_labeled, y_labeled = engineer.prepare_features_labels(labeled_df)
    X_unlabeled, _ = engineer.prepare_features_labels(unlabeled_df)
    X_test, y_test = engineer.prepare_features_labels(test_df)
    
    # Tạo validation set từ labeled
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    print(f"\nFinal dataset sizes:")
    print(f"  Training (labeled): {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Unlabeled pool: {len(X_unlabeled)}")
    print(f"  Test: {len(X_test)}")
    
    # ========== BƯỚC 3: BASELINE MODEL ==========
    print("\n" + "="*80)
    print("BƯỚC 3: HUẤN LUYỆN BASELINE MODEL")
    print("="*80)
    
    baseline = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    
    evaluator = ModelEvaluator()
    metrics_baseline = evaluator.evaluate(y_test, y_pred_baseline, "Baseline")
    evaluator.print_evaluation(metrics_baseline)
    evaluator.save_metrics(metrics_baseline, '../results/metrics_baseline.json')
    
    # ========== BƯỚC 4: SELF-TRAINING ==========
    print("\n" + "="*80)
    print("BƯỚC 4: HUẤN LUYỆN SELF-TRAINING MODEL")
    print("="*80)
    
    self_trainer = SelfTraining(
        base_model=None,  # Sử dụng HistGradientBoosting mặc định
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_iter=10,
        min_new_per_iter=20,
        verbose=True
    )
    
    self_trainer.fit(
        X_labeled=X_train,
        y_labeled=y_train,
        X_unlabeled=X_unlabeled,
        X_val=X_val,
        y_val=y_val
    )
    
    y_pred_self = self_trainer.predict(X_test)
    
    metrics_self = evaluator.evaluate(y_test, y_pred_self, "Self-Training")
    evaluator.print_evaluation(metrics_self)
    evaluator.save_metrics(metrics_self, '../results/metrics_self_training.json')
    self_trainer.save_history('../results/history_self_training.json')
    
    # Plot history
    self_trainer.plot_history(save_path='../results/self_training_history.png')
    
    # ========== BƯỚC 5: CO-TRAINING ==========
    print("\n" + "="*80)
    print("BƯỚC 5: HUẤN LUYỆN CO-TRAINING MODEL")
    print("="*80)
    
    # Lấy 2 views features
    view1_features, view2_features = engineer.get_feature_views()
    
    print(f"\nView 1 (Temporal + Lag): {len(view1_features)} features")
    print(f"View 2 (Weather): {len(view2_features)} features")
    
    # Đảm bảo cả 2 views đều có features trong X
    view1_features = [f for f in view1_features if f in X_train.columns]
    view2_features = [f for f in view2_features if f in X_train.columns]
    
    print(f"\nFiltered View 1: {len(view1_features)} features")
    print(f"Filtered View 2: {len(view2_features)} features")
    
    co_trainer = CoTraining(
        model_a=None,
        model_b=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_iter=10,
        max_new_per_model=100,
        min_new_per_iter=20,
        verbose=True
    )
    
    # Kết hợp lại X_labeled từ train và val cho co-training
    X_labeled_full = pd.concat([X_train, X_val], ignore_index=True)
    y_labeled_full = pd.concat([y_train, y_val], ignore_index=True)
    
    co_trainer.fit(
        X_labeled=X_labeled_full,
        y_labeled=y_labeled_full,
        X_unlabeled=X_unlabeled,
        view1_features=view1_features,
        view2_features=view2_features,
        X_val=X_val,
        y_val=y_val
    )
    
    y_pred_co = co_trainer.predict(X_test, use_ensemble=True)
    
    metrics_co = evaluator.evaluate(y_test, y_pred_co, "Co-Training")
    evaluator.print_evaluation(metrics_co)
    evaluator.save_metrics(metrics_co, '../results/metrics_co_training.json')
    co_trainer.save_history('../results/history_co_training.json')
    
    # ========== BƯỚC 6: SO SÁNH KẾT QUẢ ==========
    print("\n" + "="*80)
    print("BƯỚC 6: SO SÁNH CÁC MÔ HÌNH")
    print("="*80)
    
    # So sánh tổng quan
    evaluator.compare_models(
        [metrics_baseline, metrics_self, metrics_co],
        save_path='../results/model_comparison.png'
    )
    
    # So sánh per-class
    evaluator.plot_per_class_comparison(
        [metrics_baseline, metrics_self, metrics_co],
        metric_name='f1',
        save_path='../results/per_class_comparison.png'
    )
    
    # Confusion matrices
    evaluator.plot_confusion_matrix(
        y_test, y_pred_baseline,
        title="Baseline - Confusion Matrix",
        save_path="../results/cm_baseline.png"
    )
    
    evaluator.plot_confusion_matrix(
        y_test, y_pred_self,
        title="Self-Training - Confusion Matrix",
        save_path="../results/cm_self_training.png"
    )
    
    evaluator.plot_confusion_matrix(
        y_test, y_pred_co,
        title="Co-Training - Confusion Matrix",
        save_path="../results/cm_co_training.png"
    )
    
    # ========== HOÀN THÀNH ==========
    print("\n" + "="*80)
    print(" " * 30 + "HOÀN THÀNH!")
    print("="*80)
    print("\nCác kết quả đã được lưu vào thư mục results/")
    print("  - Metrics: metrics_*.json")
    print("  - History: history_*.json")
    print("  - Plots: *.png")
    print("\nChạy dashboard để xem kết quả:")
    print("  streamlit run dashboard/app.py")
    print("="*80 + "\n")


def create_sample_data(output_path: str):
    """
    Tạo dữ liệu mẫu để demo
    """
    print("\nĐang tạo dữ liệu mẫu...")
    
    # Tạo dữ liệu giả
    np.random.seed(42)
    n_samples = 5000
    
    dates = pd.date_range('2015-01-01', periods=n_samples, freq='H')
    
    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'PM2.5': np.random.lognormal(3, 1, n_samples),  # Log-normal distribution
        'TEMP': np.random.normal(15, 10, n_samples),
        'PRES': np.random.normal(1015, 10, n_samples),
        'DEWP': np.random.normal(5, 8, n_samples),
        'RAIN': np.random.exponential(0.5, n_samples),
        'WSPM': np.random.exponential(3, n_samples)
    })
    
    # Đảm bảo PM2.5 không âm
    df['PM2.5'] = df['PM2.5'].clip(lower=0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dữ liệu mẫu đã được tạo: {output_path}")


if __name__ == "__main__":
    main()
