"""
Module tiền xử lý dữ liệu cho dự án AIR GUARD
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime


class DataPreprocessor:
    """
    Lớp xử lý tiền xử lý dữ liệu chất lượng không khí
    """
    
    def __init__(self, cutoff_date: str = '2017-01-01'):
        """
        Args:
            cutoff_date: Ngày cắt để chia train/test
        """
        self.cutoff_date = pd.to_datetime(cutoff_date)
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file CSV
        
        Args:
            filepath: Đường dẫn đến file dữ liệu
            
        Returns:
            DataFrame chứa dữ liệu
        """
        df = pd.read_csv(filepath)
        print(f"Loaded data shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu: xử lý missing values, outliers
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame đã được làm sạch
        """
        df = df.copy()
        
        # Chuyển đổi cột thời gian
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'year' in df.columns and 'month' in df.columns:
            df['date'] = pd.to_datetime(
                df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1),
                format='%Y-%m-%d-%H'
            )
        
        # Sắp xếp theo thời gian
        df = df.sort_values('date').reset_index(drop=True)
        
        # Xử lý missing values cho PM2.5
        if 'PM2.5' in df.columns:
            # Forward fill và backward fill
            df['PM2.5'] = df['PM2.5'].fillna(method='ffill').fillna(method='bfill')
            
            # Nếu vẫn còn NaN, dùng median
            if df['PM2.5'].isna().any():
                df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].median())
        
        # Xử lý missing values cho các cột khác
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'PM2.5':
                df[col] = df[col].fillna(df[col].median())
        
        # Xử lý outliers cho PM2.5 (giữ lại vì có thể là giá trị thực)
        # Chỉ loại bỏ giá trị âm nếu có
        if 'PM2.5' in df.columns:
            df = df[df['PM2.5'] >= 0]
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def create_aqi_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo nhãn AQI dựa trên nồng độ PM2.5
        
        Theo tiêu chuẩn AQI:
        - Good: 0-12
        - Moderate: 12.1-35.4
        - Unhealthy for Sensitive: 35.5-55.4
        - Unhealthy: 55.5-150.4
        - Very Unhealthy: 150.5-250.4
        - Hazardous: 250.5+
        
        Args:
            df: DataFrame có cột PM2.5
            
        Returns:
            DataFrame với cột AQI_category
        """
        df = df.copy()
        
        def categorize_pm25(pm25):
            if pd.isna(pm25):
                return np.nan
            elif pm25 <= 12.0:
                return 'Good'
            elif pm25 <= 35.4:
                return 'Moderate'
            elif pm25 <= 55.4:
                return 'Unhealthy_Sensitive'
            elif pm25 <= 150.4:
                return 'Unhealthy'
            elif pm25 <= 250.4:
                return 'Very_Unhealthy'
            else:
                return 'Hazardous'
        
        df['AQI_category'] = df['PM2.5'].apply(categorize_pm25)
        
        # Encode labels thành số
        label_map = {
            'Good': 0,
            'Moderate': 1,
            'Unhealthy_Sensitive': 2,
            'Unhealthy': 3,
            'Very_Unhealthy': 4,
            'Hazardous': 5
        }
        df['AQI_label'] = df['AQI_category'].map(label_map)
        
        print("\nAQI Distribution:")
        print(df['AQI_category'].value_counts())
        
        return df
    
    def train_test_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia dữ liệu thành train và test theo thời gian
        
        Args:
            df: DataFrame đầy đủ
            
        Returns:
            train_df, test_df
        """
        train_df = df[df['date'] < self.cutoff_date].copy()
        test_df = df[df['date'] >= self.cutoff_date].copy()
        
        print(f"\nTrain set: {train_df.shape[0]} samples")
        print(f"Test set: {test_df.shape[0]} samples")
        print(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        return train_df, test_df
    
    def create_labeled_unlabeled_split(
        self,
        train_df: pd.DataFrame,
        labeled_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia tập train thành labeled và unlabeled
        
        Args:
            train_df: Tập huấn luyện
            labeled_ratio: Tỷ lệ dữ liệu giữ nhãn
            random_state: Random seed
            
        Returns:
            labeled_df, unlabeled_df
        """
        np.random.seed(random_state)
        
        # Đảm bảo mỗi lớp đều có ít nhất một vài mẫu
        labeled_indices = []
        
        for label in train_df['AQI_label'].unique():
            if pd.isna(label):
                continue
            
            label_indices = train_df[train_df['AQI_label'] == label].index.tolist()
            n_labeled = max(5, int(len(label_indices) * labeled_ratio))
            
            sampled = np.random.choice(
                label_indices, 
                size=min(n_labeled, len(label_indices)),
                replace=False
            )
            labeled_indices.extend(sampled)
        
        # Tạo labeled và unlabeled sets
        labeled_df = train_df.loc[labeled_indices].copy()
        unlabeled_df = train_df.drop(labeled_indices).copy()
        
        # Xóa nhãn trong unlabeled set
        unlabeled_df['AQI_label_true'] = unlabeled_df['AQI_label'].copy()
        unlabeled_df['AQI_category_true'] = unlabeled_df['AQI_category'].copy()
        unlabeled_df['AQI_label'] = -1  # Đánh dấu là unlabeled
        
        print(f"\nLabeled set: {labeled_df.shape[0]} samples ({labeled_ratio*100:.1f}%)")
        print(f"Unlabeled set: {unlabeled_df.shape[0]} samples")
        print("\nLabeled distribution:")
        print(labeled_df['AQI_category'].value_counts())
        
        return labeled_df, unlabeled_df
    
    def preprocess_pipeline(
        self,
        filepath: str,
        labeled_ratio: float = 0.1,
        random_state: int = 42
    ) -> dict:
        """
        Pipeline đầy đủ cho tiền xử lý
        
        Args:
            filepath: Đường dẫn file dữ liệu
            labeled_ratio: Tỷ lệ dữ liệu có nhãn
            random_state: Random seed
            
        Returns:
            Dictionary chứa các DataFrame đã xử lý
        """
        print("=" * 60)
        print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
        print("=" * 60)
        
        # Load và clean
        df = self.load_data(filepath)
        df = self.clean_data(df)
        
        # Tạo nhãn AQI
        df = self.create_aqi_labels(df)
        
        # Chia train/test
        train_df, test_df = self.train_test_split(df)
        
        # Chia labeled/unlabeled
        labeled_df, unlabeled_df = self.create_labeled_unlabeled_split(
            train_df, labeled_ratio, random_state
        )
        
        print("\n" + "=" * 60)
        print("HOÀN THÀNH TIỀN XỬ LÝ")
        print("=" * 60)
        
        return {
            'labeled': labeled_df,
            'unlabeled': unlabeled_df,
            'test': test_df,
            'full_train': train_df
        }


if __name__ == "__main__":
    # Ví dụ sử dụng
    preprocessor = DataPreprocessor(cutoff_date='2017-01-01')
    
    # Giả sử có file data
    # data = preprocessor.preprocess_pipeline(
    #     'data/beijing_pm25.csv',
    #     labeled_ratio=0.1
    # )
