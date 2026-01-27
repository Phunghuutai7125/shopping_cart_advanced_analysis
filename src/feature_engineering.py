"""
Module tạo đặc trưng cho dự án AIR GUARD
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class FeatureEngineer:
    """
    Lớp tạo đặc trưng cho bài toán dự báo AQI
    """
    
    def __init__(self):
        self.temporal_features = []
        self.weather_features = []
        self.lag_features = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các đặc trưng thời gian
        
        Args:
            df: DataFrame có cột 'date'
            
        Returns:
            DataFrame với các đặc trưng thời gian
        """
        df = df.copy()
        
        # Trích xuất các thành phần thời gian
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        
        # Đặc trưng chu kỳ (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Đặc trưng thời gian đặc biệt
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Mùa (4 mùa)
        df['season'] = df['month'] % 12 // 3 + 1
        
        self.temporal_features = [
            'year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'quarter',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dayofweek_sin', 'dayofweek_cos',
            'is_weekend', 'is_rush_hour', 'season'
        ]
        
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'PM2.5',
        lags: List[int] = [1, 2, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Tạo các đặc trưng lag (giá trị quá khứ)
        
        Args:
            df: DataFrame
            target_col: Cột để tạo lag
            lags: Danh sách số bước lag
            
        Returns:
            DataFrame với lag features
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        self.lag_features = []
        
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            self.lag_features.append(col_name)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            # Mean
            col_name = f'{target_col}_rolling_mean_{window}'
            df[col_name] = df[target_col].shift(1).rolling(window=window).mean()
            self.lag_features.append(col_name)
            
            # Std
            col_name = f'{target_col}_rolling_std_{window}'
            df[col_name] = df[target_col].shift(1).rolling(window=window).std()
            self.lag_features.append(col_name)
            
            # Max
            col_name = f'{target_col}_rolling_max_{window}'
            df[col_name] = df[target_col].shift(1).rolling(window=window).max()
            self.lag_features.append(col_name)
            
            # Min
            col_name = f'{target_col}_rolling_min_{window}'
            df[col_name] = df[target_col].shift(1).rolling(window=window).min()
            self.lag_features.append(col_name)
        
        # Difference features
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24'] = df[target_col].diff(24)
        self.lag_features.extend([f'{target_col}_diff_1', f'{target_col}_diff_24'])
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo và xử lý các đặc trưng thời tiết
        
        Args:
            df: DataFrame có các cột thời tiết
            
        Returns:
            DataFrame với đặc trưng thời tiết
        """
        df = df.copy()
        
        # Xác định các cột thời tiết có sẵn
        possible_weather_cols = [
            'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
            'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction'
        ]
        
        self.weather_features = [col for col in possible_weather_cols if col in df.columns]
        
        # Tạo thêm các đặc trưng tương tác
        if 'TEMP' in df.columns and 'DEWP' in df.columns:
            # Humidity approximation
            df['humidity_approx'] = 100 * np.exp(
                (17.625 * df['DEWP']) / (243.04 + df['DEWP'])
            ) / np.exp(
                (17.625 * df['TEMP']) / (243.04 + df['TEMP'])
            )
            self.weather_features.append('humidity_approx')
        
        if 'WSPM' in df.columns:
            # Wind categories
            df['wind_calm'] = (df['WSPM'] < 1).astype(int)
            df['wind_strong'] = (df['WSPM'] > 10).astype(int)
            self.weather_features.extend(['wind_calm', 'wind_strong'])
        
        if 'RAIN' in df.columns:
            # Rain indicator
            df['is_raining'] = (df['RAIN'] > 0).astype(int)
            self.weather_features.append('is_raining')
        
        # Xử lý missing values trong weather features
        for col in self.weather_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các đặc trưng tương tác
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame với interaction features
        """
        df = df.copy()
        
        # Tương tác giữa thời gian và thời tiết
        if 'TEMP' in df.columns:
            df['temp_hour'] = df['TEMP'] * df['hour']
            df['temp_month'] = df['TEMP'] * df['month']
        
        if 'WSPM' in df.columns:
            df['wind_hour'] = df['WSPM'] * df['hour']
        
        # Tương tác PM2.5 lag với thời tiết
        if 'PM2.5_lag_1' in df.columns and 'WSPM' in df.columns:
            df['pm_wind_interaction'] = df['PM2.5_lag_1'] * df['WSPM']
        
        return df
    
    def get_feature_views(self) -> Tuple[List[str], List[str]]:
        """
        Trả về hai view đặc trưng cho co-training
        
        Returns:
            view1 (temporal + lag), view2 (weather + environmental)
        """
        view1 = self.temporal_features + self.lag_features
        view2 = self.weather_features
        
        return view1, view2
    
    def feature_engineering_pipeline(
        self,
        df: pd.DataFrame,
        create_lags: bool = True
    ) -> pd.DataFrame:
        """
        Pipeline đầy đủ cho feature engineering
        
        Args:
            df: DataFrame đầu vào
            create_lags: Có tạo lag features không
            
        Returns:
            DataFrame với tất cả features
        """
        print("=" * 60)
        print("BẮT ĐẦU FEATURE ENGINEERING")
        print("=" * 60)
        
        # Tạo temporal features
        print("\n1. Creating temporal features...")
        df = self.create_temporal_features(df)
        print(f"   Added {len(self.temporal_features)} temporal features")
        
        # Tạo weather features
        print("\n2. Creating weather features...")
        df = self.create_weather_features(df)
        print(f"   Added {len(self.weather_features)} weather features")
        
        # Tạo lag features
        if create_lags:
            print("\n3. Creating lag features...")
            df = self.create_lag_features(df)
            print(f"   Added {len(self.lag_features)} lag features")
        
        # Tạo interaction features
        print("\n4. Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Loại bỏ các hàng có NaN do lag
        initial_rows = len(df)
        df = df.dropna(subset=self.lag_features if create_lags else [])
        print(f"\n5. Dropped {initial_rows - len(df)} rows with NaN in lag features")
        
        print("\n" + "=" * 60)
        print("HOÀN THÀNH FEATURE ENGINEERING")
        print(f"Total features: {len(df.columns)}")
        print("=" * 60)
        
        return df
    
    def prepare_features_labels(
        self,
        df: pd.DataFrame,
        target_col: str = 'AQI_label'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Chuẩn bị X và y cho training
        
        Args:
            df: DataFrame với tất cả features và labels
            target_col: Tên cột target
            
        Returns:
            X (features), y (labels)
        """
        # Các cột không dùng làm feature
        exclude_cols = [
            'date', 'AQI_label', 'AQI_category', 'PM2.5',
            'AQI_label_true', 'AQI_category_true',
            'year', 'No'  # year có thể gây overfitting
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        return X, y


if __name__ == "__main__":
    # Ví dụ sử dụng
    engineer = FeatureEngineer()
    
    # Giả sử có DataFrame
    # df_with_features = engineer.feature_engineering_pipeline(df)
    # X, y = engineer.prepare_features_labels(df_with_features)
