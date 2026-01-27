"""
Module Co-Training cho dự án AIR GUARD
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from typing import Dict, List, Tuple
import json


class CoTraining:
    """
    Triển khai thuật toán Co-Training với 2 views
    """
    
    def __init__(
        self,
        model_a=None,
        model_b=None,
        confidence_threshold: float = 0.9,
        max_iter: int = 10,
        max_new_per_model: int = 100,
        min_new_per_iter: int = 20,
        verbose: bool = True
    ):
        """
        Args:
            model_a: Mô hình cho view 1
            model_b: Mô hình cho view 2
            confidence_threshold: Ngưỡng tin cậy
            max_iter: Số vòng lặp tối đa
            max_new_per_model: Số mẫu tối đa mỗi mô hình thêm mỗi vòng
            min_new_per_iter: Số mẫu tối thiểu tổng cộng mỗi vòng
            verbose: In thông tin
        """
        if model_a is None:
            self.base_model_a = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            self.base_model_a = model_a
            
        if model_b is None:
            self.base_model_b = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=43  # Seed khác một chút
            )
        else:
            self.base_model_b = model_b
            
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
        self.max_new_per_model = max_new_per_model
        self.min_new_per_iter = min_new_per_iter
        self.verbose = verbose
        
        self.model_a = None
        self.model_b = None
        self.history = []
        self.view1_features = None
        self.view2_features = None
        
    def fit(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: pd.Series,
        X_unlabeled: pd.DataFrame,
        view1_features: List[str],
        view2_features: List[str],
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> 'CoTraining':
        """
        Huấn luyện mô hình co-training
        
        Args:
            X_labeled: Features tập có nhãn
            y_labeled: Labels tập có nhãn
            X_unlabeled: Features tập không nhãn
            view1_features: Danh sách features cho view 1
            view2_features: Danh sách features cho view 2
            X_val: Features validation
            y_val: Labels validation
            
        Returns:
            self
        """
        self.view1_features = view1_features
        self.view2_features = view2_features
        
        if self.verbose:
            print("=" * 70)
            print("BẮT ĐẦU CO-TRAINING")
            print("=" * 70)
            print(f"Initial labeled samples: {len(X_labeled)}")
            print(f"Initial unlabeled samples: {len(X_unlabeled)}")
            print(f"View 1 features: {len(view1_features)}")
            print(f"View 2 features: {len(view2_features)}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print("=" * 70)
        
        # Khởi tạo training sets cho cả 2 mô hình
        X_train_a = X_labeled[view1_features].copy()
        X_train_b = X_labeled[view2_features].copy()
        y_train_a = y_labeled.copy()
        y_train_b = y_labeled.copy()
        
        unlabeled_pool = X_unlabeled.copy()
        unlabeled_indices = set(range(len(unlabeled_pool)))
        
        # Huấn luyện mô hình ban đầu
        self.model_a = clone(self.base_model_a)
        self.model_b = clone(self.base_model_b)
        
        self.model_a.fit(X_train_a, y_train_a)
        self.model_b.fit(X_train_b, y_train_b)
        
        # Đánh giá ban đầu
        initial_metrics = self._evaluate_iteration(
            iteration=0,
            X_train_a=X_train_a,
            y_train_a=y_train_a,
            X_train_b=X_train_b,
            y_train_b=y_train_b,
            unlabeled_size=len(unlabeled_pool),
            new_from_a=0,
            new_from_b=0,
            X_val=X_val,
            y_val=y_val
        )
        self.history.append(initial_metrics)
        
        # Vòng lặp co-training
        for iteration in range(1, self.max_iter + 1):
            if len(unlabeled_indices) == 0:
                if self.verbose:
                    print(f"\n[Iteration {iteration}] No more unlabeled data. Stopping.")
                break
            
            # Lấy unlabeled pool hiện tại
            current_unlabeled_idx = sorted(list(unlabeled_indices))
            current_unlabeled = unlabeled_pool.iloc[current_unlabeled_idx]
            
            # Model A dự đoán và chọn mẫu tự tin
            indices_from_a, labels_from_a = self._select_confident_samples(
                self.model_a,
                current_unlabeled[view1_features],
                current_unlabeled_idx
            )
            
            # Model B dự đoán và chọn mẫu tự tin
            indices_from_b, labels_from_b = self._select_confident_samples(
                self.model_b,
                current_unlabeled[view2_features],
                current_unlabeled_idx
            )
            
            # Kiểm tra điều kiện dừng
            total_new = len(indices_from_a) + len(indices_from_b)
            if total_new < self.min_new_per_iter:
                if self.verbose:
                    print(f"\n[Iteration {iteration}] Only {total_new} new samples (< {self.min_new_per_iter}). Stopping.")
                break
            
            # Thêm mẫu từ Model A vào training set của Model B
            if len(indices_from_a) > 0:
                X_new_b = unlabeled_pool.iloc[indices_from_a][view2_features]
                y_new_b = pd.Series(labels_from_a)
                
                X_train_b = pd.concat([X_train_b, X_new_b], ignore_index=True)
                y_train_b = pd.concat([y_train_b, y_new_b], ignore_index=True)
            
            # Thêm mẫu từ Model B vào training set của Model A
            if len(indices_from_b) > 0:
                X_new_a = unlabeled_pool.iloc[indices_from_b][view1_features]
                y_new_a = pd.Series(labels_from_b)
                
                X_train_a = pd.concat([X_train_a, X_new_a], ignore_index=True)
                y_train_a = pd.concat([y_train_a, y_new_a], ignore_index=True)
            
            # Loại bỏ khỏi unlabeled pool
            used_indices = set(indices_from_a + indices_from_b)
            unlabeled_indices -= used_indices
            
            # Huấn luyện lại cả 2 mô hình
            self.model_a = clone(self.base_model_a)
            self.model_b = clone(self.base_model_b)
            
            self.model_a.fit(X_train_a, y_train_a)
            self.model_b.fit(X_train_b, y_train_b)
            
            # Đánh giá
            iter_metrics = self._evaluate_iteration(
                iteration=iteration,
                X_train_a=X_train_a,
                y_train_a=y_train_a,
                X_train_b=X_train_b,
                y_train_b=y_train_b,
                unlabeled_size=len(unlabeled_indices),
                new_from_a=len(indices_from_a),
                new_from_b=len(indices_from_b),
                X_val=X_val,
                y_val=y_val
            )
            self.history.append(iter_metrics)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("HOÀN THÀNH CO-TRAINING")
            print(f"Final Model A training size: {len(X_train_a)}")
            print(f"Final Model B training size: {len(X_train_b)}")
            print(f"Unlabeled remaining: {len(unlabeled_indices)}")
            print("=" * 70)
        
        return self
    
    def _select_confident_samples(
        self,
        model,
        X_unlabeled: pd.DataFrame,
        original_indices: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Chọn các mẫu có độ tin cậy cao
        """
        proba = model.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        predicted_labels = np.argmax(proba, axis=1)
        
        confident_mask = max_proba >= self.confidence_threshold
        confident_local_idx = np.where(confident_mask)[0]
        
        # Giới hạn số lượng
        if len(confident_local_idx) > self.max_new_per_model:
            # Sắp xếp theo độ tin cậy và lấy top
            sorted_idx = np.argsort(max_proba[confident_mask])[::-1]
            confident_local_idx = confident_local_idx[sorted_idx[:self.max_new_per_model]]
        
        # Chuyển về original indices
        selected_indices = [original_indices[i] for i in confident_local_idx]
        selected_labels = predicted_labels[confident_local_idx].tolist()
        
        return selected_indices, selected_labels
    
    def _evaluate_iteration(
        self,
        iteration: int,
        X_train_a: pd.DataFrame,
        y_train_a: pd.Series,
        X_train_b: pd.DataFrame,
        y_train_b: pd.Series,
        unlabeled_size: int,
        new_from_a: int,
        new_from_b: int,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict:
        """
        Đánh giá sau mỗi vòng lặp
        """
        metrics = {
            'iteration': iteration,
            'model_a_train_size': len(X_train_a),
            'model_b_train_size': len(X_train_b),
            'unlabeled_size': unlabeled_size,
            'new_from_model_a': new_from_a,
            'new_from_model_b': new_from_b
        }
        
        if X_val is not None and y_val is not None:
            from sklearn.metrics import accuracy_score, f1_score
            
            # Đánh giá Model A
            y_val_pred_a = self.model_a.predict(X_val[self.view1_features])
            metrics['model_a_val_accuracy'] = accuracy_score(y_val, y_val_pred_a)
            metrics['model_a_val_f1_macro'] = f1_score(y_val, y_val_pred_a, average='macro')
            
            # Đánh giá Model B
            y_val_pred_b = self.model_b.predict(X_val[self.view2_features])
            metrics['model_b_val_accuracy'] = accuracy_score(y_val, y_val_pred_b)
            metrics['model_b_val_f1_macro'] = f1_score(y_val, y_val_pred_b, average='macro')
            
            # Ensemble (voting)
            from scipy.stats import mode
            y_val_pred_ensemble = mode(np.vstack([y_val_pred_a, y_val_pred_b]), axis=0)[0][0]
            metrics['ensemble_val_accuracy'] = accuracy_score(y_val, y_val_pred_ensemble)
            metrics['ensemble_val_f1_macro'] = f1_score(y_val, y_val_pred_ensemble, average='macro')
        
        if self.verbose:
            print(f"\n[Iteration {iteration}]")
            print(f"  Model A training size: {metrics['model_a_train_size']}")
            print(f"  Model B training size: {metrics['model_b_train_size']}")
            print(f"  New from Model A: {metrics['new_from_model_a']}")
            print(f"  New from Model B: {metrics['new_from_model_b']}")
            print(f"  Unlabeled remaining: {metrics['unlabeled_size']}")
            if 'model_a_val_accuracy' in metrics:
                print(f"  Model A Val Acc: {metrics['model_a_val_accuracy']:.4f}, F1: {metrics['model_a_val_f1_macro']:.4f}")
                print(f"  Model B Val Acc: {metrics['model_b_val_accuracy']:.4f}, F1: {metrics['model_b_val_f1_macro']:.4f}")
                print(f"  Ensemble Val Acc: {metrics['ensemble_val_accuracy']:.4f}, F1: {metrics['ensemble_val_f1_macro']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Dự đoán nhãn
        """
        if self.model_a is None or self.model_b is None:
            raise ValueError("Models not trained yet. Call fit() first.")
        
        if use_ensemble:
            pred_a = self.model_a.predict(X[self.view1_features])
            pred_b = self.model_b.predict(X[self.view2_features])
            from scipy.stats import mode
            return mode(np.vstack([pred_a, pred_b]), axis=0)[0][0]
        else:
            # Dùng model A làm chính
            return self.model_a.predict(X[self.view1_features])
    
    def get_history(self) -> List[Dict]:
        """
        Lấy lịch sử huấn luyện
        """
        return self.history
    
    def save_history(self, filepath: str):
        """
        Lưu lịch sử ra file JSON
        """
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {filepath}")


if __name__ == "__main__":
    print("Co-Training Module")
    print("Use this module with preprocessed data and feature views")
