"""
Module Self-Training cho dự án AIR GUARD
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from typing import Dict, List, Tuple
import json


class SelfTraining:
    """
    Triển khai thuật toán Self-Training
    """
    
    def __init__(
        self,
        base_model=None,
        confidence_threshold: float = 0.9,
        max_iter: int = 10,
        min_new_per_iter: int = 20,
        verbose: bool = True
    ):
        """
        Args:
            base_model: Mô hình cơ sở (nếu None, dùng HistGradientBoostingClassifier)
            confidence_threshold: Ngưỡng tin cậy để gán nhãn
            max_iter: Số vòng lặp tối đa
            min_new_per_iter: Số mẫu tối thiểu mỗi vòng
            verbose: In thông tin quá trình
        """
        if base_model is None:
            self.base_model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            self.base_model = base_model
            
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
        self.min_new_per_iter = min_new_per_iter
        self.verbose = verbose
        
        self.model = None
        self.history = []
        
    def fit(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: pd.Series,
        X_unlabeled: pd.DataFrame,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> 'SelfTraining':
        """
        Huấn luyện mô hình self-training
        
        Args:
            X_labeled: Features của tập có nhãn
            y_labeled: Labels của tập có nhãn
            X_unlabeled: Features của tập không nhãn
            X_val: Features validation (optional)
            y_val: Labels validation (optional)
            
        Returns:
            self
        """
        if self.verbose:
            print("=" * 70)
            print("BẮT ĐẦU SELF-TRAINING")
            print("=" * 70)
            print(f"Initial labeled samples: {len(X_labeled)}")
            print(f"Initial unlabeled samples: {len(X_unlabeled)}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print("=" * 70)
        
        # Khởi tạo
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        unlabeled_pool = X_unlabeled.copy()
        
        # Huấn luyện mô hình ban đầu
        self.model = clone(self.base_model)
        self.model.fit(X_train, y_train)
        
        # Đánh giá ban đầu
        initial_metrics = self._evaluate_iteration(
            iteration=0,
            X_train=X_train,
            y_train=y_train,
            unlabeled_size=len(unlabeled_pool),
            new_labeled=len(X_labeled),
            X_val=X_val,
            y_val=y_val
        )
        self.history.append(initial_metrics)
        
        # Vòng lặp self-training
        for iteration in range(1, self.max_iter + 1):
            if len(unlabeled_pool) == 0:
                if self.verbose:
                    print(f"\n[Iteration {iteration}] No more unlabeled data. Stopping.")
                break
            
            # Dự đoán xác suất cho unlabeled pool
            proba = self.model.predict_proba(unlabeled_pool)
            max_proba = np.max(proba, axis=1)
            predicted_labels = np.argmax(proba, axis=1)
            
            # Chọn mẫu có độ tin cậy cao
            confident_mask = max_proba >= self.confidence_threshold
            confident_indices = np.where(confident_mask)[0]
            
            if len(confident_indices) < self.min_new_per_iter:
                if self.verbose:
                    print(f"\n[Iteration {iteration}] Only {len(confident_indices)} confident samples (< {self.min_new_per_iter}). Stopping.")
                break
            
            # Thêm mẫu tự tin vào tập huấn luyện
            X_new = unlabeled_pool.iloc[confident_indices]
            y_new = predicted_labels[confident_indices]
            
            X_train = pd.concat([X_train, X_new], ignore_index=True)
            y_train = pd.concat([y_train, pd.Series(y_new)], ignore_index=True)
            
            # Loại bỏ khỏi unlabeled pool
            unlabeled_pool = unlabeled_pool.drop(unlabeled_pool.index[confident_indices])
            unlabeled_pool = unlabeled_pool.reset_index(drop=True)
            
            # Huấn luyện lại mô hình
            self.model = clone(self.base_model)
            self.model.fit(X_train, y_train)
            
            # Đánh giá
            iter_metrics = self._evaluate_iteration(
                iteration=iteration,
                X_train=X_train,
                y_train=y_train,
                unlabeled_size=len(unlabeled_pool),
                new_labeled=len(confident_indices),
                X_val=X_val,
                y_val=y_val,
                avg_confidence=float(np.mean(max_proba[confident_mask]))
            )
            self.history.append(iter_metrics)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("HOÀN THÀNH SELF-TRAINING")
            print(f"Final training size: {len(X_train)}")
            print(f"Unlabeled remaining: {len(unlabeled_pool)}")
            print("=" * 70)
        
        return self
    
    def _evaluate_iteration(
        self,
        iteration: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        unlabeled_size: int,
        new_labeled: int,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        avg_confidence: float = None
    ) -> Dict:
        """
        Đánh giá sau mỗi vòng lặp
        """
        metrics = {
            'iteration': iteration,
            'train_size': len(X_train),
            'unlabeled_size': unlabeled_size,
            'new_labeled': new_labeled
        }
        
        if avg_confidence is not None:
            metrics['avg_confidence'] = avg_confidence
        
        if X_val is not None and y_val is not None:
            from sklearn.metrics import accuracy_score, f1_score
            
            y_val_pred = self.model.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            metrics['val_f1_macro'] = f1_score(y_val, y_val_pred, average='macro')
        
        if self.verbose:
            print(f"\n[Iteration {iteration}]")
            print(f"  Training size: {metrics['train_size']}")
            print(f"  New labeled: {metrics['new_labeled']}")
            print(f"  Unlabeled remaining: {metrics['unlabeled_size']}")
            if 'avg_confidence' in metrics:
                print(f"  Avg confidence: {metrics['avg_confidence']:.4f}")
            if 'val_accuracy' in metrics:
                print(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
                print(f"  Val F1 (macro): {metrics['val_f1_macro']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán nhãn
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán xác suất
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.model.predict_proba(X)
    
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
    
    def plot_history(self, save_path: str = None):
        """
        Vẽ biểu đồ quá trình self-training
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = [h['iteration'] for h in self.history]
        train_sizes = [h['train_size'] for h in self.history]
        new_labeled = [h['new_labeled'] for h in self.history]
        unlabeled_sizes = [h['unlabeled_size'] for h in self.history]
        
        # Training size
        axes[0, 0].plot(iterations, train_sizes, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Training Size', fontsize=12)
        axes[0, 0].set_title('Training Set Size Growth', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # New labeled per iteration
        axes[0, 1].bar(iterations[1:], new_labeled[1:], alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('New Labeled Samples', fontsize=12)
        axes[0, 1].set_title('Samples Added Each Iteration', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Unlabeled pool size
        axes[1, 0].plot(iterations, unlabeled_sizes, marker='s', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Unlabeled Pool Size', fontsize=12)
        axes[1, 0].set_title('Unlabeled Pool Depletion', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation metrics
        if 'val_accuracy' in self.history[0]:
            val_acc = [h['val_accuracy'] for h in self.history]
            val_f1 = [h['val_f1_macro'] for h in self.history]
            
            axes[1, 1].plot(iterations, val_acc, marker='o', linewidth=2, label='Accuracy')
            axes[1, 1].plot(iterations, val_f1, marker='s', linewidth=2, label='F1 (macro)')
            axes[1, 1].set_xlabel('Iteration', fontsize=12)
            axes[1, 1].set_ylabel('Score', fontsize=12)
            axes[1, 1].set_title('Validation Performance', fontsize=14, fontweight='bold')
            axes[1, 1].legend(fontsize=11)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Validation Data', 
                           ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Ví dụ sử dụng
    print("Self-Training Module")
    print("Use this module with preprocessed data")
