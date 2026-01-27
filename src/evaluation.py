"""
Module đánh giá mô hình cho dự án AIR GUARD
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List


class ModelEvaluator:
    """
    Lớp đánh giá hiệu năng mô hình phân loại
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: Tên các lớp (theo thứ tự label 0, 1, 2,...)
        """
        if class_names is None:
            self.class_names = [
                'Good', 'Moderate', 'Unhealthy_Sensitive',
                'Unhealthy', 'Very_Unhealthy', 'Hazardous'
            ]
        else:
            self.class_names = class_names
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Đánh giá mô hình với các metrics cơ bản
        
        Args:
            y_true: Nhãn thực tế
            y_pred: Nhãn dự đoán
            model_name: Tên mô hình
            
        Returns:
            Dictionary chứa các metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics['per_class'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i])
                }
        
        return metrics
    
    def print_evaluation(self, metrics: Dict):
        """
        In kết quả đánh giá
        """
        print("\n" + "=" * 70)
        print(f"ĐÁNH GIÁ MÔ HÌNH: {metrics['model_name']}")
        print("=" * 70)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1-score (macro):  {metrics['f1_macro']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 60)
        for class_name, scores in metrics['per_class'].items():
            print(f"{class_name:<25} {scores['precision']:>10.4f} {scores['recall']:>10.4f} {scores['f1']:>10.4f}")
        print("=" * 70)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: str = None
    ):
        """
        Vẽ confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]],
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def compare_models(
        self,
        metrics_list: List[Dict],
        save_path: str = None
    ):
        """
        So sánh nhiều mô hình
        """
        model_names = [m['model_name'] for m in metrics_list]
        accuracy = [m['accuracy'] for m in metrics_list]
        f1_macro = [m['f1_macro'] for m in metrics_list]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        axes[0].bar(model_names, accuracy, alpha=0.7, color='steelblue')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracy):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # F1-macro comparison
        axes[1].bar(model_names, f1_macro, alpha=0.7, color='coral')
        axes[1].set_ylabel('F1-Score (Macro)', fontsize=12)
        axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(f1_macro):
            axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_comparison(
        self,
        metrics_list: List[Dict],
        metric_name: str = 'f1',
        save_path: str = None
    ):
        """
        So sánh hiệu năng từng lớp giữa các mô hình
        """
        model_names = [m['model_name'] for m in metrics_list]
        
        # Collect data
        classes = list(metrics_list[0]['per_class'].keys())
        data = []
        
        for model_metrics in metrics_list:
            model_scores = [
                model_metrics['per_class'][cls][metric_name]
                for cls in classes
            ]
            data.append(model_scores)
        
        # Plot
        x = np.arange(len(classes))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, (model_name, scores) in enumerate(zip(model_names, data)):
            offset = width * i - width * (len(model_names) - 1) / 2
            ax.bar(x + offset, scores, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('AQI Category', fontsize=12)
        ax.set_ylabel(f'{metric_name.upper()}-Score', fontsize=12)
        ax.set_title(f'Per-Class {metric_name.upper()}-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-class comparison saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, metrics: Dict, filepath: str):
        """
        Lưu metrics ra file JSON
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> Dict:
        """
        Đọc metrics từ file JSON
        """
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        return metrics


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Use this module to evaluate classification models")
