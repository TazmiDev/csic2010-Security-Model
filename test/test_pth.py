import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

from training.model import LightweightModel
from training.data_loader import create_data_loaders
from training.feature_extractor import FeatureExtractor

class ModelEvaluator:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.feature_extractor = FeatureExtractor()
        
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', {})
        self.model = LightweightModel(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f'Model loaded from: {self.model_path}')
        print(f'Model config: {model_config}')
        
    def evaluate(self, test_loader: DataLoader):
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequence_features, lexical_features, statistical_features, pattern_features, labels = batch
                
                lexical_features = lexical_features.to(self.device)
                statistical_features = statistical_features.to(self.device)
                pattern_features = pattern_features.to(self.device)
                sequence_features = sequence_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(
                    sequence_features,
                    lexical_features,
                    statistical_features,
                    pattern_features
                )
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, predictions, labels):
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary'),
            'recall': recall_score(labels, predictions, average='binary'),
            'f1_score': f1_score(labels, predictions, average='binary'),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        print('\n' + '='*60)
        print('模型评估结果 / Model Evaluation Results')
        print('='*60)
        print(f'准确率 / Accuracy:  {metrics["accuracy"]:.4f}')
        print(f'精确率 / Precision: {metrics["precision"]:.4f}')
        print(f'召回率 / Recall:    {metrics["recall"]:.4f}')
        print(f'F1分数 / F1 Score:  {metrics["f1_score"]:.4f}')
        print('='*60)
        
        cm = metrics['confusion_matrix']
        print('\n混淆矩阵 / Confusion Matrix:')
        print('                预测正常    预测攻击')
        print(f'实际正常 / Normal:    {cm[0, 0]:6d}    {cm[0, 1]:6d}')
        print(f'实际攻击 / Attack:    {cm[1, 0]:6d}    {cm[1, 1]:6d}')
        print('='*60)
        
        tn, fp, fn, tp = cm.ravel()
        print(f'\n详细指标 / Detailed Metrics:')
        print(f'真阴性 / True Negative:  {tn}')
        print(f'假阳性 / False Positive: {fp}')
        print(f'假阴性 / False Negative: {fn}')
        print(f'真阳性 / True Positive:  {tp}')
        print('='*60)
        
        if fp + tn > 0:
            fpr = fp / (fp + tn)
            print(f'误报率 / False Positive Rate: {fpr:.4f}')
        
        if fn + tp > 0:
            fnr = fn / (fn + tp)
            print(f'漏报率 / False Negative Rate: {fnr:.4f}')
        
        print('='*60)
    
    def plot_confusion_matrix(self, cm, save_path: str = None):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('混淆矩阵 / Confusion Matrix')
        plt.ylabel('真实标签 / True Label')
        plt.xlabel('预测标签 / Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n混淆矩阵已保存到 / Confusion matrix saved to: {save_path}')
        
        plt.show()
    
    def plot_roc_curve(self, labels, probabilities, save_path: str = None):
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='随机猜测 / Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 / False Positive Rate')
        plt.ylabel('真阳性率 / True Positive Rate')
        plt.title('ROC曲线 / ROC Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'ROC曲线已保存到 / ROC curve saved to: {save_path}')
        
        plt.show()
        
        return roc_auc
    
    def analyze_errors(self, test_loader, predictions, labels, num_samples=5):
        print('\n' + '='*60)
        print('错误分析 / Error Analysis')
        print('='*60)
        
        error_indices = np.where(predictions != labels)[0]
        
        if len(error_indices) == 0:
            print('没有预测错误 / No prediction errors found!')
            return
        
        print(f'总错误数 / Total errors: {len(error_indices)}')
        print(f'错误率 / Error rate: {len(error_indices) / len(labels):.4f}')
        
        false_positives = error_indices[labels[error_indices] == 0]
        false_negatives = error_indices[labels[error_indices] == 1]
        
        print(f'\n假阳性 / False Positives (误报): {len(false_positives)}')
        print(f'假阴性 / False Negatives (漏报): {len(false_negatives)}')
        print('='*60)
        
        if len(false_positives) > 0:
            print('\n假阳性样本示例 / False Positive Samples:')
            for i, idx in enumerate(false_positives[:num_samples]):
                batch_idx = idx // test_loader.batch_size
                sample_idx = idx % test_loader.batch_size
                print(f'\n样本 {i+1} / Sample {i+1}:')
        
        if len(false_negatives) > 0:
            print('\n假阴性样本示例 / False Negative Samples:')
            for i, idx in enumerate(false_negatives[:num_samples]):
                batch_idx = idx // test_loader.batch_size
                sample_idx = idx % test_loader.batch_size
                print(f'\n样本 {i+1} / Sample {i+1}:')
        
        print('='*60)


def main():
    model_path = '../models/waf_model.pth'
    test_path = '../resources/payload_test.csv'
    output_dir = '../models'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print('加载模型 / Loading model...')
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    
    print('\n加载测试数据 / Loading test data...')
    _, test_loader = create_data_loaders(
        '../resources/payload_train.csv',
        test_path,
        batch_size=32,
        max_length=512,
        num_workers=0
    )
    
    print(f'测试集大小 / Test set size: {len(test_loader.dataset)}')
    
    print('\n开始评估 / Starting evaluation...')
    predictions, labels, probabilities = evaluator.evaluate(test_loader)
    
    metrics = evaluator.calculate_metrics(predictions, labels)
    evaluator.print_metrics(metrics)
    
    print('\n生成可视化图表 / Generating visualizations...')
    cm_save_path = os.path.join(output_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], cm_save_path)
    
    roc_save_path = os.path.join(output_dir, 'roc_curve.png')
    roc_auc = evaluator.plot_roc_curve(labels, probabilities, roc_save_path)
    print(f'ROC AUC: {roc_auc:.4f}')
    
    print('\n生成分类报告 / Generating classification report...')
    print(classification_report(labels, predictions, 
                              target_names=['Normal', 'Attack'],
                              digits=4))
    
    print('\n' + '='*60)
    print('评估完成 / Evaluation completed!')
    print('='*60)


if __name__ == '__main__':
    main()
