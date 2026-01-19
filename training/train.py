import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from model import LightweightModel
from data_loader import create_data_loaders
import os
import matplotlib.pyplot as plt


class Trainer:
    """
    模型训练器，负责训练和验证网络安全检测模型
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        # 使用交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 使用Adam优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 学习率调度器，当验证损失不再下降时降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # 记录训练过程中的指标
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        执行一个训练周期
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            if len(batch) == 5:
                sequence, lexical, statistical, pattern, labels = batch
                # 将数据移到指定设备上
                sequence = sequence.to(self.device)
                lexical = lexical.to(self.device)
                statistical = statistical.to(self.device)
                pattern = pattern.to(self.device)
                labels = labels.to(self.device)

                # 清零梯度，前向传播，计算损失，反向传播，更新参数
                self.optimizer.zero_grad()
                outputs = self.model(sequence, lexical, statistical, pattern)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型性能
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if len(batch) == 5:
                    sequence, lexical, statistical, pattern, labels = batch
                    # 将验证数据移到指定设备上
                    sequence = sequence.to(self.device)
                    lexical = lexical.to(self.device)
                    statistical = statistical.to(self.device)
                    pattern = pattern.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(sequence, lexical, statistical, pattern)
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_path: str = 'checkpoints'
    ):
        os.makedirs(save_path, exist_ok=True)

        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            self.scheduler.step(val_loss)

            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                self.save_model(os.path.join(save_path, 'best_model.pth'))
                print(f'Best model saved with val accuracy: {val_accuracy:.2f}%')

            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))

        print(f'\nTraining completed!')
        print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')

        self.plot_training_history(save_path)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)

    def plot_training_history(self, save_path: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()


def main():
    train_path = '../resources/payload_train.csv'
    test_path = '../resources/payload_test.csv'
    output_dir = '../models'
    
    os.makedirs(output_dir, exist_ok=True)

    print('Loading data...')
    train_loader, test_loader = create_data_loaders(
        train_path,
        test_path,
        batch_size=32,
        max_length=512,
        num_workers=0
    )

    print('Creating model...')
    model = LightweightModel(
        vocab_size=100,
        embedding_dim=64,
        num_filters=128,
        filter_sizes=[3, 4, 5],
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.3,
        lexical_size=19,
        statistical_size=14,
        pattern_size=5,
        sequence_length=512
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    trainer = Trainer(
        model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5
    )

    print('Starting training...')
    trainer.train(
        train_loader,
        test_loader,
        epochs=50,
        save_path=output_dir
    )

    print(f'\nModel saved to: {os.path.join(output_dir, "best_model.pth")}')

    final_model_path = os.path.join(output_dir, 'waf_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': 100,
            'embedding_dim': 64,
            'num_filters': 128,
            'filter_sizes': [3, 4, 5],
            'lstm_hidden': 64,
            'lstm_layers': 1,
            'dropout': 0.3,
            'lexical_size': 19,
            'statistical_size': 14,
            'pattern_size': 5,
            'sequence_length': 512
        },
        'training_info': {
            'train_losses': trainer.train_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_losses': trainer.val_losses,
            'val_accuracies': trainer.val_accuracies,
        }
    }, final_model_path)

    print(f'Final model saved to: {final_model_path}')


if __name__ == '__main__':
    main()
