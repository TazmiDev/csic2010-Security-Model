import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from feature_extractor import FeatureExtractor


class CSIC2010Dataset(Dataset):
    """
    网络安全数据集类，用于加载和预处理CSIC 2010数据
    """
    def __init__(self, csv_path: str, feature_extractor: FeatureExtractor, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.max_length = max_length

        # 将标签转换为数值格式：'norm' -> 0, 其他 -> 1
        self.df['label'] = self.df['label'].apply(lambda x: 0 if x == 'norm' else 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        payload = str(self.df.iloc[idx]['payload'])
        label = self.df.iloc[idx]['label']

        # 提取所有类型的特征
        lexical, statistical, pattern, sequence = self.feature_extractor.extract_all_features(payload)

        # 转换为张量
        lexical_tensor = torch.tensor(lexical, dtype=torch.float32)
        statistical_tensor = torch.tensor(statistical, dtype=torch.float32)
        pattern_tensor = torch.tensor(pattern, dtype=torch.float32)
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, lexical_tensor, statistical_tensor, pattern_tensor, label_tensor


def collate_fn(batch):
    sequences, lexicals, statisticals, patterns, labels = zip(*batch)
    
    sequences = torch.stack(sequences).long()
    lexicals = torch.stack(lexicals).float()
    statisticals = torch.stack(statisticals).float()
    patterns = torch.stack(patterns).float()
    labels = torch.stack(labels).long()
    
    return sequences, lexicals, statisticals, patterns, labels


def create_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    feature_extractor = FeatureExtractor(max_length=max_length)

    train_dataset = CSIC2010Dataset(train_path, feature_extractor, max_length)
    test_dataset = CSIC2010Dataset(test_path, feature_extractor, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


def create_combined_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    feature_extractor = FeatureExtractor(max_length=max_length)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'norm' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'norm' else 1)

    train_features = []
    train_labels = []

    for _, row in train_df.iterrows():
        payload = str(row['payload'])
        features = feature_extractor.extract_combined_features(payload)
        train_features.append(features)
        train_labels.append(row['label'])

    test_features = []
    test_labels = []

    for _, row in test_df.iterrows():
        payload = str(row['payload'])
        features = feature_extractor.extract_combined_features(payload)
        test_features.append(features)
        test_labels.append(row['label'])

    train_features = np.array(train_features, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_features = np.array(test_features, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features),
        torch.from_numpy(test_labels)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
