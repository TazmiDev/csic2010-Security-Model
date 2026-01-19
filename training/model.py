import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightModel(nn.Module):
    """
    轻量级网络安全检测模型，结合CNN、LSTM和多种特征提取器进行恶意载荷检测
    """
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 64,
        num_filters: int = 128,
        filter_sizes: list = [3, 4, 5],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        lexical_size: int = 19,
        statistical_size: int = 14,
        pattern_size: int = 5,
        sequence_length: int = 512
    ):
        super(LightweightModel, self).__init__()

        # 词嵌入层，将输入序列转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 卷积层，使用不同大小的卷积核提取局部特征
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # LSTM层，捕获序列中的长期依赖关系
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # 词汇特征处理层，将词汇学特征映射到固定维度
        self.lexical_fc = nn.Sequential(
            nn.Linear(lexical_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # 统计特征处理层，将统计学特征映射到固定维度
        self.statistical_fc = nn.Sequential(
            nn.Linear(statistical_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # 模式特征处理层，将模式特征映射到固定维度
        self.pattern_fc = nn.Sequential(
            nn.Linear(pattern_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        cnn_output_size = num_filters * len(filter_sizes)
        lstm_output_size = lstm_hidden * 2
        combined_size = cnn_output_size + lstm_output_size + 32 + 32 + 32

        # 分类器，将所有特征组合后输出最终分类结果
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, sequence, lexical, statistical, pattern):
        # 序列嵌入和维度变换
        embedded = self.embedding(sequence)
        embedded = embedded.permute(0, 2, 1)

        # CNN特征提取
        cnn_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.adaptive_max_pool1d(conv_out, 1)
            cnn_outputs.append(pooled.squeeze(2))

        cnn_features = torch.cat(cnn_outputs, dim=1)

        # LSTM特征提取
        embedded_seq = embedded.permute(0, 2, 1)
        lstm_out, (hidden, _) = self.lstm(embedded_seq)
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # 处理各种手工提取的特征
        lexical_features = self.lexical_fc(lexical)
        statistical_features = self.statistical_fc(statistical)
        pattern_features = self.pattern_fc(pattern)

        # 合并所有特征
        combined = torch.cat([
            cnn_features,
            lstm_features,
            lexical_features,
            statistical_features,
            pattern_features
        ], dim=1)

        # 通过分类器得到最终输出
        output = self.classifier(combined)
        return output

    def get_risk_score(self, sequence, lexical, statistical, pattern):
        """
        获取风险评分，返回属于恶意类别（标签1）的概率
        """
        logits = self.forward(sequence, lexical, statistical, pattern)
        probs = F.softmax(logits, dim=1)
        risk_score = probs[:, 1]
        return risk_score


class UltraLightModel(nn.Module):
    """
    超轻量级网络安全检测模型，适用于资源受限环境
    """
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.2,
        feature_size: int = 38
    ):
        super(UltraLightModel, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 双层卷积层，用于提取局部特征
        self.conv1 = nn.Conv1d(embedding_dim, hidden_size, kernel_size=3)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3)

        # LSTM层，捕获序列信息
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=True
        )

        # 特征处理层
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, sequence, features):
        embedded = self.embedding(sequence)
        embedded = embedded.permute(0, 2, 1)

        x = F.relu(self.conv1(embedded))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)

        x = x.permute(0, 2, 1)
        lstm_out, (hidden, _) = self.lstm(x)
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)

        feature_features = self.feature_fc(features)

        combined = torch.cat([lstm_features, feature_features], dim=1)
        output = self.classifier(combined)

        return output

    def get_risk_score(self, sequence, features):
        logits = self.forward(sequence, features)
        probs = F.softmax(logits, dim=1)
        risk_score = probs[:, 1]
        return risk_score
