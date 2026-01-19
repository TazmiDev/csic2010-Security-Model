# 网络安全轻量模型

这是一个用于网络安全的轻量模型，用来帮助开发者测试自己的安全产品。

## 项目简介

本项目实现了一个基于深度学习的轻量级网络入侵检测模型，结合了CNN、LSTM和多种手工提取的特征，能够有效识别恶意网络载荷。模型设计注重效率和准确性，适合在资源受限的环境中部署。

## 功能特点

- **多模态特征融合**：结合词汇学、统计学和模式特征与深度学习特征
- **轻量级架构**：优化的网络结构，适合资源受限环境
- **高效检测**：快速准确地识别潜在的网络威胁
- **易于部署**：支持ONNX格式导出，便于跨平台部署

## 文件结构

```
models/           # 训练好的模型文件
resources/        # 数据集文件
test/             # 测试脚本
training/         # 训练相关代码
utils/            # 工具脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
   ```bash
   cd training
   python train.py
   ```

2. 导出ONNX模型：
   ```bash
   python ../utils/export_onnx.py --model_path path/to/model.pth --onnx_path path/to/model.onnx
   ```

3. 运行测试：
   ```bash
   cd test
   python test_onnx.py
   ```

## 模型架构

本项目提供了两种模型：

- **LightweightModel**：标准轻量级模型，平衡了准确性和效率
- **UltraLightModel**：超轻量级模型，适用于资源极度受限的场景

## 数据集

项目使用CSIC 2010数据集进行训练和测试，该数据集包含正常流量和恶意流量样本。