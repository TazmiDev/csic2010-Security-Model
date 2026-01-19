# csic2010-Security-Model

## Lightweight Network Security Model

This is a lightweight model for network security, designed to help developers test their security products.

[中文文档](README.md) | [English](README-EN.md)

## Project Introduction

This project implements a deep learning-based lightweight network intrusion detection model that combines CNN, LSTM, and multiple handcrafted features to effectively identify malicious network payloads. The model design focuses on efficiency and accuracy, suitable for deployment in resource-constrained environments.

## Features

- **Multi-modal Feature Fusion**: Combines lexical, statistical, and pattern features with deep learning features
- **Lightweight Architecture**: Optimized network structure suitable for resource-constrained environments
- **Efficient Detection**: Rapid and accurate identification of potential network threats
- **Easy Deployment**: Supports ONNX format export for cross-platform deployment

## File Structure

```
models/           # Trained model files
resources/        # Dataset files
test/             # Test scripts
training/         # Training-related code
utils/            # Utility scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
   ```bash
   cd training
   python train.py
   ```

2. Export ONNX model:
   ```bash
   python ../utils/export_onnx.py --model_path path/to/model.pth --onnx_path path/to/model.onnx
   ```

3. Run tests:
   ```bash
   cd test
   python test_onnx.py
   ```

## Model Architecture

This project provides two models:

- **LightweightModel**: Standard lightweight model balancing accuracy and efficiency
- **UltraLightModel**: Ultra-lightweight model for extremely resource-constrained scenarios

## Dataset

The project uses the CSIC 2010 dataset for training and testing, which contains samples of both normal traffic and malicious traffic.
