import torch
import torch.onnx
import argparse
import os
from training.model import LightweightModel, UltraLightModel


def export_to_onnx(
    model_path: str,
    onnx_path: str,
    model_type: str = 'lightweight',
    batch_size: int = 1,
    sequence_length: int = 512,
    opset_version: int = 14,
    dynamic_axes: bool = True
):
    """
    将PyTorch模型导出为ONNX格式
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading model from: {model_path}')
    checkpoint = torch.load(model_path, map_location=device)

    # 根据模型类型创建对应的模型实例
    if model_type == 'lightweight':
        model_config = checkpoint.get('model_config', {
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
        })
        model = LightweightModel(**model_config)
    else:
        model_config = checkpoint.get('model_config', {
            'vocab_size': 100,
            'embedding_dim': 32,
            'hidden_size': 64,
            'dropout': 0.2,
            'feature_size': 38
        })
        model = UltraLightModel(**model_config)

    # 加载模型权重并设置为评估模式
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f'Model config: {model_config}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    print('\nCreating dummy inputs...')
    if model_type == 'lightweight':
        dummy_sequence = torch.randint(0, 100, (batch_size, sequence_length)).to(device)
        dummy_lexical = torch.randn(batch_size, 19).to(device)
        dummy_statistical = torch.randn(batch_size, 14).to(device)
        dummy_pattern = torch.randn(batch_size, 5).to(device)
        dummy_inputs = (dummy_sequence, dummy_lexical, dummy_statistical, dummy_pattern)
        input_names = ['sequence', 'lexical', 'statistical', 'pattern']
    else:
        dummy_sequence = torch.randint(0, 100, (batch_size, sequence_length)).to(device)
        dummy_features = torch.randn(batch_size, 38).to(device)
        dummy_inputs = (dummy_sequence, dummy_features)
        input_names = ['sequence', 'features']

    output_names = ['output']

    if dynamic_axes:
        if model_type == 'lightweight':
            dynamic_axes_dict = {
                'sequence': {0: 'batch_size', 1: 'sequence_length'},
                'lexical': {0: 'batch_size'},
                'statistical': {0: 'batch_size'},
                'pattern': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes_dict = {
                'sequence': {0: 'batch_size', 1: 'sequence_length'},
                'features': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
    else:
        dynamic_axes_dict = None

    print(f'\nExporting to ONNX...')
    print(f'ONNX path: {onnx_path}')
    print(f'Opset version: {opset_version}')
    print(f'Dynamic axes: {dynamic_axes}')

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        dynamo=False
    )

    print(f'\n✓ Model successfully exported to: {onnx_path}')

    print('\nVerifying ONNX model...')
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('✓ ONNX model is valid')

    print('\nONNX Model Info:')
    print(f'  IR version: {onnx_model.ir_version}')
    print(f'  Producer name: {onnx_model.producer_name}')
    print(f'  Graph inputs: {[inp.name for inp in onnx_model.graph.input]}')
    print(f'  Graph outputs: {[out.name for out in onnx_model.graph.output]}')

    print('\nTesting ONNX model with ONNX Runtime...')
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path)

    if model_type == 'lightweight':
        ort_inputs = {
            'sequence': dummy_sequence.cpu().numpy(),
            'lexical': dummy_lexical.cpu().numpy(),
            'statistical': dummy_statistical.cpu().numpy(),
            'pattern': dummy_pattern.cpu().numpy()
        }
    else:
        ort_inputs = {
            'sequence': dummy_sequence.cpu().numpy(),
            'features': dummy_features.cpu().numpy()
        }

    ort_outputs = ort_session.run(None, ort_inputs)

    print('\nComparing PyTorch and ONNX Runtime outputs...')
    with torch.no_grad():
        torch_outputs = model(*dummy_inputs)

    max_diff = torch.max(torch.abs(torch_outputs - torch.tensor(ort_outputs))).item()
    print(f'Maximum difference: {max_diff:.6f}')

    if max_diff < 1e-4:
        print('✓ Outputs match within tolerance (1e-4)')
    else:
        print(f'⚠ Warning: Outputs differ by {max_diff:.6f}')

    print('\n' + '='*60)
    print('ONNX export completed successfully!')
    print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch WAF model to ONNX format')
    parser.add_argument(
        '--model_path',
        type=str,
        default='../models/waf_model.pth',
        help='Path to the PyTorch model file (.pth)'
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        default='../models/waf_model.onnx',
        help='Path to save the ONNX model file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['lightweight', 'ultralight'],
        default='lightweight',
        help='Type of model to export'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for dummy input'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=512,
        help='Sequence length for dummy input'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=14,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--static',
        action='store_true',
        help='Export with static axes (no dynamic batch/sequence)'
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)

    export_to_onnx(
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        opset_version=args.opset_version,
        dynamic_axes=not args.static
    )


if __name__ == '__main__':
    main()
