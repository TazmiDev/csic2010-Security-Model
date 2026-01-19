import torch
import numpy as np
import onnxruntime as ort
from training.model import LightweightModel
from training.feature_extractor import FeatureExtractor


def test_pytorch_model(model_path: str, test_payloads: list):
    print('='*60)
    print('Testing PyTorch Model')
    print('='*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    model = LightweightModel(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_extractor = FeatureExtractor(max_length=512)
    
    results = []
    for payload in test_payloads:
        lexical, statistical, pattern, sequence = feature_extractor.extract_all_features(payload)
        
        with torch.no_grad():
            sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
            lexical_tensor = torch.tensor(lexical, dtype=torch.float32).unsqueeze(0).to(device)
            statistical_tensor = torch.tensor(statistical, dtype=torch.float32).unsqueeze(0).to(device)
            pattern_tensor = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(sequence_tensor, lexical_tensor, statistical_tensor, pattern_tensor)
            probs = torch.softmax(outputs, dim=1)
            risk_score = probs[0, 1].item()
            prediction = torch.argmax(outputs, dim=1).item()
        
        results.append({
            'payload': payload,
            'prediction': prediction,
            'risk_score': risk_score,
            'prob_normal': probs[0, 0].item(),
            'prob_attack': probs[0, 1].item()
        })
    
    return results


def test_onnx_model(onnx_path: str, test_payloads: list):
    print('\n' + '='*60)
    print('Testing ONNX Model')
    print('='*60)
    
    session = ort.InferenceSession(onnx_path)
    feature_extractor = FeatureExtractor(max_length=512)
    
    results = []
    for payload in test_payloads:
        lexical, statistical, pattern, sequence = feature_extractor.extract_all_features(payload)
        
        inputs = {
            'sequence': np.expand_dims(sequence, axis=0).astype(np.int64),
            'lexical': np.expand_dims(lexical, axis=0).astype(np.float32),
            'statistical': np.expand_dims(statistical, axis=0).astype(np.float32),
            'pattern': np.expand_dims(pattern, axis=0).astype(np.float32)
        }
        
        outputs = session.run(None, inputs)
        logits = outputs[0]
        probs = softmax(logits[0])
        prediction = np.argmax(probs)
        risk_score = probs[1]
        
        results.append({
            'payload': payload,
            'prediction': prediction,
            'risk_score': risk_score,
            'prob_normal': probs[0],
            'prob_attack': probs[1]
        })
    
    return results


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def compare_results(pytorch_results, onnx_results):
    print('\n' + '='*60)
    print('Comparison Results')
    print('='*60)
    
    for i, (pt, onnx) in enumerate(zip(pytorch_results, onnx_results)):
        print(f'\n--- Test Case {i+1} ---')
        print(f'Payload: {pt["payload"][:80]}...')
        print(f'PyTorch: pred={pt["prediction"]}, risk={pt["risk_score"]:.4f}, normal={pt["prob_normal"]:.4f}, attack={pt["prob_attack"]:.4f}')
        print(f'ONNX:    pred={onnx["prediction"]}, risk={onnx["risk_score"]:.4f}, normal={onnx["prob_normal"]:.4f}, attack={onnx["prob_attack"]:.4f}')
        
        pred_match = pt["prediction"] == onnx["prediction"]
        risk_diff = abs(pt["risk_score"] - onnx["risk_score"])
        
        if pred_match:
            print(f'✓ Predictions match')
        else:
            print(f'✗ Predictions differ!')
        
        if risk_diff < 0.001:
            print(f'✓ Risk scores match (diff: {risk_diff:.6f})')
        else:
            print(f'⚠ Risk scores differ (diff: {risk_diff:.6f})')


def main():
    model_path = '../models/waf_model.pth'
    onnx_path = '../models/waf_model.onnx'
    
    test_payloads = [
        '/index.php?id=1',
        '/index.php?id=1\' OR \'1\'=\'1',
        '<script>alert(1)</script>',
        '/normal/page.html',
        '/api/user?id=123',
        '/index.php?id=1; DROP TABLE users--',
        '<img src=x onerror=alert(1)>',
        '/search?q=test',
        '/login?username=admin&password=123',
        '/admin/config.php?file=../../etc/passwd'
    ]
    
    print('Testing PyTorch model...')
    pytorch_results = test_pytorch_model(model_path, test_payloads)
    
    print('\nTesting ONNX model...')
    onnx_results = test_onnx_model(onnx_path, test_payloads)
    
    compare_results(pytorch_results, onnx_results)
    
    print('\n' + '='*60)
    print('Test completed!')
    print('='*60)


if __name__ == '__main__':
    main()
