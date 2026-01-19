import re
import numpy as np
from typing import Tuple


class FeatureExtractor:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.vocab = {}
        self.vocab_size = 0
        self._build_vocab()

    def _build_vocab(self):
        common_keywords = [
            'select', 'insert', 'update', 'delete', 'drop', 'union', 'where',
            'script', 'alert', 'onerror', 'onload', 'eval', 'expression',
            'iframe', 'document', 'window', 'location', 'cookie', 'href',
            'javascript', 'vbscript', 'data:', 'fromCharCode', 'unescape',
            'exec', 'system', 'cmd', 'bash', 'sh', 'powershell', 'wget',
            'curl', 'nc', 'netcat', 'telnet', 'ftp', 'tftp', 'http', 'https',
            'admin', 'root', 'password', 'passwd', 'login', 'user', 'username',
            'etc/passwd', 'boot.ini', 'win.ini', 'config', 'php', 'asp', 'jsp',
            'cgi', 'shtml', 'include', 'require', 'file_get_contents', 'fopen',
            'file_put_contents', 'fwrite', 'fread', 'readfile', 'file', 'glob',
            'dir', 'ls', 'cat', 'more', 'less', 'head', 'tail', 'grep', 'find',
            'chmod', 'chown', 'mv', 'cp', 'rm', 'mkdir', 'rmdir', 'ln', 'touch'
        ]

        for idx, keyword in enumerate(common_keywords):
            self.vocab[keyword.lower()] = idx + 1

        self.vocab_size = len(self.vocab) + 1

    def extract_lexical_features(self, payload: str) -> np.ndarray:
        features = []

        features.append(len(payload))
        features.append(len(payload.split()))
        features.append(len(re.findall(r'[A-Z]', payload)))
        features.append(len(re.findall(r'[a-z]', payload)))
        features.append(len(re.findall(r'[0-9]', payload)))
        features.append(len(re.findall(r'[^a-zA-Z0-9\s]', payload)))
        features.append(len(re.findall(r'<[^>]+>', payload)))
        features.append(len(re.findall(r'&[a-zA-Z]+;', payload)))
        features.append(len(re.findall(r'\\x[0-9a-fA-F]{2}', payload)))
        features.append(len(re.findall(r'%[0-9a-fA-F]{2}', payload)))
        features.append(len(re.findall(r'/\*.*?\*/', payload, re.DOTALL)))
        features.append(len(re.findall(r'--.*$', payload, re.MULTILINE)))
        features.append(len(re.findall(r'#.*$', payload, re.MULTILINE)))
        features.append(len(re.findall(r'\|', payload)))
        features.append(len(re.findall(r'&', payload)))
        features.append(len(re.findall(r';', payload)))
        features.append(len(re.findall(r'\$\(', payload)))
        features.append(len(re.findall(r'`[^`]*`', payload)))
        features.append(len(re.findall(r'\${[^}]+}', payload)))

        return np.array(features, dtype=np.float32)

    def extract_statistical_features(self, payload: str) -> np.ndarray:
        features = []

        chars = list(payload)
        if chars:
            features.append(np.mean([ord(c) for c in chars]))
            features.append(np.std([ord(c) for c in chars]))
        else:
            features.extend([0.0, 0.0])

        words = payload.split()
        if words:
            word_lengths = [len(w) for w in words]
            features.append(np.mean(word_lengths))
            features.append(np.std(word_lengths))
            features.append(max(word_lengths))
            features.append(min(word_lengths))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        features.append(payload.count(' '))
        features.append(payload.count('\t'))
        features.append(payload.count('\n'))
        features.append(payload.count('\r'))
        features.append(payload.count('.'))
        features.append(payload.count(','))
        features.append(payload.count('/'))
        features.append(payload.count('?'))

        return np.array(features, dtype=np.float32)

    def extract_pattern_features(self, payload: str) -> np.ndarray:
        features = []

        patterns = {
            'sql_injection': [
                r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE)\b",
                r"(?i)\b(OR|AND)\s+\d+\s*=\s*\d+",
                r"(?i)'.*OR.*'.*=.*'",
                r"(?i)\b(1=1|1=2|2=2)\b",
                r"(?i)--\s*$",
                r"(?i)/\*.*\*/",
                r"(?i)\b(exec|execute)\s*\(",
            ],
            'xss': [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"(?i)<iframe",
                r"(?i)<object",
                r"(?i)<embed",
                r"(?i)eval\s*\(",
                r"(?i)fromCharCode",
            ],
            'command_injection': [
                r"(?i)(;|\||&)\s*(\w+)",
                r"(?i)\$\([^)]+\)",
                r"(?i)`[^`]+`",
                r"(?i)\$\{[^}]+\}",
                r"(?i)(wget|curl|nc|netcat|telnet|ftp|tftp)\s+",
                r"(?i)(bash|sh|cmd|powershell)\s+",
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\.\.%2f",
                r"\.\.%5c",
                r"/etc/passwd",
                r"c:\\windows\\system32",
            ],
            'directory_listing': [
                r"(?i)\b(ls|dir|ll)\s+",
                r"(?i)\b(find|locate)\s+",
                r"(?i)\b(grep|egrep|fgrep)\s+",
            ],
        }

        for category, pattern_list in patterns.items():
            match_count = 0
            for pattern in pattern_list:
                matches = re.findall(pattern, payload, re.IGNORECASE | re.DOTALL)
                match_count += len(matches)
            features.append(float(match_count))

        return np.array(features, dtype=np.float32)

    def extract_sequence_features(self, payload: str) -> np.ndarray:
        normalized = payload.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        words = normalized.split()

        sequence = []
        for word in words[:self.max_length]:
            if word in self.vocab:
                sequence.append(self.vocab[word])
            else:
                sequence.append(0)

        while len(sequence) < self.max_length:
            sequence.append(0)

        return np.array(sequence, dtype=np.int64)

    def extract_all_features(self, payload: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lexical = self.extract_lexical_features(payload)
        statistical = self.extract_statistical_features(payload)
        pattern = self.extract_pattern_features(payload)
        sequence = self.extract_sequence_features(payload)

        return lexical, statistical, pattern, sequence

    def extract_combined_features(self, payload: str) -> np.ndarray:
        lexical, statistical, pattern, sequence = self.extract_all_features(payload)

        combined = np.concatenate([
            lexical,
            statistical,
            pattern,
            sequence.astype(np.float32) / self.vocab_size
        ])

        return combined.astype(np.float32)
