# Linear A ML Decipherment Project
# Complete implementation structure with working modules

"""
Project Directory Structure:
linear_a_decipherment/
├── data/
│   ├── raw/
│   │   ├── linear_a_corpus/
│   │   ├── linear_b_corpus/
│   │   ├── images/
│   │   └── external_sources/
│   ├── processed/
│   │   ├── tokenized/
│   │   ├── embeddings/
│   │   └── splits/
│   └── metadata/
├── src/
│   ├── data_processing/
│   ├── computer_vision/
│   ├── linguistic_analysis/
│   ├── models/
│   ├── transfer_learning/
│   └── validation/
├── notebooks/
├── tests/
├── configs/
└── outputs/
"""

# ============================================================================
# 1. SIGN INVENTORY MANAGER
# ============================================================================

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class LinearASign:
    """Represents a single Linear A sign with metadata"""
    sign_id: str
    unicode: Optional[str]
    category: str  # syllabogram, logogram, numeral, transaction
    frequency: int
    variants: List[str]
    linear_b_equivalent: Optional[str]
    positions: Dict[str, float]  # initial, medial, final distribution
    
class SignInventory:
    """Manages the canonical Linear A sign inventory"""
    
    def __init__(self):
        self.signs: Dict[str, LinearASign] = {}
        self.sign_to_id: Dict[str, str] = {}
        
    def add_sign(self, sign: LinearASign):
        self.signs[sign.sign_id] = sign
        if sign.unicode:
            self.sign_to_id[sign.unicode] = sign.sign_id
            
    def get_sign(self, identifier: str) -> Optional[LinearASign]:
        if identifier in self.signs:
            return self.signs[identifier]
        return self.signs.get(self.sign_to_id.get(identifier))
    
    def save(self, path: str):
        data = {sid: asdict(sign) for sid, sign in self.signs.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        inventory = cls()
        for sid, sign_data in data.items():
            sign = LinearASign(**sign_data)
            inventory.add_sign(sign)
        return inventory

# ============================================================================
# 2. CORPUS TOKENIZER
# ============================================================================

import re
from pathlib import Path

class LinearATokenizer:
    """Tokenizes Linear A inscriptions with metadata preservation"""
    
    def __init__(self, sign_inventory: SignInventory):
        self.inventory = sign_inventory
        self.uncertain_pattern = re.compile(r'[\[\]⟨⟩{}]')
        
    def tokenize_inscription(self, text: str, preserve_structure: bool = True):
        """
        Convert inscription to sign sequence
        Handles: word breaks (|), numerals, uncertain readings
        """
        tokens = []
        words = text.split('|')
        
        for word in words:
            word_tokens = []
            # Handle uncertain readings
            chars = list(word.strip())
            buffer = ""
            uncertain = False
            
            for char in chars:
                if char in '[]⟨⟩{}':
                    uncertain = not uncertain
                    continue
                    
                if char == ' ':
                    if buffer:
                        sign = self.inventory.get_sign(buffer)
                        word_tokens.append({
                            'sign': buffer,
                            'sign_id': sign.sign_id if sign else None,
                            'uncertain': uncertain
                        })
                        buffer = ""
                else:
                    buffer += char
                    
            if buffer:
                sign = self.inventory.get_sign(buffer)
                word_tokens.append({
                    'sign': buffer,
                    'sign_id': sign.sign_id if sign else None,
                    'uncertain': uncertain
                })
                
            if word_tokens:
                tokens.append(word_tokens)
                
        return tokens
    
    def process_corpus(self, corpus_path: Path, output_path: Path):
        """Process entire corpus directory"""
        results = []
        
        for file in corpus_path.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        # Format: DOC_ID: TEXT
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            doc_id, text = parts
                            tokens = self.tokenize_inscription(text.strip())
                            results.append({
                                'doc_id': doc_id.strip(),
                                'tokens': tokens,
                                'source_file': file.name
                            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

# ============================================================================
# 3. STATISTICAL ANALYSIS MODULE
# ============================================================================

import numpy as np
from collections import Counter, defaultdict
from scipy import stats

class LinguisticAnalyzer:
    """Performs statistical linguistic analysis on tokenized corpus"""
    
    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self.sign_sequences = self._extract_sequences()
        
    def _extract_sequences(self) -> List[List[str]]:
        sequences = []
        for doc in self.corpus:
            for word in doc['tokens']:
                seq = [token['sign_id'] for token in word if token['sign_id']]
                if seq:
                    sequences.append(seq)
        return sequences
    
    def compute_ngram_frequencies(self, n: int = 2) -> Counter:
        """Calculate n-gram frequencies"""
        ngrams = []
        for seq in self.sign_sequences:
            for i in range(len(seq) - n + 1):
                ngrams.append(tuple(seq[i:i+n]))
        return Counter(ngrams)
    
    def compute_entropy(self, n: int = 1) -> float:
        """Calculate Shannon entropy for n-grams"""
        ngrams = self.compute_ngram_frequencies(n)
        total = sum(ngrams.values())
        probs = [count / total for count in ngrams.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def conditional_entropy(self) -> float:
        """H(X_n | X_{n-1})"""
        unigram_entropy = self.compute_entropy(1)
        bigram_entropy = self.compute_entropy(2)
        return bigram_entropy - unigram_entropy
    
    def test_zipf_law(self) -> Dict:
        """Test if word frequencies follow Zipf's law"""
        word_strs = [''.join(seq) for seq in self.sign_sequences]
        freq = Counter(word_strs)
        ranks = np.arange(1, len(freq) + 1)
        frequencies = np.array(sorted(freq.values(), reverse=True))
        
        # Log-log regression
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_ranks, log_freqs
        )
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'zipf_compliant': abs(slope + 1) < 0.3  # Theoretical slope = -1
        }
    
    def positional_distribution(self) -> Dict[str, Dict]:
        """Analyze sign position preferences"""
        position_counts = defaultdict(lambda: {'initial': 0, 'medial': 0, 'final': 0})
        
        for seq in self.sign_sequences:
            if len(seq) == 1:
                position_counts[seq[0]]['initial'] += 1
                position_counts[seq[0]]['final'] += 1
            else:
                position_counts[seq[0]]['initial'] += 1
                position_counts[seq[-1]]['final'] += 1
                for sign in seq[1:-1]:
                    position_counts[sign]['medial'] += 1
        
        # Normalize
        for sign, counts in position_counts.items():
            total = sum(counts.values())
            position_counts[sign] = {
                pos: count / total for pos, count in counts.items()
            }
        
        return dict(position_counts)
    
    def mutual_information_matrix(self) -> pd.DataFrame:
        """Calculate MI between sign pairs"""
        bigrams = self.compute_ngram_frequencies(2)
        unigrams = self.compute_ngram_frequencies(1)
        total = sum(unigrams.values())
        
        signs = sorted(unigrams.keys())
        mi_matrix = np.zeros((len(signs), len(signs)))
        
        for i, sign1 in enumerate(signs):
            for j, sign2 in enumerate(signs):
                p_xy = bigrams.get((sign1, sign2), 0) / total
                p_x = unigrams[sign1] / total
                p_y = unigrams[sign2] / total
                
                if p_xy > 0:
                    mi_matrix[i, j] = p_xy * np.log2(p_xy / (p_x * p_y))
        
        return pd.DataFrame(mi_matrix, index=signs, columns=signs)

# ============================================================================
# 4. COMPUTER VISION PIPELINE
# ============================================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class LinearASignDetector:
    """YOLO-based sign detection in tablet images"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Placeholder for YOLOv8 - requires ultralytics package
        self.model = None  # Load pretrained YOLO
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_signs(self, image_path: str) -> List[Dict]:
        """Detect sign bounding boxes in image"""
        # Returns: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        raise NotImplementedError("Requires trained YOLO model")

class SignClassifier(nn.Module):
    """ResNet-based sign classification"""
    
    def __init__(self, num_classes: int = 90):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class CVPipeline:
    """End-to-end computer vision pipeline"""
    
    def __init__(self, detector: LinearASignDetector, classifier: SignClassifier):
        self.detector = detector
        self.classifier = classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def process_tablet_image(self, image_path: str) -> List[Dict]:
        """Extract sign sequence from tablet image"""
        detections = self.detector.detect_signs(image_path)
        image = Image.open(image_path)
        
        results = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            sign_crop = image.crop((x1, y1, x2, y2))
            sign_tensor = self.transform(sign_crop).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.classifier(sign_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            results.append({
                'bbox': det['bbox'],
                'predicted_sign': pred_class,
                'confidence': confidence
            })
        
        return sorted(results, key=lambda x: x['bbox'][0])  # Left-to-right

# ============================================================================
# 5. TRANSFER LEARNING MODULE
# ============================================================================

class MultiTaskLinearModel(nn.Module):
    """Joint Linear A + Linear B sequence modeling"""
    
    def __init__(self, vocab_size_a: int, vocab_size_b: int, 
                 embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        # Shared embedding for common signs
        self.shared_embedding = nn.Embedding(vocab_size_a, embedding_dim)
        
        # Separate LSTMs for each script
        self.lstm_a = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Prediction heads
        self.head_a = nn.Linear(hidden_dim, vocab_size_a)
        self.head_b = nn.Linear(hidden_dim, vocab_size_b)
        
    def forward(self, x, script='a'):
        emb = self.shared_embedding(x)
        
        if script == 'a':
            out, _ = self.lstm_a(emb)
            return self.head_a(out)
        else:
            out, _ = self.lstm_b(emb)
            return self.head_b(out)

# ============================================================================
# 6. MAIN EXECUTION PIPELINE
# ============================================================================

class LinearAProject:
    """Orchestrates the entire decipherment project"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sign_inventory = None
        self.corpus = None
        self.analyzer = None
        
    def phase_0_setup(self):
        """Foundation and research"""
        print("Phase 0: Setting up project structure...")
        # Create directories
        for dir in ['data/raw', 'data/processed', 'outputs', 'models']:
            Path(dir).mkdir(parents=True, exist_ok=True)
    
    def phase_1_data_engineering(self):
        """Build datasets"""
        print("Phase 1: Data engineering...")
        
        # Initialize sign inventory
        self.sign_inventory = SignInventory()
        # Add signs (would load from database in practice)
        
        # Tokenize corpus
        tokenizer = LinearATokenizer(self.sign_inventory)
        self.corpus = tokenizer.process_corpus(
            Path('data/raw/linear_a_corpus'),
            Path('data/processed/tokenized_corpus.json')
        )
        
    def phase_2_cv_pipeline(self):
        """Computer vision setup"""
        print("Phase 2: Training CV models...")
        # Train sign detector and classifier
        # (Requires annotated images)
        
    def phase_3_linguistic_analysis(self):
        """Statistical analysis"""
        print("Phase 3: Linguistic analysis...")
        
        self.analyzer = LinguisticAnalyzer(self.corpus)
        
        # Entropy
        uni_entropy = self.analyzer.compute_entropy(1)
        bi_entropy = self.analyzer.compute_entropy(2)
        cond_entropy = self.analyzer.conditional_entropy()
        
        # Zipf's law
        zipf_results = self.analyzer.test_zipf_law()
        
        # Positional analysis
        positions = self.analyzer.positional_distribution()
        
        print(f"Unigram entropy: {uni_entropy:.3f} bits")
        print(f"Conditional entropy: {cond_entropy:.3f} bits")
        print(f"Zipf compliance: {zipf_results['zipf_compliant']}")
        
        return {
            'entropy': {'unigram': uni_entropy, 'conditional': cond_entropy},
            'zipf': zipf_results,
            'positions': positions
        }
    
    def run_full_pipeline(self):
        """Execute complete analysis"""
        self.phase_0_setup()
        self.phase_1_data_engineering()
        # self.phase_2_cv_pipeline()  # Requires image data
        results = self.phase_3_linguistic_analysis()
        return results

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    config = {
        'data_path': 'data/raw',
        'output_path': 'outputs',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    project = LinearAProject(config)
    
    # Run phases sequentially
    # results = project.run_full_pipeline()
    
    print("Linear A Decipherment Framework Initialized")
    print("Ready for data ingestion and analysis")