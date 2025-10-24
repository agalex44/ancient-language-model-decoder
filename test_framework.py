"""
Linear A Decipherment - Test Suite
Run with: pytest test_suite.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json
import tempfile

# Import from main framework
import sys
sys.path.insert(0, 'src/')

from typing import List, Dict

# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

class MockDataGenerator:
    """Generate synthetic test data"""
    
    @staticmethod
    def create_sign_inventory(num_signs: int = 90) -> Dict:
        """Create mock sign inventory"""
        signs = {}
        for i in range(num_signs):
            signs[f"LA{i:03d}"] = {
                'sign_id': f"LA{i:03d}",
                'unicode': f"U+{10800 + i:04X}",
                'category': ['syllabogram', 'logogram', 'numeral'][i % 3],
                'frequency': np.random.randint(1, 100),
                'variants': [f"LA{i:03d}_v{j}" for j in range(np.random.randint(1, 4))],
                'linear_b_equivalent': f"LB{i:03d}" if i < 70 else None,
                'positions': {
                    'initial': np.random.random(),
                    'medial': np.random.random(),
                    'final': np.random.random()
                }
            }
        return signs
    
    @staticmethod
    def create_corpus(num_docs: int = 100, vocab_size: int = 90) -> List[Dict]:
        """Create mock tokenized corpus"""
        corpus = []
        for i in range(num_docs):
            num_words = np.random.randint(3, 15)
            tokens = []
            for _ in range(num_words):
                word_length = np.random.randint(1, 6)
                word_tokens = []
                for _ in range(word_length):
                    sign_id = f"LA{np.random.randint(0, vocab_size):03d}"
                    word_tokens.append({
                        'sign': f"sign_{sign_id}",
                        'sign_id': sign_id,
                        'uncertain': np.random.random() < 0.1
                    })
                tokens.append(word_tokens)
            
            corpus.append({
                'doc_id': f"HT_{i:04d}",
                'tokens': tokens,
                'source_file': f"test_{i}.txt"
            })
        return corpus

# ============================================================================
# TESTS: SIGN INVENTORY
# ============================================================================

class TestSignInventory:
    """Test sign inventory management"""
    
    def test_sign_creation(self):
        """Test creating and storing signs"""
        from linear_a_project import SignInventory, LinearASign
        
        inventory = SignInventory()
        sign = LinearASign(
            sign_id="LA001",
            unicode="U+10800",
            category="syllabogram",
            frequency=50,
            variants=["LA001_v1", "LA001_v2"],
            linear_b_equivalent="LB001",
            positions={'initial': 0.3, 'medial': 0.5, 'final': 0.2}
        )
        
        inventory.add_sign(sign)
        retrieved = inventory.get_sign("LA001")
        
        assert retrieved is not None
        assert retrieved.sign_id == "LA001"
        assert retrieved.frequency == 50
        assert len(retrieved.variants) == 2
    
    def test_sign_lookup_by_unicode(self):
        """Test retrieving signs by Unicode codepoint"""
        from linear_a_project import SignInventory, LinearASign
        
        inventory = SignInventory()
        sign = LinearASign(
            sign_id="LA001",
            unicode="U+10800",
            category="syllabogram",
            frequency=50,
            variants=[],
            linear_b_equivalent=None,
            positions={}
        )
        
        inventory.add_sign(sign)
        retrieved = inventory.get_sign("U+10800")
        
        assert retrieved is not None
        assert retrieved.sign_id == "LA001"
    
    def test_sign_serialization(self):
        """Test saving and loading inventory"""
        from linear_a_project import SignInventory, LinearASign
        
        inventory = SignInventory()
        sign = LinearASign(
            sign_id="LA001",
            unicode="U+10800",
            category="syllabogram",
            frequency=50,
            variants=["LA001_v1"],
            linear_b_equivalent="LB001",
            positions={'initial': 0.3, 'medial': 0.5, 'final': 0.2}
        )
        inventory.add_sign(sign)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        inventory.save(temp_path)
        loaded = SignInventory.load(temp_path)
        
        assert len(loaded.signs) == 1
        assert loaded.get_sign("LA001").frequency == 50
        
        Path(temp_path).unlink()

# ============================================================================
# TESTS: TOKENIZATION
# ============================================================================

class TestTokenization:
    """Test corpus tokenization"""
    
    def test_basic_tokenization(self):
        """Test tokenizing simple inscription"""
        from linear_a_project import LinearATokenizer, SignInventory, LinearASign
        
        inventory = SignInventory()
        for i in range(10):
            sign = LinearASign(
                sign_id=f"LA{i:03d}",
                unicode=f"U+{10800+i:04X}",
                category="syllabogram",
                frequency=10,
                variants=[],
                linear_b_equivalent=None,
                positions={}
            )
            inventory.add_sign(sign)
        
        tokenizer = LinearATokenizer(inventory)
        text = "LA001 LA002 | LA003 LA004 LA005"
        tokens = tokenizer.tokenize_inscription(text)
        
        assert len(tokens) == 2  # Two words
        assert len(tokens[0]) == 2  # First word has 2 signs
        assert len(tokens[1]) == 3  # Second word has 3 signs
    
    def test_uncertain_readings(self):
        """Test handling uncertain sign readings"""
        from linear_a_project import LinearATokenizer, SignInventory, LinearASign
        
        inventory = SignInventory()
        sign = LinearASign(
            sign_id="LA001",
            unicode="LA001",
            category="syllabogram",
            frequency=10,
            variants=[],
            linear_b_equivalent=None,
            positions={}
        )
        inventory.add_sign(sign)
        
        tokenizer = LinearATokenizer(inventory)
        text = "[LA001]"  # Uncertain reading
        tokens = tokenizer.tokenize_inscription(text)
        
        assert len(tokens) == 1
        assert tokens[0][0]['uncertain'] == True

# ============================================================================
# TESTS: STATISTICAL ANALYSIS
# ============================================================================

class TestLinguisticAnalysis:
    """Test statistical linguistic analysis"""
    
    @pytest.fixture
    def mock_corpus(self):
        return MockDataGenerator.create_corpus(num_docs=100)
    
    def test_ngram_frequencies(self, mock_corpus):
        """Test n-gram frequency computation"""
        from linear_a_project import LinguisticAnalyzer
        
        analyzer = LinguisticAnalyzer(mock_corpus)
        bigrams = analyzer.compute_ngram_frequencies(n=2)
        
        assert len(bigrams) > 0
        assert sum(bigrams.values()) > 0
    
    def test_entropy_calculation(self, mock_corpus):
        """Test entropy metrics"""
        from linear_a_project import LinguisticAnalyzer
        
        analyzer = LinguisticAnalyzer(mock_corpus)
        unigram_entropy = analyzer.compute_entropy(n=1)
        bigram_entropy = analyzer.compute_entropy(n=2)
        conditional_entropy = analyzer.conditional_entropy()
        
        assert unigram_entropy > 0
        assert bigram_entropy > unigram_entropy
        assert conditional_entropy > 0
        assert conditional_entropy < unigram_entropy  # Should be lower
    
    def test_zipf_law(self, mock_corpus):
        """Test Zipf's law testing"""
        from linear_a_project import LinguisticAnalyzer
        
        analyzer = LinguisticAnalyzer(mock_corpus)
        zipf_results = analyzer.test_zipf_law()
        
        assert 'slope' in zipf_results
        assert 'r_squared' in zipf_results
        assert 'zipf_compliant' in zipf_results
        assert zipf_results['r_squared'] >= 0  # Valid R²
        assert zipf_results['r_squared'] <= 1
    
    def test_positional_distribution(self, mock_corpus):
        """Test sign position analysis"""
        from linear_a_project import LinguisticAnalyzer
        
        analyzer = LinguisticAnalyzer(mock_corpus)
        positions = analyzer.positional_distribution()
        
        assert len(positions) > 0
        for sign, pos_dist in positions.items():
            # Probabilities should sum to ~1
            total = sum(pos_dist.values())
            assert 0.99 < total < 1.01
    
    def test_mutual_information(self, mock_corpus):
        """Test mutual information matrix"""
        from linear_a_project import LinguisticAnalyzer
        
        analyzer = LinguisticAnalyzer(mock_corpus)
        mi_matrix = analyzer.mutual_information_matrix()
        
        assert mi_matrix.shape[0] == mi_matrix.shape[1]  # Square matrix
        assert mi_matrix.shape[0] > 0  # Non-empty
        # MI values should be non-negative
        assert (mi_matrix.values >= 0).all()

# ============================================================================
# TESTS: COMPUTER VISION
# ============================================================================

class TestComputerVision:
    """Test CV pipeline components"""
    
    def test_sign_classifier_architecture(self):
        """Test sign classifier model"""
        from linear_a_project import SignClassifier
        
        model = SignClassifier(num_classes=90)
        batch = torch.randn(4, 3, 224, 224)  # 4 images
        output = model(batch)
        
        assert output.shape == (4, 90)  # Batch size x num classes
    
    def test_cv_pipeline_structure(self):
        """Test CV pipeline initialization"""
        from linear_a_project import CVPipeline, SignClassifier
        
        # Mock detector
        class MockDetector:
            def detect_signs(self, image_path):
                return [
                    {'bbox': [10, 10, 50, 50], 'confidence': 0.9},
                    {'bbox': [60, 10, 100, 50], 'confidence': 0.85}
                ]
        
        detector = MockDetector()
        classifier = SignClassifier(num_classes=90)
        pipeline = CVPipeline(detector, classifier)
        
        assert pipeline.detector is not None
        assert pipeline.classifier is not None

# ============================================================================
# TESTS: TRANSFER LEARNING
# ============================================================================

class TestTransferLearning:
    """Test transfer learning components"""
    
    def test_multitask_model_architecture(self):
        """Test multitask model structure"""
        from linear_a_project import MultiTaskLinearModel
        
        model = MultiTaskLinearModel(
            vocab_size_a=90,
            vocab_size_b=87,
            embedding_dim=128,
            hidden_dim=256
        )
        
        batch_size = 4
        seq_length = 10
        x = torch.randint(0, 90, (batch_size, seq_length))
        
        output_a = model(x, script='a')
        output_b = model(x, script='b')
        
        assert output_a.shape == (batch_size, seq_length, 90)
        assert output_b.shape == (batch_size, seq_length, 87)
    
    def test_shared_embedding(self):
        """Test shared embedding layer"""
        from linear_a_project import MultiTaskLinearModel
        
        model = MultiTaskLinearModel(
            vocab_size_a=90,
            vocab_size_b=87,
            embedding_dim=128,
            hidden_dim=256
        )
        
        # Same input should produce same embeddings
        x = torch.randint(0, 90, (1, 5))
        
        with torch.no_grad():
            emb1 = model.shared_embedding(x)
            emb2 = model.shared_embedding(x)
        
        assert torch.allclose(emb1, emb2)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test full pipeline integration"""
    
    def test_end_to_end_data_processing(self):
        """Test complete data processing pipeline"""
        from linear_a_project import SignInventory, LinearASign, LinearATokenizer
        
        # Create inventory
        inventory = SignInventory()
        for i in range(10):
            sign = LinearASign(
                sign_id=f"LA{i:03d}",
                unicode=f"LA{i:03d}",
                category="syllabogram",
                frequency=10,
                variants=[],
                linear_b_equivalent=None,
                positions={}
            )
            inventory.add_sign(sign)
        
        # Tokenize
        tokenizer = LinearATokenizer(inventory)
        text = "LA001 LA002 | LA003"
        tokens = tokenizer.tokenize_inscription(text)
        
        # Create mock corpus
        corpus = [{'doc_id': 'TEST_001', 'tokens': tokens, 'source_file': 'test.txt'}]
        
        # Analyze
        from linear_a_project import LinguisticAnalyzer
        analyzer = LinguisticAnalyzer(corpus)
        entropy = analyzer.compute_entropy(1)
        
        assert entropy > 0
    
    def test_project_initialization(self):
        """Test project orchestrator"""
        from linear_a_project import LinearAProject
        
        config = {
            'data_path': 'data/raw',
            'output_path': 'outputs',
            'device': 'cpu'
        }
        
        project = LinearAProject(config)
        assert project.config == config
        assert project.sign_inventory is None  # Not initialized yet

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance benchmarks"""
    
    def test_tokenization_speed(self):
        """Test tokenization performance on large corpus"""
        from linear_a_project import SignInventory, LinearASign, LinearATokenizer
        import time
        
        # Create large inventory
        inventory = SignInventory()
        for i in range(90):
            sign = LinearASign(
                sign_id=f"LA{i:03d}",
                unicode=f"LA{i:03d}",
                category="syllabogram",
                frequency=10,
                variants=[],
                linear_b_equivalent=None,
                positions={}
            )
            inventory.add_sign(sign)
        
        tokenizer = LinearATokenizer(inventory)
        
        # Create large text
        text = " | ".join([f"LA{np.random.randint(0, 90):03d}" for _ in range(1000)])
        
        start = time.time()
        tokens = tokenizer.tokenize_inscription(text)
        duration = time.time() - start
        
        assert duration < 1.0  # Should complete in < 1 second
    
    def test_analysis_memory_efficiency(self):
        """Test memory usage of analysis"""
        import psutil
        import os
        from linear_a_project import LinguisticAnalyzer
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large corpus
        corpus = MockDataGenerator.create_corpus(num_docs=1000, vocab_size=90)
        analyzer = LinguisticAnalyzer(corpus)
        _ = analyzer.compute_ngram_frequencies(n=3)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        assert mem_increase < 500  # Should use < 500 MB

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src", "--cov-report=html"])