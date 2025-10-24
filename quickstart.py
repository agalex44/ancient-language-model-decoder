#!/usr/bin/env python3
"""
Quick demonstration using synthetic data
"""
from linear_a_project import *
import numpy as np

def generate_synthetic_corpus(num_docs=100):
    """Create fake Linear A corpus"""
    corpus = []
    for i in range(num_docs):
        num_words = np.random.randint(3, 10)
        tokens = []
        for _ in range(num_words):
            word_len = np.random.randint(2, 6)
            word_tokens = []
            for _ in range(word_len):
                sign_id = f"LA{np.random.randint(0, 90):03d}"
                word_tokens.append({
                    'sign': f'sign_{sign_id}',
                    'sign_id': sign_id,
                    'uncertain': False
                })
            tokens.append(word_tokens)
        
        corpus.append({
            'doc_id': f'SYNTH_{i:04d}',
            'tokens': tokens,
            'source_file': 'synthetic'
        })
    return corpus

def main():
    print("=" * 60)
    print("LINEAR A DECIPHERMENT - QUICKSTART DEMO")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1/3] Generating synthetic corpus...")
    corpus = generate_synthetic_corpus(num_docs=200)
    print(f"  ✓ Created {len(corpus)} synthetic documents")
    
    # Run analysis
    print("\n[2/3] Running linguistic analysis...")
    analyzer = LinguisticAnalyzer(corpus)
    
    uni_entropy = analyzer.compute_entropy(1)
    cond_entropy = analyzer.conditional_entropy()
    zipf = analyzer.test_zipf_law()
    
    print(f"  ✓ Unigram entropy: {uni_entropy:.3f} bits")
    print(f"  ✓ Conditional entropy: {cond_entropy:.3f} bits")
    print(f"  ✓ Zipf compliance: {zipf['zipf_compliant']}")
    print(f"  ✓ Zipf R²: {zipf['r_squared']:.3f}")
    
    # Save report
    print("\n[3/3] Saving results...")
    report = f"""
QUICKSTART REPORT
=================

Corpus Statistics:
- Documents: {len(corpus)}
- Average words per document: {np.mean([len(d['tokens']) for d in corpus]):.1f}

Linguistic Metrics:
- Unigram Entropy: {uni_entropy:.3f} bits
- Conditional Entropy: {cond_entropy:.3f} bits
- Zipf Slope: {zipf['slope']:.3f}
- Zipf R²: {zipf['r_squared']:.3f}
- Language-like: {'YES' if zipf['zipf_compliant'] else 'NO'}

Next Steps:
1. Download real Linear A corpus from SigLA
2. Run preprocess_corpus.py on real data
3. Train computer vision models on tablet images
4. Run Bayesian decipherment
"""
    
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/QUICKSTART_REPORT.txt', 'w') as f:
        f.write(report)
    
    print("  ✓ Report saved to outputs/QUICKSTART_REPORT.txt")
    print("\n" + "=" * 60)
    print("Demo complete! Check outputs/ directory.")
    print("=" * 60)

if __name__ == '__main__':
    main()