#!/usr/bin/env python3
"""
Simple runner that uses LinearA framework to produce linguistic_analysis.json
"""
import argparse
import json
from pathlib import Path
from linear_a_project import LinguisticAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    corpus = json.loads(Path(args.corpus).read_text())
    analyzer = LinguisticAnalyzer(corpus)
    
    results = {
        'entropy': {
            'unigram': float(analyzer.compute_entropy(1)),
            'bigram': float(analyzer.compute_entropy(2)),
            'trigram': float(analyzer.compute_entropy(3)),
            'conditional': float(analyzer.conditional_entropy())
        },
        'zipf': analyzer.test_zipf_law(),
        'positions': analyzer.positional_distribution(),
        'ngrams': {
            'unigrams': {str(k): v for k, v in analyzer.compute_ngram_frequencies(1).most_common(50)},
            'bigrams': {str(k): v for k, v in analyzer.compute_ngram_frequencies(2).most_common(50)},
            'trigrams': {str(k): v for k, v in analyzer.compute_ngram_frequencies(3).most_common(50)}
        }
    }
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) / 'linguistic_analysis.json'
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_file}")

if __name__ == '__main__':
    main()
