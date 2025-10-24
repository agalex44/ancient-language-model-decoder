#!/usr/bin/env python3
"""
Visualization utilities for Linear A analysis
"""
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style('whitegrid')

class Visualizer:
    def __init__(self, output_dir: str = 'outputs/visualizations/'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_entropy_comparison(self, results: dict):
        """Compare entropy across scripts"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Unigram', 'Bigram', 'Conditional']
        linear_a = [results['linear_a']['unigram'], 
                    results['linear_a']['bigram'],
                    results['linear_a']['conditional']]
        linear_b = [results['linear_b']['unigram'],
                    results['linear_b']['bigram'],
                    results['linear_b']['conditional']]
        random = [results['random']['unigram'],
                  results['random']['bigram'],
                  results['random']['conditional']]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, linear_a, width, label='Linear A')
        ax.bar(x, linear_b, width, label='Linear B')
        ax.bar(x + width, random, width, label='Random')
        
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Entropy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'entropy_comparison.png', dpi=300)
        plt.close()
    
    def plot_zipf(self, ranks, frequencies):
        """Plot Zipf's law (log-log)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(ranks, frequencies, 'o', alpha=0.5, markersize=4)
        
        # Fit line
        coeffs = np.polyfit(np.log(ranks), np.log(frequencies), 1)
        fitted = np.exp(coeffs[1]) * ranks ** coeffs[0]
        ax.loglog(ranks, fitted, 'r-', linewidth=2, 
                 label=f'Slope = {coeffs[0]:.2f}')
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
        ax.set_title("Zipf's Law Test")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'zipf_plot.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=False, help='Results dir or file', default='outputs/results/linguistic_analysis.json')
    parser.add_argument('--output', required=False, help='Output dir for plots', default='outputs/visualizations/')
    args = parser.parse_args()
    
    viz = Visualizer(output_dir=args.output)
    
    results_path = Path(args.results)
    if results_path.is_dir():
        results_file = results_path / 'linguistic_analysis.json'
    else:
        results_file = results_path
    
    if results_file.exists():
        data = json.load(open(results_file, 'r'))
        # Build compatible structure for plot calls
        try:
            comp = {
                'linear_a': {
                    'unigram': data['entropy'].get('unigram', 0),
                    'bigram': data['entropy'].get('bigram', 0),
                    'conditional': data['entropy'].get('conditional', 0)
                },
                'linear_b': {
                    'unigram': data.get('linear_b', {}).get('unigram', 0),
                    'bigram': data.get('linear_b', {}).get('bigram', 0),
                    'conditional': data.get('linear_b', {}).get('conditional', 0)
                },
                'random': {
                    'unigram': data.get('random', {}).get('unigram', 0),
                    'bigram': data.get('random', {}).get('bigram', 0),
                    'conditional': data.get('random', {}).get('conditional', 0)
                }
            }
            viz.plot_entropy_comparison(comp)
        except Exception:
            print("Entropy comparison skipped (unexpected format)")
        
        # Zipf plot if ngram frequencies exist
        try:
            freqs = data['ngrams']['unigrams']
            freqlist = sorted([v for v in freqs.values()], reverse=True)
            ranks = list(range(1, len(freqlist)+1))
            viz.plot_zipf(ranks, freqlist)
        except Exception:
            print("Zipf plot skipped (missing ngram frequencies)")
    
    else:
        print(f"No results file found at {results_file}")
    print("Visualization module complete")