#!/usr/bin/env python3
"""
Visualization utilities for Linear A analysis
"""
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
    print("Visualization module loaded")