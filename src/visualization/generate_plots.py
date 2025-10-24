#!/usr/bin/env python3
"""
Wrapper to call root Visualizer or internal plotting utilities.
Keeps README examples and pipeline consistent.
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    # Prefer root visualizations.py if available
    viz_script = Path('visualizations.py')
    if viz_script.exists():
        cmd = [sys.executable, str(viz_script), '--results', args.results, '--output', args.output]
        import subprocess
        subprocess.run(cmd, check=True)
    else:
        print("No visualizations.py found at repo root", flush=True)

if __name__ == '__main__':
    main()
