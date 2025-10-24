#!/usr/bin/env python3
"""
Lightweight data validation script used by pipeline.
Checks that tokenized directory contains at least one JSON and some structure.
"""
import argparse
from pathlib import Path
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, help='Path to tokenized corpus directory')
    args = parser.parse_args()
    
    p = Path(args.corpus)
    if not p.exists():
        print(f"Corpus path {p} not found", file=sys.stderr)
        sys.exit(1)
    
    json_files = list(p.glob('*.json'))
    if not json_files:
        print("No JSON files found in tokenized corpus directory", file=sys.stderr)
        sys.exit(1)
    
    # quick sanity check of first file
    try:
        data = json.loads(json_files[0].read_text())
        if not isinstance(data, list):
            print("Tokenized file format unexpected (expected list)", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to read {json_files[0]}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Data validation passed")
    sys.exit(0)

if __name__ == '__main__':
    main()
