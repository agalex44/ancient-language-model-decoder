#!/usr/bin/env python3
"""
Stub HMM trainer. Writes a marker file to outputs/models/hmm/
"""
import argparse
from pathlib import Path
import time
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--components', type=int, default=50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_meta = out / 'hmm_meta.json'
    print(f"Training HMM (stub) on {args.corpus} components={args.components}")
    time.sleep(0.5)
    model_meta.write_text(json.dumps({'components': args.components}))
    print(f"Saved stub HMM metadata to {model_meta}")

if __name__ == '__main__':
    main()
