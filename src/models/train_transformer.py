#!/usr/bin/env python3
"""
Stub transformer trainer. Produces outputs/models/transformer/model.pt
"""
import argparse
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / 'model.pt'
    print(f"Training Transformer (stub) epochs={args.epochs} device={args.device}")
    time.sleep(0.5)
    model_file.write_text("stub-transformer")
    print(f"Saved stub transformer to {model_file}")

if __name__ == '__main__':
    main()
