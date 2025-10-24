#!/usr/bin/env python3
"""
Stub trainer for ResNet sign classifier.
Produces outputs/models/sign_classifier/best.pt as placeholder.
"""
import argparse
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / 'best.pt'
    print(f"Training classifier (stub) epochs={args.epochs} device={args.device}")
    time.sleep(0.5)
    model_file.write_text("stub-resnet-model")
    print(f"Saved dummy classifier to {model_file}")

if __name__ == '__main__':
    main()
