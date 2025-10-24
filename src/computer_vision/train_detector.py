#!/usr/bin/env python3
"""
Stub trainer for sign detector (YOLOv8 wrapper)
This stub creates a dummy model file so pipeline proceeds.
Replace with full training implementation later.
"""
import argparse
from pathlib import Path
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dummy_model = out / 'best.pt'
    
    # Simulate training
    print(f"Training detector (stub) epochs={args.epochs} device={args.device}")
    time.sleep(0.5)
    dummy_model.write_text("stub-yolov8-model")
    print(f"Saved dummy detector to {dummy_model}")

if __name__ == '__main__':
    main()
