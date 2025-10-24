#!/usr/bin/env python3
"""
Simple phonotactic validator stub.
Reads hypotheses JSON and writes validation summary JSON.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypotheses', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    hyp_path = Path(args.hypotheses)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If hypotheses missing, write a placeholder validation
    if not hyp_path.exists():
        print(f"No hypotheses found at {hyp_path}, writing empty validation")
        result = {'mean_score': 0.0, 'valid_percentage': 0.0}
    else:
        hyps = json.loads(hyp_path.read_text())
        # Dummy scoring
        mean_score = sum(h.get('score', 0.5) for h in hyps) / max(1, len(hyps))
        valid_pct = sum(1 for h in hyps if h.get('confidence', 0) > 0.6) / max(1, len(hyps)) * 100
        result = {'mean_score': float(mean_score), 'valid_percentage': float(valid_pct)}
    
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote validation results to {out_path}")

if __name__ == '__main__':
    main()
