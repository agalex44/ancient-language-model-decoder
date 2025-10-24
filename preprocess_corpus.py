#!/usr/bin/env python3
"""
Preprocess Linear A corpus from raw format to tokenized JSON
"""
import argparse
import json
from pathlib import Path
from linear_a_project import SignInventory, LinearATokenizer

def parse_sigla_format(filepath: Path):
    """Parse SigLA database export format"""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Format: HT_001: da-ro pa-i-to | ku-ro
            parts = line.split(':', 1)
            if len(parts) == 2:
                doc_id = parts[0].strip()
                text = parts[1].strip()
                documents.append({
                    'doc_id': doc_id,
                    'text': text
                })
    return documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Raw corpus file')
    parser.add_argument('--sign-inventory', required=True, help='Sign inventory JSON')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--split-ratios', default='0.8,0.1,0.1', help='Train/val/test split')
    args = parser.parse_args()

    # Load sign inventory
    inventory = SignInventory.load(args.sign_inventory)
    tokenizer = LinearATokenizer(inventory)

    # Parse and tokenize
    documents = parse_sigla_format(Path(args.input))
    tokenized = []

    for doc in documents:
        tokens = tokenizer.tokenize_inscription(doc['text'])
        tokenized.append({
            'doc_id': doc['doc_id'],
            'tokens': tokens,
            'raw_text': doc['text']
        })

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create train/val/test splits
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('split-ratios must be three comma-separated numbers that sum to 1.0')

    import random
    random.shuffle(tokenized)
    n = len(tokenized)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = tokenized[:n_train]
    val = tokenized[n_train:n_train + n_val]
    test = tokenized[n_train + n_val:]

    with open(output_path / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(output_path / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    with open(output_path / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    # Also keep legacy tokenized_corpus.json for compatibility
    with open(output_path / 'tokenized_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(tokenized, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(tokenized)} documents: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == '__main__':
    main()