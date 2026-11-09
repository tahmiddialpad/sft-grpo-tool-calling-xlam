#!/usr/bin/env python3
"""Freeze the corrected tokenizer template and verify released xLAM checks."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template=(ROOT/'metadata/revision_chat_template.jinja').read_text()
    checks=json.loads((ROOT/'metadata/tokenizer_checks.json').read_text())
    for case in checks['cases']:
        actual=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hashlib.sha256(json.dumps(actual).encode()).hexdigest()!=case['prompt_ids_sha256']:
            raise ValueError('Tokenizer mismatch: '+case['id'])
    args.out.mkdir(parents=True,exist_ok=True);tokenizer.save_pretrained(args.out)
    print('Verified',len(checks['cases']),'xLAM tokenizer cases')
