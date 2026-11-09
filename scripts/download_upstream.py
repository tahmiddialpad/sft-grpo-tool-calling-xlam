#!/usr/bin/env python3
"""Acquire the licensed upstream snapshot into ignored outputs/ and verify its hash."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/upstream/xlam_train.jsonl')
    args=parser.parse_args()
    info=json.loads((ROOT/'data/upstream/dataset_info.json').read_text())
    path=hf_hub_download(info['repo'],'xlam_function_calling_60k.json',repo_type='dataset',revision=info['revision'])
    rows=json.loads(Path(path).read_text())
    if not isinstance(rows,list) or len(rows)!=60000:raise ValueError('Unexpected upstream dataset structure')
    order=json.loads((ROOT/'metadata/raw_serialization.json').read_text())['column_order']
    content=''.join(json.dumps({k:row[k] for k in order},ensure_ascii=False)+'\n' for row in rows).encode()
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    expected=next(r for r in manifest['files'] if r['profile']=='upstream')['source_unredacted_sha256']
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError('Current upstream snapshot differs from the original recorded hash; no training input was replaced')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(content)
    print('Verified upstream snapshot saved locally in ignored outputs/.')
