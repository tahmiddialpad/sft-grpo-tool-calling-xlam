#!/usr/bin/env python3
"""CPU verification of frozen data, split IDs, and all released result scores."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_saved import evaluate,read
from preprocess_xlam import reconstruct


def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for entry in manifest['files']:
        path=ROOT/entry['path'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['compressed_sha256'],path
        payload=gzip.decompress(data) if path.suffix=='.gz' else data
        assert hashlib.sha256(payload).hexdigest()==entry['uncompressed_sha256'],path
        assert len(payload)==entry['uncompressed_bytes'],path
    expected={'original':{'train':56438,'val':500,'test':1000},'revision_v2':{'train':10752,'val':500,'test':1000}}
    for profile,counts in expected.items():
        ids={}
        for split,n in counts.items():
            rows=read(ROOT/'data'/profile/'grpo'/f'xlam_{split}.jsonl.gz')
            assert len(rows)==n
            ids[split]={r['id'] for r in rows}
            assert len(ids[split])==n and all(r['id'].startswith('xlam-') and r['gold_calls'] for r in rows)
            assert all(r.get('source','xlam')=='xlam' for r in rows)
        assert not ids['train']&ids['test'] and not ids['train']&ids['val'] and not ids['test']&ids['val']
        if profile=='revision_v2':
            a=read(ROOT/'data/revision_v2/grpo/xlam_sftgrpo_sft.jsonl.gz')
            b=read(ROOT/'data/revision_v2/grpo/xlam_sftgrpo_grpo.jsonl.gz')
            assert len(a)==len(b)==5376
            assert not {r['id'] for r in a}&{r['id'] for r in b}
            assert {r['id'] for r in a+b}==ids['train']
    rebuilt=reconstruct(ROOT/'data/upstream/xlam_train.jsonl.gz')
    for split,rows in rebuilt.items():
        frozen=read(ROOT/'data/original/processed'/f'xlam_{split}.jsonl.gz')
        assert rows==frozen,('historical normalization mismatch',split)
    excluded=set(json.loads((ROOT/'metadata/original_training_exclusions.json').read_text()))
    assert len(excluded)==5
    train=read(ROOT/'data/original/grpo/xlam_train.jsonl.gz')
    assert {r['id'][:-3] for r in train}=={r['id'] for r in rebuilt['train'] if r['id'] not in excluded}
    metrics=list(csv.DictReader((ROOT/'results/metrics.csv').open()))
    for row in metrics:
        path=ROOT/row['predictions']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['predictions_sha256']
        result=evaluate(path,ROOT/'data'/row['profile']/'grpo/xlam_test.jsonl.gz',row['profile'])
        if abs(result['exact_match']-float(row['saved_exact_match']))>1e-10:
            raise ValueError('Saved score reproduction failed: '+str(path)+': '+str(result))
        assert result['n']==int(row['n'])==1000
    for path in (ROOT/'configs').rglob('*.sh'):
        import subprocess
        subprocess.run(['bash','-n',str(path)],check=True)
        assert '/app/tct' not in path.read_text()
    print(json.dumps({'data_files':len(manifest['files']),'verified_evaluations':len(metrics),
        'historical_normalization':'exact on public redacted snapshot','split_overlap':0,'status':'pass'},indent=2))


if __name__=='__main__':main()
