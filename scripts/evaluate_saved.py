#!/usr/bin/env python3
"""Recompute scores from released generations, without a GPU or model download."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import legacy_reward
import grpo_reward


def read(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:return [json.loads(line) for line in stream if line.strip()]


def score(text,gold,profile):
    if profile=='original':
        calls=legacy_reward.parse_calls(text)
        name,arg,exact=legacy_reward.score_calls(calls,gold) if calls is not None else (0.0,0.0,False)
        return dict(exact=int(exact),name=name,arg=arg,parsed=int(calls is not None),pred=calls,
                    valid=int(calls is not None),status='calls' if calls is not None else 'invalid',
                    literal_exact=int(calls is not None and grpo_reward.literal_exact(calls,gold)))
    return grpo_reward.evaluate_output(text,gold)


def evaluate(predictions,references,profile,reparse_original=False):
    rows=read(predictions);refs=read(references);gold={r['id']:r['gold_calls'] for r in refs}
    if len(gold)!=len(refs) or len(rows)!=len(refs) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(gold):
        raise ValueError('Predictions must contain exactly the reference IDs, once each')
    scores=[]
    for row in rows:
        if profile=='original' and not reparse_original:
            if 'pred' not in row:raise ValueError('Historical reproduction requires the released parsed-call field; use --reparse-original for raw parsing')
            calls=row['pred']
            name,arg,exact=legacy_reward.score_calls(calls,gold[row['id']]) if calls is not None else (0.0,0.0,False)
            scores.append(dict(exact=int(exact),name=name,arg=arg,parsed=int(calls is not None)))
        else:scores.append(score(row['raw'],gold[row['id']],profile))
    return dict(profile=profile,n=len(rows),exact_match=sum(r['exact'] for r in scores)/len(rows),
        name_acc=sum(r['name'] for r in scores)/len(rows),arg_score=sum(r['arg'] for r in scores)/len(rows),
        parse_rate=sum(r['parsed'] for r in scores)/len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=('original','revision_v2'))
    parser.add_argument('--predictions',required=True,type=Path)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--reparse-original',action='store_true',help='Use the available archival raw parser instead of the authoritative saved historical parse')
    args=parser.parse_args()
    reference=args.references or ROOT/'data'/args.profile/'grpo/xlam_test.jsonl.gz'
    print(json.dumps(evaluate(args.predictions,reference,args.profile,args.reparse_original),indent=2))
