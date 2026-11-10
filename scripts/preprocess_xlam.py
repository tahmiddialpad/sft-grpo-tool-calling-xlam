#!/usr/bin/env python3
"""Reconstruct the historical xLAM unified splits from the released raw snapshot.

Uses the original normalized-query-plus-called-names deduplication and seed 42.
Frozen training exports additionally remove the five released training IDs.

"""
import argparse
import gzip
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parents[1]


def convert(row,index):
    tools=json.loads(row['tools']) if isinstance(row['tools'],str) else row['tools']
    answers=json.loads(row['answers']) if isinstance(row['answers'],str) else row['answers']
    calls=[{'name':a['name'],'arguments':a['arguments'] if isinstance(a.get('arguments'),dict) else {}} for a in answers]
    return {'id':f'xlam-{index:06d}','source':'xlam','tools':tools,
        'messages':[{'role':'user','content':row['query']},{'role':'assistant','content':None,'tool_calls':calls}],
        'n_turns':1,'n_tools_available':len(tools),'n_calls':len(calls),
        'parallel':len(calls)>1,'multi_turn':False,'irrelevant':len(calls)==0}


def reconstruct(raw):
    opener=gzip.open if str(raw).endswith('.gz') else open
    with opener(raw,'rt') as stream:rows=[json.loads(line) for line in stream if line.strip()]
    kept=[];seen=set()
    for index,row in enumerate(rows):
        record=convert(row,index)
        key=(re.sub(r'[^a-z0-9]+',' ',record['messages'][0]['content'].lower()).strip(),
             tuple(sorted(c['name'] for c in record['messages'][1]['tool_calls'])))
        if key in seen:continue
        seen.add(key);kept.append(record)
    random.Random(42).shuffle(kept)
    return {'test':kept[:1000],'val':kept[1000:1500],'train':kept[1500:]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/upstream/xlam_train.jsonl.gz')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/rebuilt_original_splits')
    args=parser.parse_args();splits=reconstruct(args.raw);args.out.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():
        (args.out/f'xlam_{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    print({name:len(rows) for name,rows in splits.items()})
