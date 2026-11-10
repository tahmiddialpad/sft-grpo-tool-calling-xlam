#!/usr/bin/env python3
"""Unpack a frozen xLAM profile and verify every decompressed SHA256."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def prepare(profile,out=None):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    destination=Path(out) if out else ROOT/'datasets'/profile
    entries=[r for r in manifest['files'] if r['profile']==profile]
    if not entries:raise ValueError('Unknown or empty profile: '+profile)
    for entry in entries:
        source=ROOT/entry['path'];target=destination/entry['prepared_path']
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.tmp')
        if source.suffix=='.gz':
            with gzip.open(source,'rb') as inp,temporary.open('wb') as outp:shutil.copyfileobj(inp,outp)
        else:shutil.copy2(source,temporary)
        digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest!=entry['uncompressed_sha256']:
            temporary.unlink();raise ValueError('Checksum mismatch: '+str(source))
        temporary.replace(target)
    print(f'Prepared {len(entries)} verified files in {destination}')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',default='original',choices=('original',))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();prepare(args.profile,args.out)
