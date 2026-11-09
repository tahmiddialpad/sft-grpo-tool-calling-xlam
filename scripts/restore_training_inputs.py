#!/usr/bin/env python3
"""Locally restore credential-redacted training inputs from an exact upstream snapshot.

No credential is tested or sent to a service. Restored files stay in ignored
datasets/. Upstream material must be acquired under its own access/license terms.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def rows_digest(rows):
    digest=hashlib.sha256()
    for row in rows:digest.update((json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=('original','revision_v2'))
    parser.add_argument('--upstream-jsonl',required=True,type=Path)
    parser.add_argument('--prepared-dir',type=Path)
    args=parser.parse_args();manifest=json.loads((ROOT/'data/manifest.json').read_text())
    raw=args.upstream_jsonl.read_bytes()
    if args.upstream_jsonl.suffix=='.gz':raw=gzip.decompress(raw)
    source=next(r for r in manifest['files'] if r['profile']=='upstream')
    if hashlib.sha256(raw).hexdigest()!=source['source_unredacted_sha256']:
        parser.error('Upstream snapshot does not match the original recorded SHA256')
    redaction=manifest['public_redaction'];placeholder=redaction['placeholder']
    candidates=set(re.findall(rb'gh[pousr]_[A-Za-z0-9]{30,}',raw))
    matching=[v for v in candidates if hashlib.sha256(v).hexdigest()==redaction['source_value_sha256']]
    if len(matching)!=1:parser.error('Expected upstream value hash not found')
    original=matching[0].decode();destination=args.prepared_dir or ROOT/'datasets'/args.profile
    def restore(value):
        if isinstance(value,str):return value.replace(placeholder,original)
        if isinstance(value,list):return [restore(v) for v in value]
        if isinstance(value,dict):return {k:restore(v) for k,v in value.items()}
        return value
    restored=0
    for entry in manifest['files']:
        if entry['profile']!=args.profile or not entry.get('public_redaction'):continue
        path=destination/entry['prepared_path']
        if not path.exists():parser.error('Prepare this profile first: '+str(path))
        if path.suffix=='.parquet':
            import pyarrow as pa
            import pyarrow.parquet as pq
            table=pq.read_table(path);rows=[restore(row) for row in table.to_pylist()]
            if rows_digest(rows)!=entry['source_parquet_rows_sha256']:raise ValueError('Restored parquet content mismatch')
            temporary=path.with_name(path.name+'.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),temporary)
        else:
            data=path.read_bytes().replace(placeholder.encode(),original.encode())
            if hashlib.sha256(data).hexdigest()!=entry['source_unredacted_sha256']:raise ValueError('Restored bytes mismatch: '+str(path))
            temporary=path.with_name(path.name+'.tmp');temporary.write_bytes(data)
        temporary.replace(path);restored+=1
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'RESTORED_FROM_UPSTREAM.json').write_text(json.dumps({'profile':args.profile,'restored_files':restored,
        'upstream_sha256':hashlib.sha256(raw).hexdigest(),'note':'Parquet table content verified; serialization bytes may differ.'},indent=2)+'\n')
    print('Locally restored and verified',restored,'training files. No credential was contacted or displayed.')


if __name__=='__main__':main()
