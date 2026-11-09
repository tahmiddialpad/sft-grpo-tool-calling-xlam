#!/usr/bin/env python3
"""Launch an xLAM-only verl recipe; an explicit model is required for the hybrid."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=('original','revision_v2'))
    parser.add_argument('--scale',required=True,choices=('0_6b','1_7b','4b','8b','14b','32b'))
    parser.add_argument('--stage',choices=('grpo','sftgrpo_grpo'),default='grpo')
    parser.add_argument('--gpu-ids',default='0,1,2,3')
    parser.add_argument('--model',help='Merged SFT checkpoint for hybrid; optional base override for GRPO')
    parser.add_argument('--allow-redacted-data',action='store_true')
    args,extra=parser.parse_known_args();ids=args.gpu_ids.split(',')
    if len(ids)!=len(set(ids)) or not all(i.isdecimal() for i in ids):parser.error('GPU IDs must be distinct nonnegative integers')
    if args.stage=='sftgrpo_grpo' and not args.model:parser.error('--model must name the merged SFT checkpoint for hybrid training')
    if not args.allow_redacted_data and not (ROOT/'datasets'/args.profile/'RESTORED_FROM_UPSTREAM.json').exists():
        parser.error('Restore exact training inputs first, or explicitly use --allow-redacted-data for a different experiment')
    script=ROOT/'configs'/args.profile/'verl'/f'{args.scale}_{args.stage}.sh'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(ids),NPROC_PER_NODE=str(len(ids)),N_GPUS=str(len(ids)))
    if args.model:env['MODEL']=args.model
    if args.profile=='revision_v2':
        import json
        from huggingface_hub import snapshot_download
        model='Qwen/Qwen3-'+args.scale.replace('_','.').replace('b','B')
        revision=json.loads((ROOT/'metadata/model_revisions.json').read_text())[model]
        base=snapshot_download(model,revision=revision)
        if not args.model:env['MODEL']=base
        tokenizer=ROOT/'datasets/revision_v2/tokenizer'
        subprocess.run([__import__('sys').executable,str(ROOT/'scripts/prepare_tokenizer.py'),
            '--model',base,'--out',str(tokenizer)],check=True)
        env['REVISION_TOKENIZER']=str(tokenizer)
    subprocess.run(['bash',str(script),*extra],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
