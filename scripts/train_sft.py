#!/usr/bin/env python3
"""Launch portable LoRA SFT, preserving effective batch 64 on four GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=('original','revision_v2'))
    parser.add_argument('--scale',required=True,choices=('0_6b','1_7b','4b','8b','14b','32b'))
    parser.add_argument('--stage',choices=('sft','sftgrpo_sft'),default='sft')
    parser.add_argument('--gpu-ids',default='0,1,2,3')
    parser.add_argument('--allow-redacted-data',action='store_true',help='Explicitly train a new experiment on public redacted prompts instead of restoring exact study inputs')
    args=parser.parse_args();ids=args.gpu_ids.split(',')
    if len(ids)!=len(set(ids)) or not all(i.isdecimal() for i in ids):parser.error('GPU IDs must be distinct nonnegative integers')
    path=ROOT/'configs'/args.profile/'llamafactory'/f'{args.scale}_{args.stage}.yaml'
    config=yaml.safe_load(path.read_text());effective=64;divisor=len(ids)*config['per_device_train_batch_size']
    if effective%divisor:parser.error('GPU count must preserve effective batch 64')
    config['gradient_accumulation_steps']=effective//divisor
    config['dataset_dir']=str(ROOT/'datasets'/args.profile/'llamafactory')
    output=ROOT/'outputs'/args.profile/f'{args.scale}_{args.stage}'
    config['output_dir']=str(output)
    work=ROOT/'outputs/launch_configs';work.mkdir(parents=True,exist_ok=True)
    resolved=work/f'{args.profile}_{args.scale}_{args.stage}.yaml';resolved.write_text(yaml.safe_dump(config,sort_keys=False))
    if not Path(config['dataset_dir'],'dataset_info.json').exists():parser.error('Run scripts/prepare_data.py for this profile first')
    if not args.allow_redacted_data and not (ROOT/'datasets'/args.profile/'RESTORED_FROM_UPSTREAM.json').exists():
        parser.error('Restore exact training inputs with scripts/restore_training_inputs.py, or explicitly use --allow-redacted-data for a different experiment')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(ids),NPROC_PER_NODE=str(len(ids)),FORCE_TORCHRUN='1')
    output.mkdir(parents=True,exist_ok=True)
    (output/'launch_metadata.json').write_text(json.dumps({'profile':args.profile,'gpu_ids':ids,'effective_batch':64,'resolved_config':config},indent=2)+'\n')
    subprocess.run(['llamafactory-cli','train',str(resolved)],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
