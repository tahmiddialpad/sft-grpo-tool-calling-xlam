#!/usr/bin/env python3
"""Verify model tokenizer and actual SFT prompt encoding before a GPU run."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--checks',required=True,type=Path)
    p.add_argument('--chat-template',required=True,type=Path)
    p.add_argument('--revision',help='Immutable base-model commit')
    p.add_argument('--factory',action='store_true');a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,revision=a.revision)
    tok.chat_template=a.chat_template.read_text()
    cases=json.loads(a.checks.read_text())['cases']
    for case in cases:
        ids=tok.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,
                                   enable_thinking=False,return_dict=False)
        if digest(ids)!=case['prompt_ids_sha256']:
            raise RuntimeError('Tokenizer/rendering mismatch: '+case['id'])
    if a.factory:
        from llamafactory.hparams import DataArguments
        from llamafactory.data.template import get_template_and_fix_tokenizer
        template=get_template_and_fix_tokenizer(tok,DataArguments(template='qwen3',enable_thinking=False))
        for case in cases:
            messages=[{'role':'observation' if m['role']=='tool' else m['role'],'content':m['content']}
                      for m in case['messages'][1:]]
            messages.append({'role':'assistant','content':case['target']})
            ids,response=template.encode_oneturn(tok,messages,system=case['messages'][0]['content'])
            if digest(ids)!=case['prompt_ids_sha256']:
                raise RuntimeError('SFT versus GRPO/eval prompt mismatch: '+case['id'])
            decoded=tok.decode(response,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            if decoded!=case['target'].strip():
                raise RuntimeError('SFT target altered during tokenization: '+case['id'])
    print(json.dumps({'preflight':'pass','model':a.model,'cases':len(cases),'factory':a.factory}),flush=True)


if __name__=='__main__':main()
