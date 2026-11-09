#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint (from LLaMA-Factory SFT, finetuning_type=lora)
into its base model, saving a full HF checkpoint verl/vLLM can load directly.

No-op (just copies through) if the given checkpoint isn't a LoRA adapter
(no adapter_config.json) — safe to call unconditionally.

Usage: python3 merge_lora_checkpoint.py --adapter-dir DIR --base-model HF_ID --out-dir DIR
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if os.path.abspath(args.out_dir) == os.path.abspath(args.adapter_dir):
        raise ValueError("Output must differ from the adapter directory")
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise ValueError("Choose a new empty output directory; existing files are never removed")

    if not os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json")):
        print(f"[merge_lora] {args.adapter_dir} is not a LoRA adapter — using as-is")
        if os.path.abspath(args.adapter_dir) != os.path.abspath(args.out_dir):
            shutil.copytree(args.adapter_dir, args.out_dir, dirs_exist_ok=True)
        return

    import warnings

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[merge_lora] loading base model {args.base_model}", flush=True)
    # Load via the EXACT class named in the checkpoint config, not
    # AutoModelForCausalLM: the adapter's saved keys embed the training-time
    # module paths, and the auto-mapping can resolve to a different wrapper
    # with different paths. Bit us with Qwen3.5: LLaMA-Factory trained against
    # Qwen3_5ForConditionalGeneration (layers under model.language_model.*),
    # but AutoModelForCausalLM resolves to the text-only variant (model.*) —
    # peft then silently matches ZERO adapter keys and "merges" a no-op,
    # producing a checkpoint identical to the untrained base.
    cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    cls = getattr(transformers, cfg.architectures[0], None) if cfg.architectures else None
    if cls is not None:
        print(f"[merge_lora] using {cfg.architectures[0]}", flush=True)
        base = cls.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[merge_lora] applying adapter from {args.adapter_dir}", flush=True)
    # peft only WARNS when adapter keys don't match the model's module paths —
    # escalate that to a hard failure, a partial/no-op merge must never be
    # written out as if it succeeded.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    for w in caught:
        if "missing adapter keys" in str(w.message).lower():
            raise RuntimeError(
                f"adapter keys did not match the base model's module paths — "
                f"refusing to write a no-op merge: {str(w.message)[:500]}")
    merged = peft_model.merge_and_unload()

    # Force a clean directory before saving: if out_dir pre-existed with a
    # stale model.safetensors.index.json + shards from an earlier/unrelated
    # run, save_pretrained() here can write a fresh single-file
    # model.safetensors alongside them WITHOUT removing the old index —
    # loaders (vLLM included) then follow the stale index and silently load
    # the OLD unmerged weights while the correct merge sits unused next to it.
    # Bit us for real: sft_qwen35-9b_{xlam,toolace,glaive} eval'd 3 checkpoints
    # that scored identically to zero-shot because of exactly this.
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    # Sanity check: if an index.json exists, every shard it references must
    # both exist AND have just been written (not a leftover from some other
    # source) — catches this exact failure mode if it recurs for any reason.
    index_path = os.path.join(args.out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            expected_shards = set(json.load(f)["weight_map"].values())
        on_disk = set(os.listdir(args.out_dir))
        missing = expected_shards - on_disk
        if missing:
            raise RuntimeError(f"save_pretrained wrote an index referencing missing "
                                f"shards {missing} in {args.out_dir} — aborting")
    # Take tokenizer files from the BASE model's hub snapshot, not the adapter
    # dir and not an AutoTokenizer round-trip. Both alternatives break older
    # loaders (e.g. vLLM 0.17's bundled transformers 4.57): anything saved by
    # transformers 5.x — including LLaMA-Factory's training-time copy in the
    # adapter dir — stamps the 5.x-only tokenizer_class "TokenizersBackend"
    # into tokenizer_config.json, which they reject. SFT doesn't change the
    # vocab (verified: adapter and hub tokenizers differ only by 7 unused
    # audio/TTS added-tokens that transformers 5.x injects for Qwen3.5), so
    # the hub files are correct AND give exact tokenization parity with
    # zero-shot eval of the same base model. NOTE: would be wrong if training
    # ever added real new tokens.
    TOK_FILES = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
        "chat_template.jinja", "generation_config.json",
        # vLLM treats ForConditionalGeneration architectures as multimodal and
        # refuses to start without the (pre)processor configs.
        "preprocessor_config.json", "video_preprocessor_config.json",
        "processor_config.json",
    ]
    from huggingface_hub import snapshot_download
    # allow_patterns only limits what gets newly DOWNLOADED — if the base
    # model is already fully cached (as it always is here), snapshot_download
    # returns the path to the COMPLETE snapshot directory regardless, which
    # also contains every multi-GB weight shard. An earlier version of this
    # script did `for name in os.listdir(tok_src)`, which copied everything it
    # found there — silently duplicating the entire untrained base model on
    # top of the real merge output (and, since the copies used the base
    # model's original shard filenames + a stale index.json, loaders picked
    # THOSE up instead of the actual merge). Only ever copy the exact
    # filenames listed above, never enumerate the directory.
    tok_src = (args.base_model if os.path.isdir(args.base_model)
               else snapshot_download(args.base_model, allow_patterns=TOK_FILES))
    for name in TOK_FILES:
        path = os.path.join(tok_src, name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(args.out_dir, name))
    print(f"[merge_lora] merged checkpoint saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
