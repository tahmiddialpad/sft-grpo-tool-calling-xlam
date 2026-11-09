#!/usr/bin/env python3
"""Held-out test-split evaluation with vLLM, data-parallel across 8 A100s.

Spawns N workers (N = gpus/tp), each pinning its own GPU(s) via
CUDA_VISIBLE_DEVICES, loading the model with vLLM offline inference, and
processing a shard of the test set. Small models (<=9B): tp=1 -> 8 replicas.

  python3 evals/scripts/eval_transfer_matrix.py \
      --model results/sft/qwen3.5-4b_xlam \
      --tests xlam \
      --gpus 8 --tp 1 --out outputs/eval

Scores each example against gold calls (exact match, name acc, arg score) using
the same matcher as GRPO training (scripts/grpo_reward.py) — reward/metric parity.
Writes results/eval/<run>/<test>_metrics.json + per-example predictions.
"""
import argparse, hashlib, json, multiprocessing as mp, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate_saved import score  # noqa: E402
from prompts import render_system  # noqa: E402  (same prompt as SFT & GRPO)


def load_test(src, data_dir=None):
    path = os.path.join(data_dir or os.path.join(ROOT, "datasets", "grpo"), f"{src}_test.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def worker(rank, args, shards, out_q):
    physical_ids = list(map(int,args.gpu_ids.split(','))) if args.gpu_ids else list(range(args.gpus))
    gpu_ids = ",".join(str(g) for g in physical_ids[rank * args.tp:(rank + 1) * args.tp])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # vLLM rejects a LoRA adapter whose rank exceeds max_lora_rank, so read the
    # actual rank out of the adapter's own config instead of assuming 32 (breaks
    # the r=64 rank-ablation checkpoints, which use lora_rank=64).
    max_lora_rank = 32
    if args.lora:
        with open(os.path.join(args.lora, "adapter_config.json")) as f:
            max_lora_rank = json.load(f)["r"]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              seed=args.seed, generation_config="vllm",
              revision=args.base_revision if args.lora or args.model.startswith('Qwen/') else None,
              tokenizer_revision=args.base_revision if args.lora or args.model.startswith('Qwen/') else None,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len,
              enable_lora=bool(args.lora), max_lora_rank=max_lora_rank, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
        revision=args.base_revision if args.lora or args.model.startswith('Qwen/') else None)
    if args.chat_template:
        with open(args.chat_template) as stream:
            tok.chat_template = stream.read()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, args.lora)

    results = {}
    for src, examples in shards.items():
        prompts = []
        for ex in examples:
            msgs = [{"role": "system", "content": render_system(ex["tools"])}]
            msgs += ex["prompt"]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=(args.thinking == "enabled")))
        def gen(ps):
            return llm.generate(ps, sp, lora_request=lora_req) if lora_req \
                else llm.generate(ps, sp)

        outs = gen(prompts)
        # vLLM can silently return fewer RequestOutputs than prompts submitted
        # (observed with no exception, no warning — requests just vanish under
        # certain internal scheduling/abort conditions). zip() would then
        # truncate to the shorter list without any signal, dropping examples
        # from the metric denominator. Detect and retry the shortfall once;
        # if it still comes up short, retry each missing prompt individually
        # rather than silently reporting a smaller n.
        if len(outs) != len(prompts):
            print(f"WARNING: rank {rank} src {src}: generate() returned {len(outs)} "
                  f"outputs for {len(prompts)} prompts — retrying shortfall", flush=True)
            outs = gen(prompts)
            if len(outs) != len(prompts):
                print(f"WARNING: rank {rank} src {src}: retry still short "
                      f"({len(outs)}/{len(prompts)}) — retrying each prompt individually",
                      flush=True)
                outs = []
                for p in prompts:
                    o = gen([p])
                    outs.append(o[0] if o else None)
        recs = []
        for ex, o in zip(examples, outs):
            if o is None:
                raise RuntimeError(f"Generation missing after retries: {ex['id']}")
            text = o.outputs[0].text
            m = score(text, ex["gold_calls"], args.profile)
            recs.append({"id": ex["id"], **m, "raw": text,
                         "gold_calls": ex["gold_calls"], "finish_reason": o.outputs[0].finish_reason})
        results[src] = recs
    out_q.put((rank, results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None, help="optional LoRA adapter path")
    ap.add_argument("--tests", nargs="+", default=["xlam"], choices=["xlam"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-ids", help="Explicit physical GPU IDs; length must equal --gpus")
    ap.add_argument("--execution-backend", default="local_vllm", choices=("local_vllm",))
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel per replica")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="cap examples per test set")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "eval"))
    ap.add_argument("--data-dir", help="Versioned GRPO dataset directory")
    ap.add_argument("--chat-template", help="Frozen canonical chat template used by corrected runs")
    ap.add_argument("--base-revision", help="Immutable base-model commit used for this experiment")
    ap.add_argument("--run-name", default=None,
                     help="output subdir name; defaults to basename(lora or model), which "
                          "is wrong for zero-shot rows (bare HF id like 'Qwen/Qwen3-8B' -> "
                          "'Qwen3-8B', not matching the caller's run_name) — always pass "
                          "this explicitly from run_all_evals.sh")
    ap.add_argument("--profile", required=True, choices=("original","revision_v2"))
    ap.add_argument("--thinking", required=True, choices=("enabled","disabled"))
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    if args.profile=="revision_v2" and args.thinking!="disabled":
        ap.error("The corrected protocol requires thinking disabled")
    args.data_dir=args.data_dir or os.path.join(ROOT,"datasets",args.profile,"grpo")
    if args.profile=="revision_v2" and not args.chat_template:
        args.chat_template=os.path.join(ROOT,"metadata","revision_chat_template.jinja")
    if args.model.startswith("Qwen/") and not args.base_revision:
        with open(os.path.join(ROOT,"metadata","model_revisions.json")) as stream:
            args.base_revision=json.load(stream)[args.model]
    if args.gpu_ids:
        ids=list(map(int,args.gpu_ids.split(',')))
        if len(ids)!=args.gpus or len(set(ids))!=len(ids) or min(ids)<0:
            ap.error('--gpu-ids must specify --gpus distinct nonnegative IDs')

    assert args.gpus % args.tp == 0, "gpus must be divisible by tp"
    n_workers = args.gpus // args.tp
    data = {src: load_test(src, args.data_dir) for src in args.tests}
    if args.limit:
        data = {s: d[:args.limit] for s, d in data.items()}

    # shard: round-robin so workers get balanced mixes
    shards = [{src: exs[r::n_workers] for src, exs in data.items()}
              for r in range(n_workers)]

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, args, shards[r], out_q))
             for r in range(n_workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    merged = {src: [] for src in args.tests}
    # A worker that dies before calling out_q.put() (e.g. vLLM engine init
    # failure from a leftover-memory GPU) never produces a result — blindly
    # calling out_q.get() n_workers times then hangs forever. Poll with a
    # timeout and stop waiting on any worker whose process has exited without
    # a result, so one bad GPU degrades to partial results instead of an
    # infinite hang (bit us: 1h47m stuck on a single crashed worker).
    n_received = 0
    grace_used = False
    while n_received < len(procs):
        alive = [p for p in procs if p.is_alive()]
        try:
            _, res = out_q.get(timeout=5)
        except Exception:
            if not alive:
                # A worker's put() can still be flushing through the pipe after
                # its process has already exited (documented multiprocessing
                # gotcha) — is_alive()==False does not mean the data is lost,
                # only that the sender is done. Give one long final drain
                # before concluding results are actually missing.
                if not grace_used:
                    grace_used = True
                    continue_draining = True
                    while continue_draining:
                        try:
                            _, res = out_q.get(timeout=30)
                            n_received += 1
                            for src, recs in res.items():
                                merged[src].extend(recs)
                        except Exception:
                            continue_draining = False
                    if n_received >= len(procs):
                        break
                dead_without_result = len(procs) - n_received
                print(f"WARNING: {dead_without_result} worker(s) exited without producing "
                      f"a result (see per-worker errors above) — proceeding with partial results",
                      flush=True)
                break
            continue
        n_received += 1
        for src, recs in res.items():
            merged[src].extend(recs)
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    run = args.run_name or os.path.basename(os.path.normpath(args.lora or args.model))
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for src, recs in merged.items():
        expected = [ex["id"] for ex in data[src]]
        actual = [r["id"] for r in recs]
        if not expected or len(set(expected)) != len(expected):
            raise RuntimeError(f"Empty or duplicate test IDs for {src}")
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise RuntimeError(f"Incomplete/duplicate evaluation for {src}: {len(actual)}/{len(expected)}")
    for src, recs in merged.items():
        n = len(recs)
        metrics = {
            "n": n,
            "exact_match": round(sum(r["exact"] for r in recs) / n, 4),
            "literal_exact_match": round(sum(r["literal_exact"] for r in recs) / n, 4),
            "name_acc": round(sum(r["name"] for r in recs) / n, 4),
            "arg_score": round(sum(r["arg"] for r in recs) / n, 4),
            "parse_rate": round(sum(r["parsed"] for r in recs) / n, 4),
            "valid_rate": round(sum(r["valid"] for r in recs) / n, 4),
        }
        for label, subset in (("call", [r for r in recs if r["gold_calls"]]),
                              ("no_call", [r for r in recs if not r["gold_calls"]])):
            metrics[f"n_{label}"] = len(subset)
            metrics[f"{label}_exact_match"] = (sum(r["exact"] for r in subset) / len(subset)
                                                if subset else None)
            for key in ("name", "arg", "parsed", "valid"):
                metrics[f"{label}_{key}"] = (sum(r[key] for r in subset) / len(subset)
                                            if subset else None)
        metrics['balanced_call_no_call_exact_match'] = (
            (metrics['call_exact_match']+metrics['no_call_exact_match'])/2
            if metrics['n_call'] and metrics['n_no_call'] else None)
        metrics['always_no_call_baseline'] = metrics['n_no_call']/n
        metrics['name_recall_on_call_rows'] = metrics['call_name']
        metrics['argument_score_on_call_rows'] = metrics['call_arg']
        metrics["data_dir"] = os.path.abspath(args.data_dir or os.path.join(ROOT, "datasets/grpo"))
        with open(os.path.join(metrics["data_dir"], f"{src}_test.jsonl"), "rb") as stream:
            metrics["data_sha256"] = hashlib.sha256(stream.read()).hexdigest()
        with open(os.path.join(ROOT, "scripts/grpo_reward.py"), "rb") as stream:
            metrics["scorer_sha256"] = hashlib.sha256(stream.read()).hexdigest()
        metrics["run_name"] = run
        metrics["model"] = args.model
        metrics["thinking"] = args.thinking == "enabled"
        metrics["profile"] = args.profile
        metrics["decode_seed"] = args.seed
        metrics["temperature"] = 0.0
        metrics["max_new_tokens"] = args.max_new
        metrics["gpu_count"] = args.gpus
        metrics["execution_backend"] = args.execution_backend
        metrics["physical_gpu_ids"] = args.gpu_ids
        import importlib.metadata
        metrics["inference_packages"] = {name:importlib.metadata.version(name)
                                         for name in ("vllm","torch","transformers")}
        metrics["tensor_parallel_size"] = args.tp
        metrics["base_model_revision"] = args.base_revision
        if args.chat_template:
            with open(args.chat_template, "rb") as stream:
                metrics["chat_template_sha256"] = hashlib.sha256(stream.read()).hexdigest()
        summary[src] = metrics
        with open(os.path.join(out_dir, f"{src}_predictions.jsonl"), "w") as f:
            for r in sorted(recs, key=lambda x: x["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"{src}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps({"model": run, "elapsed_s": round(time.time() - t0, 1),
                      **summary}, indent=2))


if __name__ == "__main__":
    main()
