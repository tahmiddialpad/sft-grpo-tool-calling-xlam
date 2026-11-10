# SFT vs GRPO for Tool Calling — xLAM

Original-paper code, data, training configurations, saved predictions, and
results for the xLAM experiments in **“SFT or RL for Tool-Calling LLM Agents?
A Controlled Study Across Data, Method, and Scale.”**

The release covers six Qwen3 scales (0.6B, 1.7B, 4B, 8B, 14B, 32B), LoRA SFT,
GRPO, SFT followed by GRPO, and zero-shot baselines: **24 evaluations**.

## Data

| Split | Samples |
|---|---:|
| Training | 56,438 |
| Validation | 500 |
| Test | 1,000 |

Every test example requires a tool call. Each method uses the same test set.
The hybrid uses disjoint training halves of 28,219 examples each.

Compressed data archives are included in `data/`. Checksums and membership are
recorded in `data/manifest.json`. Public training copies redact one
credential-shaped upstream schema default; test references and saved predictions
are unchanged. The restoration command below reconstructs the study training
inputs locally from the upstream snapshot.

## Verify the results

No GPU, model download, or dataset account is needed for saved-output scoring.

```bash
python -m unittest discover -s tests
python scripts/validate_release.py
python scripts/evaluate_saved.py \
  --predictions results/original/predictions/zeroshot_qwen3-0_6b.jsonl.gz
```

Saved-output evaluation matches the released parsed predictions against the
gold tool calls; it recomputes scores rather than copying stored score fields.
Use `--reparse-original` to extract calls from raw text with the included parser.
The scorer reports normalized tool-call exact match, tool-name score, argument
score, and parse rate. String argument values are case-insensitive and numeric
strings are normalized.

## Prepare training inputs

```bash
python scripts/prepare_data.py
# For the exact study training inputs:
pip install huggingface_hub pyarrow
python scripts/download_upstream.py
python scripts/restore_training_inputs.py \
  --upstream-jsonl outputs/upstream/xlam_train.jsonl
```

The upstream download uses your Hugging Face access. An existing JSONL snapshot
with the recorded hash can also be supplied. Restored files stay in ignored
`datasets/` and `outputs/` directories. For a new experiment using the public
redacted prompts, pass `--allow-redacted-data` to the training launcher.

The raw-data preprocessing is also included:

```bash
python scripts/preprocess_xlam.py
```

It applies the original normalized-query/called-name deduplication and seed-42
split. Five training exclusions are recorded in
`metadata/original_training_exclusions.json`.

## Train

Install LLaMA-Factory for SFT, and use a separate verl/vLLM environment for GRPO
and inference. See [environment setup](docs/ENVIRONMENTS.md).
Launchers default to four GPUs; SFT preserves effective batch size 64.

```bash
# LoRA SFT
python scripts/train_sft.py --scale 0_6b --stage sft --gpu-ids 0,1,2,3

# GRPO
python scripts/train_grpo.py --scale 0_6b --stage grpo --gpu-ids 0,1,2,3

# SFT -> GRPO
python scripts/train_sft.py --scale 0_6b --stage sftgrpo_sft --gpu-ids 0,1,2,3
python scripts/merge_lora_checkpoint.py \
  --adapter-dir outputs/original/0_6b_sftgrpo_sft \
  --base-model Qwen/Qwen3-0.6B --out-dir outputs/original/0_6b_warm_merged
python scripts/train_grpo.py --scale 0_6b --stage sftgrpo_grpo \
  --model outputs/original/0_6b_warm_merged --gpu-ids 0,1,2,3
```

LoRA uses rank 32, alpha 64 and dropout 0.05. Pure SFT uses two epochs. The
hybrid warm-up uses one epoch; its GRPO stage uses two epochs on the other half.
GRPO uses eight rollouts per prompt and the released reward function.

## Inference and evaluation

```bash
python scripts/run_vllm.py --model Qwen/Qwen3-0.6B --thinking enabled \
  --gpus 4 --gpu-ids 0,1,2,3 --run-name zero_shot --out outputs/eval
```

Use `--lora <adapter-directory>` for an SFT adapter or `--model <checkpoint>`
for a merged model. Thinking mode, tensor parallelism and model revision are
explicit command-line options.

## Results

See [the original results](docs/RESULTS.md), [numerical metrics](results/metrics.csv),
and [saved predictions](results/original/predictions). All 24 evaluations use
1,000 test samples each.

## Licensing

Code: [Apache-2.0](LICENSE). Data: [CC BY 4.0](DATA_LICENSE.md), adapted from
[Salesforce APIGen/xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
The [upstream dataset card](data/upstream/README.md) contains attribution and
the APIGen citation. Base-model weights are obtained separately.
