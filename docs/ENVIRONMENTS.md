# Environments and reproducibility limits

## CPU artifact verification

Python 3.11 or 3.12 and the standard library are sufficient for the unit tests,
archive verification, raw historical reconstruction, and saved-output scoring.
No model or dataset service is contacted by those commands.

## Revised inference and GRPO

`requirements-eval.txt` records the revised local inference versions: vLLM
0.11.0, PyTorch 2.8.0, transformers 4.57.1, and peft 0.17.1. A suitable NVIDIA
driver and CUDA runtime are required. The study's local evaluation used four
A100 80GB GPUs, with tensor parallelism 2 for 32B.

GRPO uses verl 0.7.0. Install the inference environment first, then
`requirements-grpo.txt`, following verl's platform-specific installation
instructions. Do not silently substitute a different major trainer or vLLM
version. Public scripts do not deploy infrastructure or access a cloud account.

The default four-GPU recipes preserve global SFT batch 64 and GRPO prompt batch
256 / actor mini-batch 64. The revised 32B recipe reduces microbatches for
memory; its results remain pending. Historical 32B recipes originated on eight
GPUs and are not certified to fit four GPUs unchanged. No training command is
executed as part of release verification.

## SFT

The public training archives contain a documented upstream credential
redaction. Use `scripts/restore_training_inputs.py` with your own exact upstream
snapshot for original training-input reproduction. The script never tests that
credential. Offline test scoring does not require restoration.

Install LLaMA-Factory from its official repository in a separate environment.
The SFT experiments used transformers 4.57.3 and LoRA rank 32 / alpha 64. The
artifact scripts do not vendor the training framework or claim that every
current framework release reproduces historical generation exactly.

Before revised training, run the tokenizer/SFT rendering checks:

```bash
python scripts/runtime_preflight.py --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --checks metadata/tokenizer_checks.json \
  --chat-template metadata/revision_chat_template.jinja --factory
```

The original profile's generated-output runtime was not completely pinned.
Its saved parsed calls reproduce the reported scores. The archival raw parser
does not reconstruct every stored historical parse decision, so raw reparsing
is a separate diagnostic (`--reparse-original`), not exact historical pipeline
reproduction. New generation may differ
because of model revisions, thinking mode, framework versions, GPU kernels,
or batching. The original and revised scorers intentionally remain separate.

## Checkpoints

Checkpoint weights are not in this Git repository. Train with the frozen data
and supplied recipes to create them. The upstream Qwen models have separate
model cards and licenses. The hybrid requires an actual merged SFT checkpoint;
its launcher refuses to silently initialize from the untrained base.
