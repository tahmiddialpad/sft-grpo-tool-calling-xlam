# Environment setup

## CPU verification

Python 3.11 or 3.12 and its standard library run the unit tests, archive
verification, raw-data reconstruction, and saved-output scoring.

```bash
python -m unittest discover -s tests
python scripts/validate_release.py
```

Saved-output scoring uses the stored parsed calls and the original matcher.
Raw-text call extraction is available separately with `--reparse-original`.

## LoRA SFT

Install LLaMA-Factory from its official repository in a separate environment.
The training data uses the ShareGPT format with an explicit system prompt.
The launcher writes a resolved configuration and launch metadata to `outputs/`.

Default layout: four GPUs, per-device batch 4, gradient accumulation 4,
effective batch 64. Other GPU layouts preserve effective batch 64.

## GRPO and inference

`requirements-eval.txt` supplies a vLLM reference environment; install
`requirements-grpo.txt` for verl training in that environment. A compatible
NVIDIA driver/CUDA runtime is required.

The recipes use verl 0.7.0, eight rollouts per prompt, global prompt batch 256,
actor mini-batch 64, learning rate 1e-6, and KL coefficient 0.001. The GPU count
is configurable. For 32B inference, select a compatible tensor-parallel layout
with `--tp`.

The hybrid launcher requires an explicitly supplied merged SFT checkpoint.
Data restoration for the upstream schema-default redaction requires
`huggingface_hub` and `pyarrow`. Checkpoint outputs are written to ignored local
directories.
