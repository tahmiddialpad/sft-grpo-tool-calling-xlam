# xLAM results

Scores are normalized call-set exact match (%). All test sets have 1,000 call-required examples.
Original and revision profiles have different training pools, test membership and format acceptance; this is not a parser-only comparison.

## original

| Model | Zero-shot | LoRA SFT | GRPO | SFT → GRPO |
|---|---:|---:|---:|---:|
| 0.6B | 66.00 | 85.60 | 79.20 | 81.10 |
| 1.7B | 68.10 | 87.20 | 81.40 | 82.60 |
| 4B | 76.60 | 88.00 | 83.30 | 84.20 |
| 8B | 75.30 | 87.70 | 84.40 | 86.20 |
| 14B | 77.80 | 88.80 | 85.10 | 86.20 |
| 32B | 78.30 | 88.70 | 85.80 | 86.00 |

## revision_v2

| Model | Zero-shot | LoRA SFT | GRPO | SFT → GRPO |
|---|---:|---:|---:|---:|
| 0.6B | 31.00 | 80.10 | 77.40 | 80.10 |
| 1.7B | 53.50 | 83.80 | 78.90 | 82.10 |
| 4B | 78.80 | 85.30 | 83.10 | 85.30 |
| 8B | 78.90 | 86.00 | 84.50 | 86.00 |
| 14B | 79.60 | 86.50 | 85.70 | 86.10 |
| 32B | 82.90 | 87.30 | pending | pending |

The original profile uses permissive historical call extraction; the corrected profile uses the versioned strict scorer. The original scorer is retained rather than silently changed.

Saved generations are included for every numeric cell. Run `python scripts/validate_release.py` to recompute them.

The corrected 32B GRPO and hybrid results are pending; no full corrected average or final ranking is asserted.
