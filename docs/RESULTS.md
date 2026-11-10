# Original xLAM results

Normalized tool-call exact match (%). Each model is evaluated on the same 1,000 xLAM test examples.

| Model | Zero-shot | LoRA SFT | GRPO | SFT → GRPO |
|---|---:|---:|---:|---:|
| 0.6B | 66.00 | 85.60 | 79.20 | 81.10 |
| 1.7B | 68.10 | 87.20 | 81.40 | 82.60 |
| 4B | 76.60 | 88.00 | 83.30 | 84.20 |
| 8B | 75.30 | 87.70 | 84.40 | 86.20 |
| 14B | 77.80 | 88.80 | 85.10 | 86.20 |
| 32B | 78.30 | 88.70 | 85.80 | 86.00 |
| **Average** | **73.68** | **87.67** | **83.20** | **84.38** |

Saved predictions and numerical metrics are included for all 24 evaluations.
