# Data license and provenance

The included xLAM source snapshot and derived data are adapted from
**Salesforce Research / the APIGen dataset authors**,
`Salesforce/xlam-function-calling-60k`, under **Creative Commons Attribution
4.0 International (CC BY 4.0)**.

- Source: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- APIGen paper: https://arxiv.org/abs/2406.18518
- License: https://creativecommons.org/licenses/by/4.0/
- Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode
- Preserved attribution and citation: `data/upstream/README.md`.

Modifications include JSONL normalization, historical deduplication, frozen
train/validation/test splits, explicit training exclusions, revised subset
selection, export to SFT/GRPO formats, and redaction of one credential-shaped
upstream tool-schema default from the public training copies. The original
study files were not modified; test and prediction files are unchanged.
SHA256 hashes identify the exact
included snapshot and each transformation's frozen outputs.

Upstream license metadata was checked at the revision recorded in
`data/upstream/dataset_info.json`. The original raw download's upstream commit
was not recorded; its local snapshot hash is provided instead. The metadata
check is not a claim that the raw download was made at that same commit.

Saved generations and numerical result data are released under CC BY 4.0 by
the artifact maintainer. Code is licensed separately under Apache-2.0.
No endorsement by Salesforce or the APIGen authors is implied.
