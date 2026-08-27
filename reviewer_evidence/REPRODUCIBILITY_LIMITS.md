# Reproducibility limits

The evidence is deliberately explicit about historical limitations.

1. The byte-identical historical controlled subset JSONL files were not separately preserved. Deterministically regenerated manifests are provided in `reproducibility/controlled_subsets/` and must be described as regenerated, not original.
2. Preserved project training code is provided, but a byte-identical executable snapshot from every historical FT run was not separately retained.
3. The historical requirements list `open_clip_torch` without an exact pinned version, so no unrelated runtime version is assigned retroactively.
4. No preserved LoRA/PEFT/adapter training run or result was found in the audited historical material.
5. The source image archive is not redistributed; use the public US-365K release referenced in the paper.
