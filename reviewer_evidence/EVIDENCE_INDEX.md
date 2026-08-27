# Reviewer evidence index

| Reviewer question | Repository evidence |
|---|---|
| What model/checkpoint produced the large-gallery FT result? | `CHECKPOINT_PROVENANCE.md`, `reproducibility/checkpoints/checkpoint_audit.csv`, `results/full_gallery/ft50k/` |
| Can the 71,918-pair metrics be checked query-by-query? | `results/full_gallery/*/retrieval_top10_and_ranks.pt` plus the statistical script in `code/` |
| Why are there 71,918 rather than 71,919 usable test pairs? | `results/full_gallery/*/bad_images.json` |
| Are the public dataset splits independent? | `DATASET_SPLIT_EVIDENCE.md`, `reproducibility/dataset_audit/manifest_audit_summary.json` |
| What controlled subsets are available? | `reproducibility/controlled_subsets/` and their uncompressed SHA256 file |
| What exactly was trainable? | `reproducibility/checkpoints/trainable_parameter_audit.txt` |
| Where are uncertainty/statistical outputs? | `results/statistics/` and `code/full_gallery_statistical_analysis.py` |
| Where are downstream representation checks? | `results/downstream/` |
| What historical limitations remain? | `REPRODUCIBILITY_LIMITS.md` |
| How are the large original checkpoint archives verified? | `LARGE_ARTIFACTS.csv` |
