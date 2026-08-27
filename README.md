# US-365K BioMedCLIP Reproducibility

**Repository owner / first author:** Bui Xuan Tung (GitHub: [`@bxtung-hub`](https://github.com/bxtung-hub))  
**Repository:** https://github.com/bxtung-hub/us365k-biomedclip-reproducibility

This repository contains the reproducibility and reviewer-verification materials for **Data-Efficient Projection-Related Tuning of BioMedCLIP for Ultrasound Image--Text Retrieval Using US-365K** (ACISM 2026).

## What is included

- final 15-page manuscript source and PDF;
- preserved data-preparation, training, evaluation, and statistical-analysis code;
- controlled retrieval summaries;
- full-gallery baseline and FT50k metrics, qualitative examples, per-query top-10/rank tensors, and bad-image record;
- downstream zero-shot/linear-probe metric outputs;
- deterministic regenerated controlled-subset manifests;
- compressed public split metadata and enriched audit manifests;
- split-independence audit, checkpoint audit, environment records, and SHA256 evidence;
- a reviewer evidence index explaining where each reported claim can be checked.

## Headline reproducibility facts

- Total parameters: **195,902,721**.
- Trainable projection-related parameters: **8,890,113 (4.5380%)**.
- Full public test split: **71,919** pairs; one unreadable image leaves **71,918 usable pairs** for the large-gallery evaluation.
- Full-gallery FT50k uses the validation-selected **epoch-2** checkpoint, SHA256 `3b7cb059523804c75cbbfc5a3ca16ad43c7910ea4f66c348f7968a64c6e1799f`.
- I2T R@10: **0.3740% -> 0.4436%**.
- T2I R@10: **0.4602% -> 0.6938%**.
- Strict-greater median rank: I2T **5806 -> 2536**, T2I **5631 -> 2482**.
- The preserved split audit supports **case-disjoint and image-disjoint** train/validation/test partitions.

## Data

US-365K is obtained from the public dataset release cited in the manuscript. The source image archive is not redistributed here. This repository includes compressed split metadata, audit manifests, and regenerated controlled-subset JSONL files.

## Large checkpoints

The original multi-volume checkpoint archives are not stored in the ordinary Git repository because individual volumes are hundreds of megabytes and the complete set is several gigabytes. `reviewer_evidence/LARGE_ARTIFACTS.csv` records their exact size and SHA256, while `reviewer_evidence/CHECKPOINT_PROVENANCE.md` records the SHA256 of each preserved `best_model.pt`.

## Important provenance note

See `reviewer_evidence/REPRODUCIBILITY_LIMITS.md`. The repository distinguishes preserved historical evidence from deterministic regeneration and does not claim that regenerated controlled-subset files are byte-identical historical originals.

## Reviewer verification

Start with [`reviewer_evidence/EVIDENCE_INDEX.md`](reviewer_evidence/EVIDENCE_INDEX.md).
