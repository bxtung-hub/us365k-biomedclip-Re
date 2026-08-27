Statistical audit - 2026-08-21
N=71,918 valid full-gallery image-text pairs.
Recall@K: exact target-index membership in saved torch.topk indices.
95% CI for Recall@K: Wilson score interval for a binomial proportion.
Paired comparison: exact McNemar test implemented as an exact two-sided binomial test on discordant queries.
Rank: saved strict-greater rank = 1 + number of candidates with similarity strictly greater than target similarity.
Rank bootstrap: 2,000 query resamples, seed 20260821, one sequential RNG stream in the fixed array order documented in the script.
Important: strict-greater rank uses a best-rank convention under exact ties and is not algebraically reconstructed from torch.topk hits.
