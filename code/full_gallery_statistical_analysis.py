#!/usr/bin/env python3
"""Reproduce query-level uncertainty analyses reported in the ACISM study.

Inputs are the saved full-gallery top-10 indices and strict-greater ranks from the
unchanged baseline and FT-50k evaluation artifacts. Recall@K is defined from
membership of the designated exact-index target in torch.topk output. Median
rank uses the saved strict-greater rank convention from the original evaluator.

The bootstrap intentionally uses one RNG stream, seed 20260821, across arrays
in the fixed order baseline-I2T, FT50k-I2T, baseline-T2I, FT50k-T2I. This makes
reported percentile intervals exactly reproducible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, norm


def wilson_interval(x: int, n: int, alpha: float = 0.05):
    z = float(norm.ppf(1 - alpha / 2))
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def exact_hits(obj, direction: str, k: int):
    idx = obj[f"{direction}_top10_indices"][:, :k].cpu()
    target = torch.arange(idx.shape[0], dtype=idx.dtype).view(-1, 1)
    return (idx == target).any(dim=1).numpy().astype(bool)


def lower_median(x):
    x = np.asarray(x)
    kth = (x.size - 1) // 2
    return float(np.partition(x, kth)[kth])


def bootstrap_rank_intervals(arrays, B=2000, seed=20260821):
    rng = np.random.default_rng(seed)
    result = []
    for label, arr in arrays:
        arr = np.asarray(arr)
        n = arr.size
        kth = (n - 1) // 2
        medians = np.empty(B, dtype=float)
        for b in range(B):
            sample = arr[rng.integers(0, n, size=n)]
            medians[b] = np.partition(sample, kth)[kth]
        lo, hi = np.percentile(medians, [2.5, 97.5])
        result.append((label, lower_median(arr), float(lo), float(hi)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--ft50k', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base = torch.load(args.baseline, map_location='cpu', weights_only=False)
    ft = torch.load(args.ft50k, map_location='cpu', weights_only=False)
    n = int(len(base['i2t_ranks']))
    assert n == len(ft['i2t_ranks']) == 71918

    recall_rows = []
    for direction in ('i2t', 't2i'):
        for k in (1, 5, 10):
            hb = exact_hits(base, direction, k)
            hf = exact_hits(ft, direction, k)
            xb, xf = int(hb.sum()), int(hf.sum())
            b_lo, b_hi = wilson_interval(xb, n)
            f_lo, f_hi = wilson_interval(xf, n)
            base_only = int(np.sum(hb & ~hf))
            ft_only = int(np.sum(~hb & hf))
            discordant = base_only + ft_only
            p = 1.0 if discordant == 0 else float(binomtest(base_only, discordant, 0.5).pvalue)
            recall_rows.append({
                'direction': direction.upper(), 'K': k, 'N': n,
                'baseline_hits': xb, 'baseline_recall_pct': 100*xb/n,
                'baseline_wilson95_lo_pct': 100*b_lo, 'baseline_wilson95_hi_pct': 100*b_hi,
                'ft50k_hits': xf, 'ft50k_recall_pct': 100*xf/n,
                'ft50k_wilson95_lo_pct': 100*f_lo, 'ft50k_wilson95_hi_pct': 100*f_hi,
                'relative_change_pct': 100*((xf/n)/(xb/n)-1) if xb else np.nan,
                'baseline_only_hits': base_only, 'ft50k_only_hits': ft_only,
                'discordant_queries': discordant, 'paired_exact_mcnemar_p': p,
            })
    pd.DataFrame(recall_rows).to_csv(outdir/'full_gallery_recall_uncertainty.csv', index=False)

    arrays = [
        ('I2T Baseline', base['i2t_ranks'].cpu().numpy()),
        ('I2T FT50k', ft['i2t_ranks'].cpu().numpy()),
        ('T2I Baseline', base['t2i_ranks'].cpu().numpy()),
        ('T2I FT50k', ft['t2i_ranks'].cpu().numpy()),
    ]
    rank_rows = []
    for label, med, lo, hi in bootstrap_rank_intervals(arrays, B=2000, seed=20260821):
        direction, model = label.split()
        rank_rows.append({'direction': direction, 'model': model, 'N': n,
                          'strict_greater_median_rank_positions': med,
                          'bootstrap95_lo_positions': lo,
                          'bootstrap95_hi_positions': hi,
                          'bootstrap_resamples': 2000, 'bootstrap_seed': 20260821})
    pd.DataFrame(rank_rows).to_csv(outdir/'full_gallery_rank_bootstrap.csv', index=False)

    # Controlled 5k R@10 Wilson intervals from recorded hit rates; these are
    # included only as uncertainty for the controlled subset, not as paired tests
    # because per-query controlled-subset hit vectors were not retained here.
    crows=[]
    for direction, model, rate in [
        ('I2T','Baseline',0.0388), ('I2T','FT50k-final-epoch3',0.0760),
        ('T2I','Baseline',0.0446), ('T2I','FT50k-final-epoch3',0.0790)]:
        nn=5000; x=int(round(rate*nn)); lo,hi=wilson_interval(x,nn)
        crows.append({'direction':direction,'model':model,'K':10,'N':nn,'hits':x,
                      'recall_pct':100*x/nn,'wilson95_lo_pct':100*lo,'wilson95_hi_pct':100*hi})
    pd.DataFrame(crows).to_csv(outdir/'controlled5k_R10_wilson.csv', index=False)

    with open(outdir/'statistical_audit_README.txt','w',encoding='utf-8') as fh:
        fh.write('Statistical audit - 2026-08-21\n')
        fh.write('N=71,918 valid full-gallery image-text pairs.\n')
        fh.write('Recall@K: exact target-index membership in saved torch.topk indices.\n')
        fh.write('95% CI for Recall@K: Wilson score interval for a binomial proportion.\n')
        fh.write('Paired comparison: exact McNemar test implemented as an exact two-sided binomial test on discordant queries.\n')
        fh.write('Rank: saved strict-greater rank = 1 + number of candidates with similarity strictly greater than target similarity.\n')
        fh.write('Rank bootstrap: 2,000 query resamples, seed 20260821, one sequential RNG stream in the fixed array order documented in the script.\n')
        fh.write('Important: strict-greater rank uses a best-rank convention under exact ties and is not algebraically reconstructed from torch.topk hits.\n')

if __name__ == '__main__':
    main()
