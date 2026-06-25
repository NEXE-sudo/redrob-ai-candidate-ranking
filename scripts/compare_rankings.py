#!/usr/bin/env python3
"""Compare two submission CSVs by top-K overlap and Spearman rank correlation.

Usage:
  python3 scripts/compare_rankings.py baseline.csv updated.csv --topk 100

CSV format: should contain columns `candidate_id` and `rank` or `final_score`.
"""
import sys
import csv
import argparse
from collections import defaultdict
import math


def read_ranks(path):
    ranks = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            cid = row.get('candidate_id') or row.get('candidate') or row.get('id')
            if not cid:
                continue
            # prefer explicit rank column
            if 'rank' in row and row['rank']:
                try:
                    r = int(row['rank'])
                except Exception:
                    r = i
            else:
                r = i
            ranks[cid] = r
    return ranks


def topk_overlap(ranks_a, ranks_b, k=100):
    top_a = set([c for c, r in ranks_a.items() if r <= k])
    top_b = set([c for c, r in ranks_b.items() if r <= k])
    inter = top_a & top_b
    return len(inter), len(top_a), len(top_b), len(inter) / max(1, k)


def spearman_rho(ranks_a, ranks_b):
    # compute Spearman rho on intersection
    common = set(ranks_a.keys()) & set(ranks_b.keys())
    n = len(common)
    if n < 2:
        return float('nan')
    d2 = 0.0
    for c in common:
        d = ranks_a[c] - ranks_b[c]
        d2 += d * d
    rho = 1.0 - (6.0 * d2) / (n * (n * n - 1))
    return rho


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('updated')
    p.add_argument('--topk', type=int, default=100)
    args = p.parse_args()

    a = read_ranks(args.baseline)
    b = read_ranks(args.updated)

    intersect = set(a.keys()) & set(b.keys())
    print(f"Items in baseline: {len(a)}, updated: {len(b)}, intersection: {len(intersect)}")

    overlap, na, nb, frac = topk_overlap(a, b, k=args.topk)
    print(f"Top-{args.topk} overlap: {overlap}/{args.topk} ({frac:.2%})")

    rho = spearman_rho(a, b)
    print(f"Spearman rho (common items): {rho:.4f}")


if __name__ == '__main__':
    main()
