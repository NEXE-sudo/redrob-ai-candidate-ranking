#!/usr/bin/env python3
"""Holdout harness: score a sampled subset of candidates and report distributions.

Usage:
  python3 scripts/holdout_harness.py --candidates <path> --n 500
"""
import argparse
import json
import random
import os
import sys
from statistics import mean, median
from datetime import datetime

# Ensure backend package is importable when running from scripts/
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'backend'))

from engines.candidate_profile_parser import CandidateProfileParser
from engines.feature_scorer import FeatureScorer
from engines.advanced_scoring_components import HoneypotDetector, CareerTrajectoryAnalyzer


def sample_candidates(path, n=500):
    with open(path, 'r') as f:
        lines = f.readlines()
    if n >= len(lines):
        chosen = lines
    else:
        chosen = random.sample(lines, n)
    return [json.loads(l) for l in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True)
    p.add_argument('--n', type=int, default=500)
    args = p.parse_args()

    candidates = sample_candidates(args.candidates, args.n)
    parser = CandidateProfileParser()
    scorer = FeatureScorer(parser=parser)
    honeypot = HoneypotDetector()

    final_scores = []
    risk_scores = []

    for c in candidates:
        parsed = parser.parse_candidate(c)
        sc = scorer.score_candidate(c, parsed, semantic_similarity=0.0, advanced_scorer=None)
        score = sc.final_score
        final_scores.append(score)
        risk = honeypot.calculate_risk_score(parsed, c)
        risk_scores.append(risk)

    print(f"Sampled candidates: {len(final_scores)}")
    print(f"Final score - mean: {mean(final_scores):.4f}, median: {median(final_scores):.4f}, min: {min(final_scores):.4f}, max: {max(final_scores):.4f}")
    print(f"Risk score - mean: {mean(risk_scores):.4f}, median: {median(risk_scores):.4f}, min: {min(risk_scores):.4f}, max: {max(risk_scores):.4f}")

    # Top 10 by score
    top = sorted(zip(final_scores, candidates), key=lambda x: x[0], reverse=True)[:10]
    print('\nTop 10 from sample:')
    for s, c in top:
        cid = c.get('candidate_id')
        print(f"  {cid}: {s:.4f}")

if __name__ == '__main__':
    main()
