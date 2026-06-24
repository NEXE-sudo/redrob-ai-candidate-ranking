"""Check honeypot risk for three candidate IDs and print results.
Run: python3 scripts/check_three_candidates.py
"""
import os
import sys
import json
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'backend'))

from engines.candidate_profile_parser import CandidateProfileParser
from engines.advanced_scoring_components import HoneypotDetector

CAND_IDS = ['CAND_0005538', 'CAND_0039754', 'CAND_0088025']

# locate candidates.jsonl in repo
candidates_path = None
possible = [
    os.path.join(ROOT, '[PUB] India_runs_data_and_ai_challenge', 'India_runs_data_and_ai_challenge', 'candidates.jsonl'),
    os.path.join(ROOT, 'India_runs_data_and_ai_challenge', 'candidates.jsonl')
]
for p in possible:
    if os.path.exists(p):
        candidates_path = p
        break
# fallback: search recursively
if not candidates_path:
    for p in Path(ROOT).rglob('candidates.jsonl'):
        candidates_path = str(p)
        break

if not candidates_path:
    print('Could not find candidates.jsonl in workspace')
    sys.exit(2)

parser = CandidateProfileParser()
detector = HoneypotDetector()

with open(candidates_path, 'r') as f:
    for line in f:
        try:
            cand = json.loads(line)
        except Exception:
            continue
        cid = cand.get('candidate_id')
        if cid in CAND_IDS:
            parsed = parser.parse_candidate(cand)
            risk = detector.calculate_risk_score(parsed, cand)
            penalty = detector.get_penalty_multiplier(risk)
            overlap = detector._has_overlapping_employment(parsed)
            print(f"{cid}: risk={risk:.3f}, penalty_multiplier={penalty:.3f}, overlap_flag={overlap}")

print('Done')
