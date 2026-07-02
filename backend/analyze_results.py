import sys
import os
import json
import pandas as pd
from pathlib import Path

ROOT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'ranking_output'

def check_output_validation():
    print("\n" + "="*50)
    print("PHASE 2: OUTPUT VALIDATION")
    print("="*50)
    output_dir = ROOT_OUTPUT_DIR
    if not output_dir.exists():
        print("Error: ranking_output dir not found")
        return False
        
    csv_path = output_dir / 'NEXE-sudo.csv'
    json_path = output_dir / 'ranking_detailed.json'
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print("Error: Missing NEXE-sudo.csv or ranking_detailed.json")
        return False
        
    df = pd.read_csv(csv_path)
    print(f"CSV Rows: {len(df)}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"JSON Entries: {len(data)}")
    
    if len(df) != 100 or len(data) != 100:
        print("FAIL: Not exactly 100 candidates!")
    else:
        print("PASS: Exactly 100 candidates.")
        
    # Check monotonically decreasing scores
    scores = df['score'].tolist()
    sorted_scores = sorted(scores, reverse=True)
    if scores != sorted_scores:
        print("FAIL: Scores are not monotonically decreasing!")
    else:
        print("PASS: Scores are monotonically decreasing.")
        
    # Check duplicates
    if len(df['candidate_id'].unique()) != len(df):
        print("FAIL: Duplicate candidate IDs found!")
    else:
        print("PASS: No duplicate candidate IDs.")
        
    # Check empty reasoning when the ranking output is expected to include it.
    empty_components = [d for d in data if not d.get('components')]
    if empty_components:
        print("FAIL: Some candidates have missing components/reasoning.")
    else:
        print("PASS: All candidates have components.")

def check_top100_audit():
    print("\n" + "="*50)
    print("PHASE 3: TOP 100 AUDIT")
    print("="*50)
    json_path = ROOT_OUTPUT_DIR / 'ranking_detailed.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ranks_to_check = [1, 5, 10, 20, 50, 75, 100]
    for r in ranks_to_check:
        if r <= len(data):
            c = data[r-1]
            print(f"Rank {r}: {c['candidate_id']} | Score: {c['final_score']}")
            print(f"  Technical: {c['components']['technical_relevance']:.2f}")
            print(f"  Profile Quality: {c['components']['profile_quality_multiplier']:.2f}")

def check_honeypot_analysis():
    print("\n" + "="*50)
    print("PHASE 4: HONEYPOT ANALYSIS")
    print("="*50)
    
    json_path = ROOT_OUTPUT_DIR / 'ranking_detailed.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load original candidate profiles to check for honeypots
    candidate_ids = set(d['candidate_id'] for d in data)
    
    candidates_file = sys.argv[1] if len(sys.argv) > 1 else "./candidates.jsonl"
    
    suspicious = 0
    with open(candidates_file, 'r') as f:
        for line in f:
            c = json.loads(line)
            if c['candidate_id'] in candidate_ids:
                # Check for honeypot indicators
                skills = c.get('skills', [])
                if len(skills) > 50:
                    print(f"Suspicious {c['candidate_id']}: >50 skills ({len(skills)})")
                    suspicious += 1
                
                exp = c.get('experience', [])
                total_years = 0
                for e in exp:
                    # just a heuristic check
                    pass
    
    print(f"Total Suspicious Profiles in Top 100: {suspicious}")

if __name__ == "__main__":
    check_output_validation()
    check_top100_audit()
    check_honeypot_analysis()
    print("\nExplainability output is reviewed directly from the ranking artifacts.")
