import sys
import json
from pathlib import Path

# Import the JD from run_ranking_optimized
sys.path.insert(0, str(Path(__file__).parent))
from run_ranking_optimized import JD_TEXT

import subprocess

candidates_file = "/home/NEXE/projects/Redrob hackathon/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"

cmd = [
    "./venv_ranking/bin/python3", 
    "benchmark_ranking.py", 
    candidates_file, 
    JD_TEXT
]

print("Running benchmark...")
subprocess.run(cmd)
