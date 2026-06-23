import sys
import json
from pathlib import Path

# Import the JD from run_ranking_optimized
sys.path.insert(0, str(Path(__file__).parent))
from run_ranking_optimized import JD_TEXT

import subprocess

candidates_file = sys.argv[1] if len(sys.argv) > 1 else './candidates.jsonl'

cmd = [
    sys.executable,
    str(Path(__file__).parent / 'benchmark_ranking.py'),
    candidates_file,
    JD_TEXT
]

print("Running benchmark...")
subprocess.run(cmd)
