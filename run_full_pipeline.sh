#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CANDIDATES="$ROOT/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
DEFAULT_OUTPUT_DIR="$ROOT/ranking_output"
DEFAULT_LOG="$ROOT/ranking_run.log"

usage() {
  cat <<EOF
Usage: $0 [CANDIDATES_JSONL] [OUTPUT_DIR]
Run the full Redrob ranking pipeline from dataset to output and validate results.

Defaults:
  CANDIDATES_JSONL = $DEFAULT_CANDIDATES
  OUTPUT_DIR       = $DEFAULT_OUTPUT_DIR
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CANDIDATES="${1:-$DEFAULT_CANDIDATES}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
OUTPUT_CSV="$OUTPUT_DIR/submission.csv"

printf "Running full pipeline from start to finish...\n"
printf "Candidates: %s\n" "$CANDIDATES"
printf "Output directory: %s\n" "$OUTPUT_DIR"
printf "Log file: %s\n\n" "$DEFAULT_LOG"

mkdir -p "$OUTPUT_DIR"

python3 backend/rank.py --candidates "$CANDIDATES" --out "$OUTPUT_CSV" 2>&1 | tee "$DEFAULT_LOG"

printf "\nPipeline complete. Output written to %s\n" "$OUTPUT_CSV"
printf "Validating generated ranking output...\n"
python3 backend/analyze_results.py 2>&1 | tee -a "$DEFAULT_LOG"

printf "\nDone. Logs saved to %s\n" "$DEFAULT_LOG"
