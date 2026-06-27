#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CANDIDATES="$ROOT/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
DEFAULT_OUTPUT_DIR="$ROOT/ranking_output"
DEFAULT_LOG="$ROOT/ranking_run.log"

FORCE_PRECOMPUTE=""

usage() {
  cat <<EOF
Usage: $0 [CANDIDATES_JSONL] [OUTPUT_DIR] [--force-precompute]
Run the full Redrob ranking pipeline from dataset to output and validate results.

Options:
  --force-precompute  Force re-running the precomputation even if a cache exists.

Defaults:
  CANDIDATES_JSONL = $DEFAULT_CANDIDATES
  OUTPUT_DIR       = $DEFAULT_OUTPUT_DIR
EOF
}

args=()
for arg in "$@"; do
  case $arg in
    -h|--help)
      usage
      exit 0
      ;;
    --force-precompute)
      FORCE_PRECOMPUTE="--force-precompute"
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

CANDIDATES="${args[0]:-$DEFAULT_CANDIDATES}"
OUTPUT_DIR="${args[1]:-$DEFAULT_OUTPUT_DIR}"
OUTPUT_CSV="$OUTPUT_DIR/submission.csv"

printf "Running full pipeline from start to finish...\n"
printf "Candidates: %s\n" "$CANDIDATES"
printf "Output directory: %s\n" "$OUTPUT_DIR"
printf "Log file: %s\n" "$DEFAULT_LOG"
[[ -n "$FORCE_PRECOMPUTE" ]] && printf "Force Precompute: ON\n"
printf "\n"

mkdir -p "$OUTPUT_DIR"

python3 backend/rank.py --candidates "$CANDIDATES" --out "$OUTPUT_CSV" $FORCE_PRECOMPUTE 2>&1 | tee "$DEFAULT_LOG"

printf "\nPipeline complete. Output written to %s\n" "$OUTPUT_CSV"
printf "Validating generated ranking output...\n"
python3 backend/analyze_results.py 2>&1 | tee -a "$DEFAULT_LOG"

printf "\nDone. Logs saved to %s\n" "$DEFAULT_LOG"
