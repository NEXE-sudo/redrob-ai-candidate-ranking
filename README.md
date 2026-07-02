# Redrob Candidate Ranker

This repository contains a candidate ranking pipeline for the Redrob challenge. It supports end-to-end candidate ranking with embedding-based retrieval, BM25 filtering, FAISS semantic retrieval, cross-encoder reranking, and result export for submission.

## What this project does

- Loads candidate profiles from the provided JSONL dataset.
- Builds or reuses precomputed embeddings for semantic matching.
- Runs a multi-stage ranking pipeline:
  - lexical filtering with BM25
  - semantic retrieval with FAISS
  - reranking with a cross-encoder
  - final scoring and tie-breaking
- Writes ranked output to `ranking_output/NEXE-sudo.csv` and `ranking_output/ranking_detailed.json`.

## Repository layout

- `backend/` — core ranking pipeline implementation.
  - `backend/rank.py` — main CLI entry point for running the pipeline.
  - `backend/analyze_results.py` — validation and analysis of generated outputs.
  - `backend/benchmark_ranking.py` and `backend/run_benchmark_with_jd.py` — benchmarking and evaluation helpers.
  - `backend/engines/` — scoring and ranking components.
    - `advanced_scorer.py` and `advanced_scoring_components.py` — feature and scoring logic.
    - `candidate_profile_parser.py` — parsing candidate fields and profile data.
    - `cross_encoder_reranker.py` — reranking with the cross-encoder model.
    - `embedding_precompute.py` — batching and caching embedding precomputation.
    - `embedding_retrieval.py` — FAISS and embedding retrieval logic.
    - `feature_scorer.py` — candidate feature scoring.
    - `optimized_ranking_engine.py` — orchestrates the multi-stage ranking flow.
    - `recruiter_jd_parser.py` — job description parsing.
    - `text_utils.py` — text normalization and processing utilities.
  - `backend/models/` — local model weights and tokenizer assets.
  - `backend/embeddings_cache/` — cached embeddings, metadata, ids, and FAISS index.
- `app.py` — Streamlit sandbox app that can run the ranking pipeline from the browser.
- `run_full_pipeline.sh` — convenience wrapper for running the full dataset end to end.
- `scripts/` — utility scripts for dataset inspection, comparison, and testing.
- `[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/` — challenge dataset, candidate schema, sample outputs, and submission metadata template.
- `ranking_output/` — generated submission CSV and detailed ranking JSON from pipeline runs.
- `submission.csv`, `submission_metadata.yaml`, and `submission_spec.txt` — submission-oriented files for the challenge.
- `test_ranking_components.py` and other test helpers under `backend/tests/` and `scripts/` — verification and regression checks.

## Running locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

The default full-dataset run uses the 100K candidate file in the challenge folder:

```bash
bash run_full_pipeline.sh
```

To skip precomputation and reuse the existing embedding cache:

```bash
bash run_full_pipeline.sh --skip-precompute
```

To force a fresh precompute:

```bash
bash run_full_pipeline.sh --force-precompute
```

### 4. Run the ranking entrypoint directly

```bash
python3 backend/rank.py \
  --candidates "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --out ranking_output/NEXE-sudo.csv
```

## Streamlit app

The sandbox app can be launched with:

```bash
streamlit run app.py
```

It can run either on a custom uploaded JSONL file or, when present, on the full 100K dataset in the repository.

## Outputs

After a successful run, the following files are produced:

- `ranking_output/NEXE-sudo.csv` — ranked submission CSV.
- `ranking_output/ranking_detailed.json` — detailed ranking breakdown.
- `backend/embeddings_cache/` — cached embeddings, IDs, and FAISS index.

## Notes

- The first full-dataset run may take a while because embeddings are precomputed and cached.
- The cache can be reused on later runs with `--skip-precompute`.
- The repository is also prepared for deployment as a Hugging Face Space.
