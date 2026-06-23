#!/usr/bin/env python3
"""
Single entry point for the Redrob ranking pipeline.
Handles both precomputation (if needed) and ranking in one command.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This satisfies the competition requirement:
    reproduce_command: "python backend/rank.py --candidates ./candidates.jsonl --out ./submission.csv"
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl"
    )
    default_output_csv = str(Path(__file__).resolve().parents[1] / "submission.csv")
    parser.add_argument(
        "--out", default=default_output_csv,
        help=f"Output CSV path (default: {default_output_csv})"
    )
    parser.add_argument(
        "--download-models", action="store_true",
        help="Download embedding and cross-encoder models locally and exit."
    )
    default_cache_dir = str(Path(__file__).resolve().parent / "embeddings_cache")
    parser.add_argument(
        "--cache-dir", default=default_cache_dir,
        help=f"Embeddings cache directory (default: {default_cache_dir})"
    )
    parser.add_argument(
        "--skip-precompute", action="store_true",
        help="Skip embedding precomputation (use if cache already exists)"
    )
    args = parser.parse_args()

    print(f"\nRedrob Ranker — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Candidates: {args.candidates}")
    print(f"Output: {args.out}")

    from engines.optimized_ranking_engine import OptimizedRankingEngine
    from engines.embedding_precompute import EmbeddingPrecomputer
    from run_ranking_optimized import JD_TEXT

    cache_dir = Path(args.cache_dir)
    embeddings_file = cache_dir / "precomputed_embeddings_embeddings.npy"
    metadata_file = cache_dir / "precomputed_embeddings_metadata.json"

    if args.download_models:
        from scripts.download_models import download_models
        download_models()
        print("\n✓ Models downloaded locally. Exiting.")
        sys.exit(0)

    precomputer = EmbeddingPrecomputer(cache_dir=str(cache_dir))

    def _cache_is_compatible() -> bool:
        if not embeddings_file.exists() or not metadata_file.exists():
            return False
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return (
                metadata.get('model_name') == precomputer.model_name
                and metadata.get('embedding_dim') == precomputer.embedding_dim
            )
        except Exception:
            return False

    if not args.skip_precompute and not _cache_is_compatible():
        if embeddings_file.exists():
            print("\n[Step 1/2] Existing embeddings cache is incompatible or built with a different model. Recomputing embeddings...")
        else:
            print("\n[Step 1/2] Precomputing embeddings (one-time setup)...")
        precomputer.precompute_embeddings(
            jsonl_path=args.candidates,
            output_prefix="precomputed_embeddings"
        )
        print("[Step 1/2] ✓ Embeddings cached")
    else:
        print("\n[Step 1/2] Embeddings cache found and compatible — skipping precomputation")

    print("\n[Step 2/2] Running ranking pipeline...")
    engine = OptimizedRankingEngine(
        embeddings_cache_dir=str(cache_dir),
        use_precomputed_embeddings=True,
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        enable_cross_encoder=True,
        enable_honeypot_detection=True
    )

    engine.load_candidates(args.candidates)
    engine.prepare_jd_text(JD_TEXT)

    results, _ = engine.rank_candidates_fast(
        top_k=100,
        bm25_top_k=5000,
        faiss_top_k=1000,
        cross_encoder_top_k=1000
    )

    out_path = Path(args.out)
    out_dir = str(out_path.parent)
    engine.save_results(results, output_dir=out_dir)

    default_csv = Path(out_dir) / "submission.csv"
    if default_csv != out_path and default_csv.exists():
        default_csv.rename(out_path)

    print(f"\n✓ Done. Submission written to {args.out}")
    print(f"  Validate with: python validate_submission.py {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
