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
import builtins
import json
import sys
from pathlib import Path
from datetime import datetime


def _silence_output(*_args, **_kwargs):
    return None


builtins.print = _silence_output

backend_dir = Path(__file__).resolve().parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl"
    )
    default_output_dir = Path(__file__).resolve().parents[1] / "ranking_output"
    default_output_csv = default_output_dir / "submission.csv"
    parser.add_argument(
        "--out", default=str(default_output_csv),
        help=f"Output CSV path (default: {default_output_csv})"
    )
    parser.add_argument(
        "--download-models", action="store_true",
        help="Download embedding and cross-encoder models locally and exit."
    )
    parser.add_argument(
        "--job-desc", default=None,
        help="Path to a job description text file. If unset, uses built-in JD_TEXT."
    )
    parser.add_argument(
        "--bm25-top-k", type=int, default=5000,
        help="BM25 retrieval pool size for semantic reranking."
    )
    parser.add_argument(
        "--faiss-top-k", type=int, default=1000,
        help="FAISS semantic retrieval top-k after BM25 filtering."
    )
    parser.add_argument(
        "--cross-encoder-top-k", type=int, default=1000,
        help="Cross-encoder reranking top-k candidates. Set 0 to disable."
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
    parser.add_argument(
        "--force-precompute", action="store_true",
        help="Force embedding precomputation even if a compatible cache exists"
    )
    args = parser.parse_args()

    from engines.optimized_ranking_engine import OptimizedRankingEngine
    from engines.embedding_precompute import EmbeddingPrecomputer

    JD_TEXT = """
Senior AI Engineer - Ranking & Recommendation Systems

We are looking for a Senior AI Engineer with expertise in ranking systems, retrieval augmented generation (RAG), and recommendation algorithms to join our elite team.

Key Responsibilities:
- Design and implement large-scale ranking engines for ML-based candidate matching
- Build semantic retrieval systems using embeddings (FAISS, Pinecone, Milvus)
- Implement learning-to-rank (LTR) models with NDCG/MRR optimization
- Lead A/B testing and online evaluation frameworks
- Optimize end-to-end ML pipelines for production systems
- Mentor junior engineers on ranking architecture and evaluation frameworks

Required Experience:
- 5-9 years of ML/AI engineering experience
- Deep expertise in ranking systems, retrieval, and recommendation
- Production experience with embedding models and vector databases
- Strong Python skills with ML frameworks (PyTorch, TensorFlow, scikit-learn)
- Proven track record in shipping ML systems at scale
- Experience with evaluation metrics (NDCG, MRR, MAP)

Preferred:
- Startup or product company experience
- Published research or open-source contributions
- Experience with LLMs and RAG systems
- Background in information retrieval and NLP

What We Offer:
- Early-stage startup environment with equity
- Work on cutting-edge ranking and retrieval challenges
- Technical leadership opportunities
- Competitive compensation and benefits
"""

    cache_dir = Path(args.cache_dir)
    embeddings_file = cache_dir / "precomputed_embeddings_embeddings.npy"
    metadata_file = cache_dir / "precomputed_embeddings_metadata.json"

    if args.download_models:
        from scripts.download_models import download_models
        download_models()
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

    if not args.skip_precompute and (args.force_precompute or not _cache_is_compatible()):
        precomputer.precompute_embeddings(
            jsonl_path=args.candidates,
            output_prefix="precomputed_embeddings"
        )
    engine = OptimizedRankingEngine(
        embeddings_cache_dir=str(cache_dir),
        use_precomputed_embeddings=True,
        embedding_model='sentence-transformers/all-mpnet-base-v2',
        enable_cross_encoder=True,
        enable_honeypot_detection=True
    )

    engine.load_candidates(args.candidates)

    if args.job_desc:
        job_desc_path = Path(args.job_desc)
        if not job_desc_path.exists():
            raise FileNotFoundError(f"Job description file not found: {job_desc_path}")
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            job_desc_text = f.read().strip()
        engine.prepare_jd_text(job_desc_text)
    else:
        engine.prepare_jd_text(JD_TEXT)

    results, _ = engine.rank_candidates_fast(
        top_k=100,
        bm25_top_k=args.bm25_top_k,
        faiss_top_k=args.faiss_top_k,
        cross_encoder_top_k=args.cross_encoder_top_k
    )

    requested_output = Path(args.out)
    if requested_output.suffix.lower() == ".csv":
        output_dir = requested_output.parent
        output_path = requested_output
    else:
        output_dir = requested_output
        output_path = output_dir / "submission.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    engine.save_results(results, output_dir=str(output_dir))

    for duplicate_path in repo_root.rglob("submission.csv"):
        if duplicate_path.resolve() != output_path.resolve():
            duplicate_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
