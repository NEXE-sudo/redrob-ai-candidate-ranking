#!/usr/bin/env python3
"""
Fast Multi-Stage Ranking Pipeline
Optimized for <5 minute execution on 100K candidates

Prerequisites:
1. Run precompute_embeddings.py first (one-time setup)
2. Ensure precomputed embeddings are cached in ./embeddings_cache/
"""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.engines.optimized_ranking_engine import OptimizedRankingEngine


# Full Senior AI Engineer job description
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


def run_ranking_pipeline(candidates_file: str):
    """Execute fast ranking pipeline"""
    
    print("\n" + "="*80)
    print("OPTIMIZED RANKING PIPELINE - SENIOR AI ENGINEER ROLE")
    print("="*80)
    print(f"Candidates: {candidates_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize engine
        print("\n[Init] Initializing ranking engine...")
        engine = OptimizedRankingEngine(use_precomputed_embeddings=True)
        
        # Load data
        print("[Load] Loading candidates...")
        engine.load_candidates(candidates_file)
        engine.prepare_jd_text(JD_TEXT)
        
        # Run ranking
        print("[Rank] Executing multi-stage ranking pipeline...")
        results, scored_candidates = engine.rank_candidates_fast(
            top_k=100,
            bm25_top_k=2000,
            faiss_top_k=500
        )
        
        # Save results
        print("[Save] Saving results...")
        engine.save_results(results, output_dir='./ranking_output')
        
        print("\n" + "="*80)
        print("RANKING COMPLETE ✓")
        print("="*80)
        print(f"Top 5 Results:")
        for result in results[:5]:
            print(f"  {result['rank']}. {result['candidate_id']} - {result['final_score']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 run_ranking_optimized.py <path_to_candidates.jsonl>")
        sys.exit(1)
    
    candidates_file = sys.argv[1]
    sys.exit(run_ranking_pipeline(candidates_file))
