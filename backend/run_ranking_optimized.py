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
sys.path.insert(0, str(Path(__file__).parent))

from engines.optimized_ranking_engine import OptimizedRankingEngine


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
    """
    Execute enhanced ranking pipeline with all 10 phases.
    
    Phases:
    1. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
    2. Recruiter-centric JD parsing
    3. Advanced scoring components
    4. Honeypot detection
    5. Behavioral signals
    6. Rebalanced scoring weights
    7. Tie-breaking compliance
    8. Improved reasoning generation
    9. Benchmarking telemetry
    10. Configurable embedding models
    """
    
    print("\n" + "="*80)
    print("ENHANCED RANKING PIPELINE - ALL 10 PHASES ENABLED")
    print("="*80)
    print(f"Candidates: {candidates_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize engine with all phases enabled (Phase 10: configurable model)
        print("\n[Init] Initializing enhanced ranking engine...")
        engine = OptimizedRankingEngine(
            use_precomputed_embeddings=True,
            embedding_model='BAAI/bge-small-en-v1.5',  # Phase 10: Configurable
            enable_cross_encoder=True,  # Phase 1: Enable cross-encoder
            enable_honeypot_detection=True  # Phase 4: Enable honeypot detection
        )
        
        # Load data
        print("[Load] Loading candidates...")
        engine.load_candidates(candidates_file)
        engine.prepare_jd_text(JD_TEXT)  # Phase 2: Recruiter-centric parsing
        
        # Run ranking with all phases (Phase 1: new thresholds, Phase 3-4: new scoring)
        print("[Rank] Executing enhanced multi-stage pipeline (Phases 1-10)...")
        results, scored_candidates = engine.rank_candidates_fast(
            top_k=100,
            bm25_top_k=3000,  # Phase 1: Increased
            faiss_top_k=1000,  # Phase 1: Increased  
            cross_encoder_top_k=250  # Phase 1: Cross-encoder reranking
        )
        
        # Save results with Phase 7 tie-breaking compliance
        print("[Save] Saving results with tie-breaking compliance...")
        output_dir = Path(__file__).resolve().parents[1] / 'ranking_output'
        engine.save_results(results, output_dir=str(output_dir))
        
        # Phase 9: Print detailed benchmarking report
        print("\n" + "="*80)
        print("ENHANCED RANKING COMPLETE ✓")
        print("="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nPhases Executed:")
        print(f"  ✓ Phase 1: Cross-encoder reranking")
        print(f"  ✓ Phase 2: Recruiter-centric JD parsing")
        print(f"  ✓ Phase 3: Advanced scoring components")
        print(f"  ✓ Phase 4: Honeypot detection")
        print(f"  ✓ Phase 5: Behavioral signals (feature scorer)")
        print(f"  ✓ Phase 6: Rebalanced scoring weights")
        print(f"  ✓ Phase 7: Tie-breaking compliance")
        print(f"  ✓ Phase 8: Improved reasoning")
        print(f"  ✓ Phase 9: Benchmarking telemetry")
        print(f"  ✓ Phase 10: Configurable embeddings")
        
        if engine.benchmark_telemetry:
            print(f"\nBenchmarking Results:")
            for stage, seconds in engine.benchmark_telemetry.items():
                if stage != 'total':
                    print(f"  {stage:20s}: {seconds:6.2f}s")
            print(f"  {'TOTAL':20s}: {engine.benchmark_telemetry.get('total', 0):6.2f}s")
        
        print(f"\nTop 5 Results:")
        for result in results[:5]:
            print(f"  {result['rank']}. {result['candidate_id']} - {result['final_score']:.4f}")
            print(f"     {result.get('reasoning', 'N/A')[:80]}...")
        
        print("="*80)
        
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
