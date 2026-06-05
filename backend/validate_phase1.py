#!/usr/bin/env python3
"""
PHASE 1: Full Dataset Execution with Metrics
Runs complete ranking pipeline and measures performance
"""
import sys
import json
import time
import os
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engines.optimized_ranking_engine import OptimizedRankingEngine

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

CANDIDATES_FILE = "/home/NEXE/projects/Redrob hackathon/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"

def get_memory_mb():
    """Get current process memory in MB"""
    try:
        with open(f'/proc/{os.getpid()}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
        return 0
    except:
        return 0

def run_phase1():
    print("\n" + "="*80)
    print("PHASE 1: FULL DATASET EXECUTION WITH METRICS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    metrics = {"phase": "1", "timestamp": datetime.now().isoformat()}
    
    try:
        initial_mem = get_memory_mb()
        print(f"Initial memory: {initial_mem:.2f} MB")
        
        # Initialize
        print("\n[1/5] Initializing...")
        t0 = time.time()
        engine = OptimizedRankingEngine(use_precomputed_embeddings=True)
        init_time = time.time() - t0
        init_mem = get_memory_mb()
        print(f"  ✓ {init_time:.2f}s | Memory: {init_mem:.2f} MB (+{init_mem-initial_mem:.2f})")
        metrics["init_time"] = init_time
        
        # Load
        print("\n[2/5] Loading candidates...")
        t0 = time.time()
        engine.load_candidates(CANDIDATES_FILE)
        load_time = time.time() - t0
        load_mem = get_memory_mb()
        n_cands = len(engine.candidates)
        print(f"  ✓ {load_time:.2f}s | {n_cands:,} candidates | Memory: {load_mem:.2f} MB (+{load_mem-init_mem:.2f})")
        metrics["load_time"] = load_time
        metrics["candidates_loaded"] = n_cands
        
        # Prepare JD
        print("\n[3/5] Preparing JD...")
        t0 = time.time()
        engine.prepare_jd_text(JD_TEXT)
        jd_time = time.time() - t0
        jd_mem = get_memory_mb()
        print(f"  ✓ {jd_time:.2f}s | Memory: {jd_mem:.2f} MB (+{jd_mem-load_mem:.2f})")
        metrics["jd_time"] = jd_time
        
        # Rank
        print("\n[4/5] Ranking pipeline...")
        t0 = time.time()
        results, scored_candidates = engine.rank_candidates_fast(
            top_k=100, bm25_top_k=2000, faiss_top_k=500
        )
        rank_time = time.time() - t0
        peak_mem = get_memory_mb()
        print(f"  ✓ {rank_time:.2f}s | Memory: {peak_mem:.2f} MB")
        metrics["rank_time"] = rank_time
        metrics["peak_memory"] = peak_mem
        
        # Save
        print("\n[5/5] Saving results...")
        t0 = time.time()
        engine.save_results(results, output_dir='./ranking_output')
        save_time = time.time() - t0
        print(f"  ✓ {save_time:.2f}s")
        metrics["save_time"] = save_time
        
        total_time = init_time + load_time + jd_time + rank_time + save_time
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Total candidates: {n_cands:,}")
        print(f"Top 100 extracted: {len(results)}")
        print(f"\nTimings:")
        print(f"  Init:   {init_time:6.2f}s")
        print(f"  Load:   {load_time:6.2f}s ({load_time/total_time*100:5.1f}%)")
        print(f"  JD:     {jd_time:6.2f}s ({jd_time/total_time*100:5.1f}%)")
        print(f"  Rank:   {rank_time:6.2f}s ({rank_time/total_time*100:5.1f}%)")
        print(f"  Save:   {save_time:6.2f}s ({save_time/total_time*100:5.1f}%)")
        print(f"  TOTAL:  {total_time:6.2f}s")
        print(f"\nMemory:")
        print(f"  Initial: {initial_mem:6.2f} MB")
        print(f"  Peak:    {peak_mem:6.2f} MB (+{peak_mem-initial_mem:.2f} MB)")
        
        print(f"\nTop 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r['candidate_id'][:20]:20s} Score: {r['final_score']:.4f}")
        
        metrics["total_time"] = total_time
        metrics["memory_increase"] = peak_mem - initial_mem
        
        os.makedirs('./ranking_output', exist_ok=True)
        with open('./ranking_output/phase1_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\n✓ Phase 1 complete")
        return metrics
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_phase1()
