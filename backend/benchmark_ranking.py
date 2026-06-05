"""
Benchmarking Script
Compare ranking approaches: BM25 only vs Embeddings only vs Hybrid

Tests on a sample to validate approach diversity and scoring quality.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.engines.candidate_profile_parser import CandidateProfileParser
from app.engines.feature_scorer import FeatureScorer
from app.engines.advanced_scorer import AdvancedScorer
from app.engines.embedding_retrieval import BM25Retriever, EmbeddingRetriever


class BenchmarkingEngine:
    """Compare 3 ranking approaches on the same data"""
    
    def __init__(self):
        self.parser = CandidateProfileParser()
        self.feature_scorer = FeatureScorer(self.parser)
        self.advanced_scorer = AdvancedScorer(self.parser)
        self.bm25_retriever = BM25Retriever()
        self.embedding_retriever = EmbeddingRetriever()
    
    def run_benchmark(
        self,
        jsonl_path: str,
        jd_text: str,
        sample_size: int = 1000,
        top_k: int = 100
    ) -> Dict[str, any]:
        """Run benchmark comparing 3 approaches"""
        
        print("\n" + "="*70)
        print("RANKING APPROACH BENCHMARK")
        print("="*70)
        print(f"Sample Size: {sample_size}")
        print(f"Top K: {top_k}")
        
        # Load sample candidates
        print(f"\n[1/4] Loading sample from {jsonl_path}...")
        candidates = self._load_sample(jsonl_path, sample_size)
        print(f"  Loaded {len(candidates)} candidates")
        
        # Build indices
        print(f"\n[2/4] Building retrieval indices...")
        t0 = datetime.now()
        self.bm25_retriever.build_index(candidates)
        bm25_time = (datetime.now() - t0).total_seconds()
        
        t0 = datetime.now()
        embeddings, ids = self.embedding_retriever.build_index(candidates)
        faiss_time = (datetime.now() - t0).total_seconds()
        
        print(f"  BM25 index: {bm25_time:.2f}s")
        print(f"  Embeddings+FAISS: {faiss_time:.2f}s")
        
        # Run 3 approaches
        print(f"\n[3/4] Running 3 ranking approaches...")
        
        results = {
            'bm25_only': self._rank_bm25_only(candidates, jd_text, top_k),
            'embeddings_only': self._rank_embeddings_only(candidates, jd_text, top_k),
            'hybrid': self._rank_hybrid(candidates, jd_text, top_k)
        }
        
        # Analyze results
        print(f"\n[4/4] Analyzing results...")
        analysis = self._analyze_results(results, top_k)
        
        # Print summary
        self._print_summary(results, analysis)
        
        return {
            'results': results,
            'analysis': analysis,
            'config': {
                'sample_size': sample_size,
                'top_k': top_k,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _load_sample(self, jsonl_path: str, max_n: int) -> List[Dict]:
        """Load sample of candidates"""
        candidates = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if len(candidates) >= max_n:
                    break
                try:
                    candidates.append(json.loads(line))
                except:
                    continue
        return candidates
    
    def _rank_bm25_only(
        self,
        candidates: List[Dict],
        jd_text: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Rank using BM25 only"""
        
        t0 = datetime.now()
        ids, scores = self.bm25_retriever.retrieve(jd_text, top_k=top_k)
        elapsed = (datetime.now() - t0).total_seconds()
        
        print(f"  BM25 only: {len(ids)} results in {elapsed:.2f}s")
        return [(id, score) for id, score in zip(ids, scores)]
    
    def _rank_embeddings_only(
        self,
        candidates: List[Dict],
        jd_text: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Rank using embeddings only"""
        
        t0 = datetime.now()
        ids, scores = self.embedding_retriever.retrieve(jd_text, top_k=top_k)
        elapsed = (datetime.now() - t0).total_seconds()
        
        print(f"  Embeddings only: {len(ids)} results in {elapsed:.2f}s")
        return [(id, score) for id, score in zip(ids, scores)]
    
    def _rank_hybrid(
        self,
        candidates: List[Dict],
        jd_text: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Rank using hybrid (BM25 + embeddings + features)"""
        
        t0 = datetime.now()
        
        # Stage 1: BM25 → 2000
        bm25_ids, bm25_scores = self.bm25_retriever.retrieve(jd_text, top_k=2000)
        
        # Stage 2: Embeddings from pool → 500
        embeddings_ids, embeddings_scores = self.embedding_retriever.retrieve(jd_text, top_k=500)
        embeddings_set = set(embeddings_ids[:500])
        
        # Score hybrid pool (combine both stages)
        pool_ids = set(bm25_ids[:2000]) | embeddings_set
        pool_ids = list(pool_ids)[:top_k]
        
        # Simple hybrid score: average of normalized ranks
        hybrid_scores = []
        for id in pool_ids:
            bm25_rank = bm25_ids.index(id) + 1 if id in bm25_ids else len(bm25_ids) + 1
            emb_rank = embeddings_ids.index(id) + 1 if id in embeddings_ids else len(embeddings_ids) + 1
            hybrid_score = 1.0 / (0.5 * (bm25_rank / len(bm25_ids)) + 0.5 * (emb_rank / len(embeddings_ids)))
            hybrid_scores.append(hybrid_score)
        
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Hybrid (BM25+Embeddings): {len(pool_ids)} results in {elapsed:.2f}s")
        
        return [(id, score) for id, score in zip(pool_ids, hybrid_scores)]
    
    def _analyze_results(
        self,
        results: Dict[str, List],
        top_k: int
    ) -> Dict:
        """Analyze differences between approaches"""
        
        # Extract IDs
        bm25_ids = set(id for id, _ in results['bm25_only'][:top_k])
        emb_ids = set(id for id, _ in results['embeddings_only'][:top_k])
        hybrid_ids = set(id for id, _ in results['hybrid'][:top_k])
        
        # Compute overlaps
        overlap_bm25_emb = len(bm25_ids & emb_ids) / top_k
        overlap_bm25_hybrid = len(bm25_ids & hybrid_ids) / top_k
        overlap_emb_hybrid = len(emb_ids & hybrid_ids) / top_k
        
        # Score distributions
        bm25_scores = np.array([score for _, score in results['bm25_only'][:top_k]])
        emb_scores = np.array([score for _, score in results['embeddings_only'][:top_k]])
        hybrid_scores = np.array([score for _, score in results['hybrid'][:top_k]])
        
        return {
            'overlap': {
                'bm25_embeddings': float(overlap_bm25_emb),
                'bm25_hybrid': float(overlap_bm25_hybrid),
                'embeddings_hybrid': float(overlap_emb_hybrid)
            },
            'score_distribution': {
                'bm25': {
                    'mean': float(np.mean(bm25_scores)),
                    'std': float(np.std(bm25_scores)),
                    'min': float(np.min(bm25_scores)),
                    'max': float(np.max(bm25_scores))
                },
                'embeddings': {
                    'mean': float(np.mean(emb_scores)),
                    'std': float(np.std(emb_scores)),
                    'min': float(np.min(emb_scores)),
                    'max': float(np.max(emb_scores))
                },
                'hybrid': {
                    'mean': float(np.mean(hybrid_scores)),
                    'std': float(np.std(hybrid_scores)),
                    'min': float(np.min(hybrid_scores)),
                    'max': float(np.max(hybrid_scores))
                }
            }
        }
    
    def _print_summary(self, results: Dict, analysis: Dict):
        """Print benchmark summary"""
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        
        print("\n[Overlap Analysis]")
        print(f"  BM25 vs Embeddings: {analysis['overlap']['bm25_embeddings']*100:.1f}% overlap")
        print(f"  BM25 vs Hybrid: {analysis['overlap']['bm25_hybrid']*100:.1f}% overlap")
        print(f"  Embeddings vs Hybrid: {analysis['overlap']['embeddings_hybrid']*100:.1f}% overlap")
        
        print("\n[Score Distribution]")
        for approach in ['bm25', 'embeddings', 'hybrid']:
            dist = analysis['score_distribution'][approach]
            print(f"\n  {approach.upper()}:")
            print(f"    Mean: {dist['mean']:.4f}")
            print(f"    Std:  {dist['std']:.4f}")
            print(f"    Min:  {dist['min']:.4f}")
            print(f"    Max:  {dist['max']:.4f}")
        
        print("\n[Top 5 Results Comparison]")
        print("\n  BM25 Only:")
        for rank, (id, score) in enumerate(results['bm25_only'][:5], 1):
            print(f"    {rank}. {id} ({score:.4f})")
        
        print("\n  Embeddings Only:")
        for rank, (id, score) in enumerate(results['embeddings_only'][:5], 1):
            print(f"    {rank}. {id} ({score:.4f})")
        
        print("\n  Hybrid:")
        for rank, (id, score) in enumerate(results['hybrid'][:5], 1):
            print(f"    {rank}. {id} ({score:.4f})")
        
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 benchmark_ranking.py <candidates.jsonl> <jd_text>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    jd_text = sys.argv[2]
    
    engine = BenchmarkingEngine()
    results = engine.run_benchmark(jsonl_path, jd_text, sample_size=1000, top_k=100)
    
    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to benchmark_results.json")
