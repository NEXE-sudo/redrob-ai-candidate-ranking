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

from engines.candidate_profile_parser import CandidateProfileParser
from engines.feature_scorer import FeatureScorer
from engines.advanced_scorer import AdvancedScorer
from engines.embedding_retrieval import BM25Retriever


class BenchmarkingEngine:
    """Compare 3 ranking approaches on the same data"""
    
    def __init__(self):
        self.parser = CandidateProfileParser()
        self.feature_scorer = FeatureScorer(self.parser)
        self.advanced_scorer = AdvancedScorer(self.parser)
        self.bm25_retriever = BM25Retriever()
    
    def run_benchmark(
        self,
        jsonl_path: str,
        jd_text: str,
        sample_size: int = 1000,
        top_k: int = 100
    ) -> Dict[str, any]:
        """Run BM25 benchmark (EmbeddingRetriever deprecated, use EmbeddingPrecomputer via OptimizedRankingEngine)"""
        
        print("\n" + "="*70)
        print("BM25 RETRIEVAL BENCHMARK")
        print("="*70)
        print(f"Sample Size: {sample_size}")
        print(f"Top K: {top_k}")
        print("\nNote: Embeddings-based approaches now use EmbeddingPrecomputer in OptimizedRankingEngine")
        
        # Load sample candidates
        print(f"\n[1/3] Loading sample from {jsonl_path}...")
        candidates = self._load_sample(jsonl_path, sample_size)
        print(f"  Loaded {len(candidates)} candidates")
        
        # Build indices
        print(f"\n[2/3] Building BM25 index...")
        t0 = datetime.now()
        self.bm25_retriever.build_index(candidates)
        bm25_time = (datetime.now() - t0).total_seconds()
        
        print(f"  BM25 index: {bm25_time:.2f}s")
        
        # Run BM25 approach only
        print(f"\n[3/3] Running BM25 retrieval...")
        
        results = {
            'bm25_only': self._rank_bm25_only(candidates, jd_text, top_k),
        }
        
        # Print summary
        self._print_summary(results, {})
        
        
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
    
    def _analyze_results(
        self,
        results: Dict[str, List],
        top_k: int
    ) -> Dict:
        """Analyze BM25 results"""
        
        # Score distributions
        bm25_scores = np.array([score for _, score in results['bm25_only'][:top_k]])
        
        return {
            'score_distribution': {
                'bm25': {
                    'mean': float(np.mean(bm25_scores)),
                    'std': float(np.std(bm25_scores)),
                    'min': float(np.min(bm25_scores)),
                    'max': float(np.max(bm25_scores))
                }
            }
        }
    
    def _print_summary(self, results: Dict, analysis: Dict):
        """Print benchmark summary"""
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        
        print("\n[Score Distribution]")
        if 'score_distribution' in analysis and 'bm25' in analysis['score_distribution']:
            dist = analysis['score_distribution']['bm25']
            print(f"\n  BM25:")
            print(f"    Mean: {dist['mean']:.4f}")
            print(f"    Std:  {dist['std']:.4f}")
            print(f"    Min:  {dist['min']:.4f}")
            print(f"    Max:  {dist['max']:.4f}")
        
        print("\n[Top 10 Results]")
        print("\n  BM25 Only:")
        for rank, (id, score) in enumerate(results['bm25_only'][:10], 1):
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
