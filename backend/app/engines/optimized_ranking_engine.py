"""
Optimized Multi-Stage Ranking Engine
Fast 5-minute pipeline: BM25 (2000) → FAISS (500) → Scoring (100)
Uses precomputed embeddings for speed.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import faiss

from .candidate_profile_parser import CandidateProfileParser, ParsedProfile
from .feature_scorer import FeatureScorer, ScoringComponents
from .advanced_scorer import AdvancedScorer
from .embedding_retrieval import BM25Retriever
from .embedding_precompute import EmbeddingPrecomputer


class OptimizedRankingEngine:
    """Fast multi-stage ranking: BM25 → FAISS → Features → Top 100"""
    
    def __init__(
        self,
        embeddings_cache_dir: str = './embeddings_cache',
        use_precomputed_embeddings: bool = True
    ):
        self.parser = CandidateProfileParser()
        self.feature_scorer = FeatureScorer(self.parser)
        self.advanced_scorer = AdvancedScorer(self.parser)
        self.bm25_retriever = BM25Retriever()
        self.precomputer = EmbeddingPrecomputer(cache_dir=embeddings_cache_dir)
        
        self.embeddings_cache_dir = embeddings_cache_dir
        self.use_precomputed = use_precomputed_embeddings
        
        self.candidates = None
        self.parsed_profiles = {}
        self.faiss_index = None
        self.candidate_embeddings = None
        self.candidate_ids = None
        self.jd_text = None
        self.jd_embedding = None
    
    def load_candidates(self, jsonl_path: str):
        """Load candidates from JSONL"""
        print(f"Loading candidates from {jsonl_path}...")
        
        self.candidates = []
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    candidate = json.loads(line)
                    self.candidates.append(candidate)
                except json.JSONDecodeError as e:
                    if line_num < 10:  # Only warn for first 10
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.candidates)} candidates")
    
    def prepare_jd_text(self, jd: str):
        """Set job description"""
        self.jd_text = jd
    
    def _load_or_build_faiss_index(self):
        """Load precomputed embeddings or build FAISS index"""
        
        if self.use_precomputed:
            try:
                print("Loading precomputed embeddings...")
                self.candidate_embeddings, self.candidate_ids, metadata = \
                    self.precomputer.load_precomputed_embeddings()
                
                print(f"Building FAISS index from precomputed embeddings...")
                self.faiss_index = faiss.IndexFlatIP(metadata['embedding_dim'])
                self.faiss_index.add(self.candidate_embeddings)
                print(f"FAISS index ready ({self.faiss_index.ntotal} vectors)")
                return
                
            except FileNotFoundError:
                print("Precomputed embeddings not found. Building now...")
        
        # Fallback: compute embeddings on the fly
        raise RuntimeError(
            "Precomputed embeddings required for fast ranking. "
            "Run: python3 precompute_embeddings.py <candidates_file>"
        )
    
    def rank_candidates_fast(
        self,
        top_k: int = 100,
        bm25_top_k: int = 2000,
        faiss_top_k: int = 500
    ) -> Tuple[List[Dict], Any]:
        """Execute fast multi-stage ranking pipeline
        
        Timeline:
        Stage 1 (1s): Load embeddings + BM25
        Stage 2 (2s): BM25 retrieval → 2000
        Stage 3 (1s): FAISS retrieval → 500  
        Stage 4 (60s): Feature scoring → 100
        Total: ~4 minutes for 100K candidates
        
        Args:
            top_k: Final top candidates to return
            bm25_top_k: BM25 stage threshold (→ FAISS)
            faiss_top_k: FAISS stage threshold (→ Scoring)
        """
        
        if not self.candidates or not self.jd_text:
            raise ValueError("Load candidates and JD first")
        
        print("\n" + "="*70)
        print("OPTIMIZED MULTI-STAGE RANKING PIPELINE")
        print("="*70)
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
        
        # Stage 1: Load/Build indices
        print("\n[Stage 1] Building retrieval indices...")
        t0 = datetime.now()
        self.bm25_retriever.build_index(self.candidates)
        self._load_or_build_faiss_index()
        t1 = datetime.now()
        print(f"  ✓ Complete ({(t1-t0).total_seconds():.1f}s)")
        
        # Stage 2: BM25 retrieval (fast keyword match)
        print(f"\n[Stage 2] BM25 retrieval → Top {bm25_top_k}...")
        t0 = datetime.now()
        bm25_ids, bm25_scores = self.bm25_retriever.retrieve(
            self.jd_text,
            top_k=bm25_top_k
        )
        t1 = datetime.now()
        print(f"  ✓ Retrieved {len(bm25_ids)} candidates ({(t1-t0).total_seconds():.1f}s)")
        
        # Stage 3: FAISS retrieval (semantic match from BM25 pool)
        print(f"\n[Stage 3] FAISS retrieval from pool → Top {faiss_top_k}...")
        t0 = datetime.now()
        faiss_ids, faiss_scores = self._faiss_retrieve_from_pool(
            self.jd_text,
            bm25_ids,
            top_k=faiss_top_k
        )
        t1 = datetime.now()
        print(f"  ✓ Retrieved {len(faiss_ids)} candidates ({(t1-t0).total_seconds():.1f}s)")
        
        # Build a map of FAISS candidates for quick lookup
        faiss_set = set(faiss_ids)
        faiss_score_map = {cid: score for cid, score in zip(faiss_ids, faiss_scores)}
        
        # Stage 4: Feature scoring & ranking
        print(f"\n[Stage 4] Feature scoring → Top {top_k}...")
        t0 = datetime.now()
        
        scored_candidates = []
        for i, candidate_id in enumerate(faiss_ids):
            if i % 100 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  Scoring {i}/{len(faiss_ids)} ({elapsed:.1f}s)...")
            
            # Find candidate
            candidate = next(
                (c for c in self.candidates if c['candidate_id'] == candidate_id),
                None
            )
            if not candidate:
                continue
            
            # Parse profile
            if candidate_id not in self.parsed_profiles:
                parsed = self.parser.parse_candidate(candidate)
                self.parsed_profiles[candidate_id] = parsed
            else:
                parsed = self.parsed_profiles[candidate_id]
            
            # Score with new components
            semantic_sim = faiss_score_map.get(candidate_id, 0.0)
            components = self.feature_scorer.score_candidate(
                candidate,
                parsed,
                semantic_similarity=semantic_sim,
                advanced_scorer=self.advanced_scorer
            )
            
            # Apply disqualifiers
            final_score = self.feature_scorer.apply_disqualifying_factors(
                components.final_score,
                parsed,
                candidate
            )
            
            scored_candidates.append({
                'candidate_id': candidate_id,
                'final_score': final_score,
                'components': components,
                'semantic_similarity': semantic_sim,
                'parsed_profile': parsed,
                'candidate_data': candidate
            })
        
        t1 = datetime.now()
        print(f"  ✓ Scored {len(scored_candidates)} candidates ({(t1-t0).total_seconds():.1f}s)")
        
        # Stage 5: Sort and finalize
        print(f"\n[Stage 5] Finalizing top {top_k}...")
        t0 = datetime.now()
        
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        results = []
        for rank, scored in enumerate(top_candidates, 1):
            result = {
                'rank': rank,
                'candidate_id': scored['candidate_id'],
                'final_score': float(scored['final_score']),
                'components': {
                    'technical_relevance': float(scored['components'].technical_relevance),
                    'production_experience': float(scored['components'].production_experience),
                    'profile_quality_multiplier': float(scored['components'].profile_quality_multiplier),
                    'behavioral_engagement': float(scored['components'].behavioral_engagement),
                    'experience_level_fit': float(scored['components'].experience_level_fit),
                    'evaluation_framework_score': float(scored['components'].evaluation_framework_score),
                    'product_mindset_score': float(scored['components'].product_mindset_score),
                    'semantic_similarity': float(scored['components'].semantic_similarity)
                }
            }
            results.append(result)
        
        t1 = datetime.now()
        print(f"  ✓ Complete ({(t1-t0).total_seconds():.1f}s)")
        
        print("\n" + "="*70)
        print(f"End: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Top candidate: {results[0]['candidate_id']} (score: {results[0]['final_score']:.4f})")
        print("="*70 + "\n")
        
        return results, scored_candidates
    
    def _faiss_retrieve_from_pool(
        self,
        query_text: str,
        pool_ids: List[str],
        top_k: int = 500
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k from a specific pool using FAISS"""
        
        # Embed query
        self.precomputer.load_model()
        query_embedding = self.precomputer.model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        # Normalize
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Get indices in full index for pool IDs
        pool_indices = []
        for pool_id in pool_ids:
            try:
                idx = self.candidate_ids.index(pool_id)
                pool_indices.append(idx)
            except ValueError:
                continue
        
        # Compute similarities for pool
        pool_embeddings = self.candidate_embeddings[pool_indices].astype('float32')
        similarities = np.dot(pool_embeddings, query_embedding.T).flatten()
        
        # Get top-k
        top_indices_in_pool = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_ids = [pool_ids[i] for i in top_indices_in_pool]
        retrieved_scores = [float(similarities[i]) for i in top_indices_in_pool]
        
        return retrieved_ids, retrieved_scores
    
    def save_results(self, results: List[Dict], output_dir: str = './ranking_output'):
        """Save ranking results to CSV and JSON"""
        
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV
        csv_data = []
        for result in results:
            csv_data.append({
                'candidate_id': result['candidate_id'],
                'rank': result['rank'],
                'score': f"{result['final_score']:.4f}"
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'submission.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # JSON
        json_path = os.path.join(output_dir, 'ranking_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {json_path}")
