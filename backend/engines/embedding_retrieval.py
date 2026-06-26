"""
Embedding Retrieval Module
Implements FAISS-based semantic search for candidate retrieval.
Uses Sentence Transformers for embedding generation.
OPTIMIZATIONS: BM25 vectorization, FAISS thread config
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Import threading configuration
try:
    from .embedding_precompute import _CPU_COUNT
except Exception:
    _CPU_COUNT = None


def _get_faiss():
    """Lazily import FAISS to avoid import-time interaction issues."""
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "FAISS is required for embedding retrieval. Install with: pip install faiss-cpu"
        ) from e
    return faiss


# NOTE: EmbeddingRetriever is superseded by EmbeddingPrecomputer (embedding_precompute.py)
# which pre-computes and caches embeddings for all candidates for efficient retrieval


class BM25Retriever:
    """BM25-based keyword retrieval for initial filtering (PHASE 7: Optimized for performance)"""
    
    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.bm25 = None
        self.candidate_ids = None
        self.corpus = None
    
    def build_index(self, candidates: List[dict], text_fields: List[str] = None):
        """Build BM25 index from candidates"""
        
        if text_fields is None:
            text_fields = ['summary', 'headline', 'current_title', 'skills_text', 'career_text']
        
        self.candidate_ids = [c['candidate_id'] for c in candidates]
        corpus = []
        
        for candidate in candidates:
            # Concatenate text fields
            text_parts = []
            
            profile = candidate.get('profile', {})
            text_parts.append(profile.get('summary', ''))
            text_parts.append(profile.get('headline', ''))
            text_parts.append(profile.get('current_title', ''))
            
            # Skills
            skills = candidate.get('skills', [])
            text_parts.append(' '.join([s['name'] for s in skills]))
            
            # Career
            career = candidate.get('career_history', [])
            for role in career:
                text_parts.append(role.get('title', ''))
                text_parts.append(role.get('description', ''))
            
            combined_text = ' '.join([t for t in text_parts if t])
            tokens = re.findall(r'\b[a-z0-9]+\b', combined_text.lower())
            corpus.append(tokens)
        
        self.corpus = corpus
        self.bm25 = self.BM25Okapi(corpus)
        print(f"[PERFORMANCE] BM25 index built with {len(corpus)} documents")
    
    def retrieve(self, query_text: str, top_k: int = 2000) -> Tuple[List[str], List[float]]:
        """Retrieve top-k candidates using BM25 (PHASE 7: Vectorized selection)"""
        
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        tokens = re.findall(r'\b[a-z0-9]+\b', query_text.lower())
        scores = self.bm25.get_scores(tokens)
        
        # PHASE 7: Use vectorized numpy operations for top-k selection
        scores_array = np.array(scores)
        top_indices = np.argsort(scores_array)[-top_k:][::-1]
        
        candidate_ids = [self.candidate_ids[i] for i in top_indices]
        similarity_scores = [float(scores[i]) for i in top_indices]
        
        return candidate_ids, similarity_scores

