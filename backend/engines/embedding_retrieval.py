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

class EmbeddingRetriever:
    """FAISS-based embedding retrieval for candidates"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.SentenceTransformer = SentenceTransformer
        self.model = None
        self.candidate_embeddings = None
        self.candidate_ids = None
        self.index = None
        
    def load_model(self):
        if self.model is None:
            local_dir = self.model_name.split('/', 1)[1] if '/' in self.model_name else self.model_name
            local_path = Path(__file__).resolve().parents[1] / 'models' / local_dir
            if local_path.exists():
                self.model = self.SentenceTransformer(str(local_path), device='cpu')
                print(f"Loaded local embedding model from: {local_path}")
            else:
                raise FileNotFoundError(
                    f"Local embedding model not found at '{local_path}'. "
                    "Please run 'python backend/scripts/download_models.py' first to "
                    "download the required model for offline use."
                )

    def build_index(self, candidates: List[dict], batch_size: int = 64, use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
        if self.model is None:
            self.load_model()

        self.candidate_ids = [c['candidate_id'] for c in candidates]
        texts = []
        for candidate in candidates:
            text_parts = []
            profile = candidate.get('profile', {})
            text_parts.append(profile.get('summary', ''))
            text_parts.append(profile.get('headline', ''))
            text_parts.append(profile.get('current_title', ''))
            skills = candidate.get('skills', [])
            text_parts.append(' '.join([s.get('name', '') for s in skills[:20]]))
            career = candidate.get('career_history', [])
            for role in career[:2]:
                text_parts.append(role.get('title', ''))
                text_parts.append(role.get('description', ''))
            combined_text = ' '.join([t for t in text_parts if t])
            texts.append(combined_text[:1000])
            
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.candidate_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        self.candidate_embeddings = self.candidate_embeddings.astype('float32', copy=False)

        faiss = _get_faiss()
        if _CPU_COUNT is not None:
            faiss.omp_set_num_threads(_CPU_COUNT)
        self.index = faiss.IndexFlatIP(self.candidate_embeddings.shape[1])
        self.index.add(self.candidate_embeddings)
        return self.candidate_embeddings, self.candidate_ids

    def retrieve(self, query_text: str, top_k: int = 500) -> Tuple[List[str], List[float]]:
        if self.candidate_embeddings is None:
            raise ValueError("Index not built.")
        if self.model is None:
            self.load_model()

        query_embedding = self.model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        similarities = np.dot(self.candidate_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_ids = [self.candidate_ids[i] for i in top_indices]
        retrieved_scores = [float(similarities[i]) for i in top_indices]
        return retrieved_ids, retrieved_scores
