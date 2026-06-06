"""
Embedding Precomputation Module
Pre-computes and caches embeddings for all candidates to enable fast retrieval.
Run this ONCE before ranking; subsequent ranking uses cached embeddings.
"""

import json
import numpy as np
import os
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pickle
import faiss


class EmbeddingPrecomputer:
    """Pre-compute and cache embeddings for all candidates"""
    
    def __init__(
        self,
        model_name: str = 'BAAI/bge-small-en-v1.5',
        embedding_dim: int = 384,
        cache_dir: str = './embeddings_cache'
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.project_root = Path(__file__).resolve().parents[2]
        self.engine_dir = Path(__file__).resolve().parents[1]
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.model = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._print_cache_diagnostics()

    def _resolve_cache_dir(self, cache_dir: str) -> Path:
        """Resolve cache_dir to an absolute path under the backend project layout."""
        cache_path = Path(cache_dir)
        if cache_path.is_absolute():
            return cache_path.resolve()
        return (self.engine_dir / cache_path).resolve()

    def _cache_paths(self, output_prefix: str = 'precomputed_embeddings'):
        embeddings_path = self.cache_dir / f'{output_prefix}_embeddings.npy'
        ids_path = self.cache_dir / f'{output_prefix}_ids.pkl'
        metadata_path = self.cache_dir / f'{output_prefix}_metadata.json'
        faiss_index_path = self.cache_dir / f'{output_prefix}_faiss.index'
        return embeddings_path, ids_path, metadata_path, faiss_index_path

    def _print_cache_diagnostics(self):
        print("\nEmbedding cache diagnostics:")
        print(f"  Current working directory: {Path.cwd()}")
        print(f"  Project root: {self.project_root}")
        print(f"  Backend engine directory: {self.engine_dir}")
        print(f"  Resolved cache directory: {self.cache_dir}")
        embeddings_path, ids_path, metadata_path, faiss_index_path = self._cache_paths()
        print("  Cache file locations:")
        print(f"    Embeddings: {embeddings_path} -> {'FOUND' if embeddings_path.exists() else 'MISSING'}")
        print(f"    IDs:        {ids_path} -> {'FOUND' if ids_path.exists() else 'MISSING'}")
        print(f"    Metadata:   {metadata_path} -> {'FOUND' if metadata_path.exists() else 'MISSING'}")
        print(f"    FAISS idx:  {faiss_index_path} -> {'FOUND' if faiss_index_path.exists() else 'MISSING'}")
    
    def load_model(self):
        """Load Sentence Transformer model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device='cpu')
            print("Model loaded successfully")
    
    def precompute_embeddings(
        self,
        jsonl_path: str,
        output_prefix: str = 'precomputed_embeddings',
        batch_size: int = 64,
        max_candidates: int = None
    ) -> Dict[str, Any]:
        """Pre-compute embeddings for all candidates
        
        Args:
            jsonl_path: Path to candidates.jsonl
            output_prefix: Prefix for output files
            batch_size: Batch size for encoding
            max_candidates: Limit number of candidates (None = all)
            
        Returns:
            Metadata dict with cache info
        """
        
        self.load_model()
        
        print(f"\nPrecomputing embeddings from {jsonl_path}")
        print(f"Output prefix: {output_prefix}")
        
        # Load and prepare candidate texts
        candidate_ids = []
        texts = []
        total_loaded = 0
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                if max_candidates and line_num >= max_candidates:
                    break
                
                try:
                    candidate = json.loads(line)
                    cid = candidate['candidate_id']
                    candidate_ids.append(cid)
                    
                    # Concatenate text fields
                    text_parts = []
                    
                    profile = candidate.get('profile', {})
                    text_parts.append(profile.get('summary', ''))
                    text_parts.append(profile.get('headline', ''))
                    text_parts.append(profile.get('current_title', ''))
                    
                    # Skills
                    skills = candidate.get('skills', [])
                    skill_text = ', '.join([s['name'] for s in skills[:20]])
                    text_parts.append(skill_text)
                    
                    # Career history
                    career = candidate.get('career_history', [])
                    for role in career[:2]:
                        text_parts.append(role.get('title', ''))
                        text_parts.append(role.get('description', ''))
                    
                    combined_text = ' '.join([t for t in text_parts if t])
                    texts.append(combined_text[:1000])  # Cap at 1000 chars
                    total_loaded += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {total_loaded} candidates")
        
        # Compute embeddings in batches
        print(f"\nComputing embeddings (batch size: {batch_size})...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Normalize for cosine similarity
            batch_embeddings = batch_embeddings / (
                np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-8
            )
            
            all_embeddings.append(batch_embeddings)
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} candidates")
        
        # Stack all embeddings
        embeddings_array = np.vstack(all_embeddings).astype('float32')
        
        print(f"\nEmbeddings shape: {embeddings_array.shape}")
        
        # Save embeddings and metadata
        embeddings_path, ids_path, metadata_path, faiss_index_path = self._cache_paths(output_prefix)

        np.save(embeddings_path, embeddings_array)
        with open(ids_path, 'wb') as f:
            pickle.dump(candidate_ids, f)

        metadata = {
            'total_candidates': len(candidate_ids),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat(),
            'embeddings_file': str(embeddings_path),
            'ids_file': str(ids_path),
            'faiss_index_file': str(faiss_index_path),
            'num_batches': len(all_embeddings)
        }

        self.build_faiss_index(embeddings_array, output_path=faiss_index_path)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nEmbeddings cached:")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  IDs: {ids_path}")
        print(f"  FAISS index: {faiss_index_path}")
        print(f"  Metadata: {metadata_path}")

        self._verify_saved_files(embeddings_path, ids_path, metadata_path, faiss_index_path)
        return metadata
    
    def load_precomputed_embeddings(self, output_prefix: str = 'precomputed_embeddings'):
        """Load pre-computed embeddings from cache
        
        Returns:
            Tuple of (embeddings, candidate_ids, metadata)
        """

        embeddings_path, ids_path, metadata_path, _ = self._cache_paths(output_prefix)
        if not all(path.exists() for path in [embeddings_path, ids_path, metadata_path]):
            missing = [str(path) for path in [embeddings_path, ids_path, metadata_path] if not path.exists()]
            raise FileNotFoundError(
                f"Precomputed embeddings not found with prefix '{output_prefix}'. Missing: {missing}"
            )

        embeddings = np.load(embeddings_path)
        with open(ids_path, 'rb') as f:
            candidate_ids = pickle.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(candidate_ids)} precomputed embeddings from cache")
        print(f"Model: {metadata['model_name']}")
        print(f"Embedding dimension: {metadata['embedding_dim']}")
        
        return embeddings, candidate_ids, metadata

    def build_faiss_index(self, embeddings: np.ndarray, output_path: str = None):
        """Build and persist a FAISS inner-product index for embeddings."""
        if output_path is None:
            _, _, _, output_path = self._cache_paths()

        embeddings = embeddings.astype('float32', copy=False)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(output_path))
        output_path = Path(output_path)
        if not output_path.exists():
            raise RuntimeError(f"Failed to save FAISS index to {output_path}")
        return output_path

    def load_faiss_index(self, index_path: str = None):
        """Load a persisted FAISS index from disk."""
        if index_path is None:
            _, _, _, index_path = self._cache_paths()

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'")

        return faiss.read_index(str(index_path))

    def _verify_saved_files(self, embeddings_path: Path, ids_path: Path, metadata_path: Path, faiss_index_path: Path):
        for path in [embeddings_path, ids_path, metadata_path, faiss_index_path]:
            if not path.exists():
                raise RuntimeError(f"Cache save failed: required file not found: {path}")


def precompute_script(candidates_file: str):
    """Standalone script to precompute embeddings"""
    
    print("\n" + "="*60)
    print("EMBEDDING PRECOMPUTATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    precomputer = EmbeddingPrecomputer()
    
    try:
        metadata = precomputer.precompute_embeddings(
            jsonl_path=candidates_file,
            output_prefix='precomputed_embeddings',
            batch_size=64
        )
        
        print("\n" + "="*60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("PRECOMPUTATION COMPLETE ✓")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 precompute_embeddings.py <path_to_candidates.jsonl>")
        sys.exit(1)
    
    candidates_file = sys.argv[1]
    sys.exit(precompute_script(candidates_file))
