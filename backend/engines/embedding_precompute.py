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
        self.cache_dir = cache_dir
        self.model = None
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
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
        embeddings_path = os.path.join(self.cache_dir, f'{output_prefix}_embeddings.npy')
        ids_path = os.path.join(self.cache_dir, f'{output_prefix}_ids.pkl')
        metadata_path = os.path.join(self.cache_dir, f'{output_prefix}_metadata.json')
        
        np.save(embeddings_path, embeddings_array)
        with open(ids_path, 'wb') as f:
            pickle.dump(candidate_ids, f)
        
        metadata = {
            'total_candidates': len(candidate_ids),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat(),
            'embeddings_file': embeddings_path,
            'ids_file': ids_path,
            'num_batches': len(all_embeddings)
        }
        
        faiss_index_path = os.path.join(self.cache_dir, f'{output_prefix}_faiss.index')
        self.build_faiss_index(embeddings_array, output_path=faiss_index_path)
        metadata['faiss_index_file'] = faiss_index_path

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nEmbeddings cached:")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  IDs: {ids_path}")
        print(f"  FAISS index: {faiss_index_path}")
        print(f"  Metadata: {metadata_path}")
        
        return metadata
    
    def load_precomputed_embeddings(self, output_prefix: str = 'precomputed_embeddings'):
        """Load pre-computed embeddings from cache
        
        Returns:
            Tuple of (embeddings, candidate_ids, metadata)
        """
        
        embeddings_path = os.path.join(self.cache_dir, f'{output_prefix}_embeddings.npy')
        ids_path = os.path.join(self.cache_dir, f'{output_prefix}_ids.pkl')
        metadata_path = os.path.join(self.cache_dir, f'{output_prefix}_metadata.json')
        
        if not all(os.path.exists(p) for p in [embeddings_path, ids_path, metadata_path]):
            raise FileNotFoundError(f"Precomputed embeddings not found with prefix '{output_prefix}'")
        
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
            output_path = os.path.join(self.cache_dir, 'precomputed_embeddings_faiss.index')

        embeddings = embeddings.astype('float32', copy=False)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, output_path)
        return output_path

    def load_faiss_index(self, index_path: str = None):
        """Load a persisted FAISS index from disk."""
        if index_path is None:
            index_path = os.path.join(self.cache_dir, 'precomputed_embeddings_faiss.index')

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'")

        return faiss.read_index(index_path)


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
