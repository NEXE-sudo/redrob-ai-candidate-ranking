#!/usr/bin/env python3
"""
Download all required models to local cache for offline execution.
Run this ONCE before submission. Requires network access.

Usage: python scripts/download_models.py
"""
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def download_models(models_dir: Path = None):
    """Download embedding and cross-encoder models to a local models directory."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    embedding_model_name = 'BAAI/bge-large-en-v1.5'
    cross_encoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

    embedding_path = models_dir / 'bge-large-en-v1.5'
    cross_encoder_path = models_dir / 'cross-encoder-ms-marco-MiniLM-L-12-v2'

    if not embedding_path.exists():
        print(f"Downloading embedding model to {embedding_path}...")
        model = SentenceTransformer(embedding_model_name)
        model.save(str(embedding_path))
        print(f"  ✓ Saved to {embedding_path}")
    else:
        print(f"Embedding model already cached at {embedding_path}")

    if not cross_encoder_path.exists():
        print(f"Downloading cross-encoder model to {cross_encoder_path}...")
        ce = CrossEncoder(cross_encoder_model_name)
        ce.save(str(cross_encoder_path))
        print(f"  ✓ Saved to {cross_encoder_path}")
    else:
        print(f"Cross-encoder model already cached at {cross_encoder_path}")

    print("\nAll requested models are cached locally.")
    print("Add models/ to .gitignore if desired.")
    return {
        'embedding_model': str(embedding_path),
        'cross_encoder_model': str(cross_encoder_path)
    }


if __name__ == '__main__':
    download_models()
