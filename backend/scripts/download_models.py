#!/usr/bin/env python3
"""
Download all required models to local cache for offline execution.
Run this ONCE before submission. Requires network access.

Usage: python scripts/download_models.py
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("Downloading BGE embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
model.save(str(MODELS_DIR / "bge-small-en-v1.5"))
print("  ✓ Saved to models/bge-small-en-v1.5")

print("Downloading cross-encoder model...")
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
ce.save(str(MODELS_DIR / "cross-encoder-ms-marco-MiniLM-L-6-v2"))
print("  ✓ Saved to models/cross-encoder-ms-marco-MiniLM-L-6-v2")

print("\nAll models downloaded. Add models/ to .gitignore if large.")
print("Update embedding_precompute.py model_name to point to local path.")
