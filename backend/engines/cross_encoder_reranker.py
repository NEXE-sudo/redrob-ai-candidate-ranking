"""
Cross-Encoder Reranking Module
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for recruiter-style reranking.
Reranks FAISS pool before feature scoring.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import os

# Lazy import to avoid loading model on module import
_cross_encoder = None


def _get_cross_encoder():
    """Lazy load cross-encoder model"""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[CrossEncoder] Loading cross-encoder model...")
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            print("[CrossEncoder] Model loaded ✓")
        except ImportError:
            raise ImportError(
                "sentence-transformers required for CrossEncoder. "
                "Install with: pip install sentence-transformers"
            )
    return _cross_encoder


class CrossEncoderReranker:
    """
    Rerank candidates using cross-encoder.
    Processes only FAISS-retrieved candidates (not all 100k).
    """
    
    def __init__(self):
        self.model = None
        self.telemetry = {
            'candidates_processed': 0,
            'runtime_seconds': 0.0,
            'scores': []
        }
    
    def _build_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Build comprehensive candidate profile text from structured data"""
        text_parts = []
        
        # Profile info
        profile = candidate.get('profile', {})
        if profile.get('current_title'):
            text_parts.append(f"Current Title: {profile['current_title']}")
        if profile.get('headline'):
            text_parts.append(f"Headline: {profile['headline']}")
        if profile.get('summary'):
            text_parts.append(f"Summary: {profile['summary']}")
        
        # Top skills (max 15)
        skills = candidate.get('skills', [])
        if skills:
            skill_names = [s['name'] for s in skills[:15]]
            text_parts.append(f"Skills: {', '.join(skill_names)}")
        
        # Most recent 3 roles
        career = candidate.get('career_history', [])
        if career:
            text_parts.append("Career History:")
            for role in career[:3]:
                role_text = f"  {role.get('title', '')} at {role.get('company', '')}"
                if role.get('description'):
                    role_text += f": {role['description'][:200]}"
                text_parts.append(role_text)
        
        return '\n'.join(text_parts)
    
    def rerank(
        self,
        jd_text: str,
        candidate_ids: List[str],
        candidates_by_id: Dict[str, Dict[str, Any]],
        top_k: int = 250
    ) -> Tuple[List[str], List[float]]:
        """
        Rerank FAISS-retrieved candidates using cross-encoder.
        
        Args:
            jd_text: Job description
            candidate_ids: List of candidate IDs from FAISS
            candidates_by_id: Lookup dict of candidate data
            top_k: Return top-k after reranking
            
        Returns:
            (reranked_ids, reranked_scores)
        """
        if not candidate_ids:
            return [], []
        
        t_start = datetime.now()
        self.model = _get_cross_encoder()
        
        # Build pairs for cross-encoder
        pairs = []
        valid_ids = []
        
        for cid in candidate_ids:
            candidate = candidates_by_id.get(cid)
            if not candidate:
                continue
            
            candidate_text = self._build_candidate_text(candidate)
            pairs.append([jd_text, candidate_text])
            valid_ids.append(cid)
        
        if not pairs:
            return [], []
        
        # Compute cross-encoder scores
        print(f"[CrossEncoder] Scoring {len(pairs)} candidate pairs...")
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Get top-k
        top_indices = sorted_indices[:top_k]
        reranked_ids = [valid_ids[i] for i in top_indices]
        reranked_scores = scores[top_indices].tolist()
        
        # Telemetry
        runtime = (datetime.now() - t_start).total_seconds()
        self.telemetry = {
            'candidates_processed': len(pairs),
            'runtime_seconds': runtime,
            'scores': reranked_scores,
            'avg_score': float(np.mean(reranked_scores)),
            'top_score': float(max(reranked_scores)) if reranked_scores else 0.0,
            'bottom_score': float(min(reranked_scores)) if reranked_scores else 0.0
        }
        
        print(f"[CrossEncoder] Scored {len(pairs)} pairs in {runtime:.1f}s")
        print(f"[CrossEncoder] Top score: {self.telemetry['top_score']:.4f}, "
              f"Avg: {self.telemetry['avg_score']:.4f}")
        
        return reranked_ids, reranked_scores
    
    def print_telemetry(self):
        """Print telemetry summary"""
        print("\n" + "="*70)
        print("CROSS-ENCODER RERANKING TELEMETRY")
        print("="*70)
        print(f"  Candidates processed: {self.telemetry['candidates_processed']}")
        print(f"  Runtime: {self.telemetry['runtime_seconds']:.2f}s")
        print(f"  Top score: {self.telemetry['top_score']:.4f}")
        print(f"  Avg score: {self.telemetry['avg_score']:.4f}")
        print(f"  Bottom score: {self.telemetry['bottom_score']:.4f}")
        print("="*70)
