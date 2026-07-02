"""
Multi-stage ranking engine for candidate retrieval and scoring.
The pipeline combines lexical retrieval, semantic retrieval, reranking, and scoring
for final candidate ranking.
"""

import builtins
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os
import math
import torch


def _silence_output(*_args, **_kwargs):
    return None


builtins.print = _silence_output

from .candidate_profile_parser import CandidateProfileParser, ParsedProfile
from .feature_scorer import FeatureScorer, ScoringComponents
from .advanced_scorer import AdvancedScorer
from .embedding_retrieval import BM25Retriever
from .embedding_precompute import EmbeddingPrecomputer, _CPU_COUNT, _get_available_ram_gb, _calculate_adaptive_batch_size
from .cross_encoder_reranker import CrossEncoderReranker
from .recruiter_jd_parser import RecruiterCentricJDParser
from .advanced_scoring_components import (
    CareerTrajectoryAnalyzer, ProductCompanyScorer, RetrievalDepthScorer,
    EvaluationFrameworkScorer, HoneypotDetector
)

import faiss


def _silence_output(*_args, **_kwargs):
    return None


print = _silence_output

# Configure FAISS threading (torch threading already configured in embedding_precompute)
faiss.omp_set_num_threads(_CPU_COUNT)


class OptimizedRankingEngine:
    """Coordinate candidate retrieval, reranking, and scoring for final ranking."""
    
    def __init__(
        self,
        embeddings_cache_dir: str = './embeddings_cache',
        use_precomputed_embeddings: bool = True,
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
        enable_cross_encoder: bool = True,
        enable_honeypot_detection: bool = True
    ):
        self.parser = CandidateProfileParser()
        self.feature_scorer = FeatureScorer(self.parser)
        self.advanced_scorer = AdvancedScorer(self.parser)
        self.bm25_retriever = BM25Retriever()
        
        # Configure the embedding model used for semantic retrieval.
        self.embedding_model_name = embedding_model
        self.precomputer = EmbeddingPrecomputer(
            model_name=embedding_model,
            cache_dir=embeddings_cache_dir
        )
        
        # Configure the cross-encoder reranker.
        self.cross_encoder_reranker = CrossEncoderReranker() if enable_cross_encoder else None
        
        # Parse the job description into structured requirements.
        self.recruiter_jd_parser = RecruiterCentricJDParser()
        self.requirement_profile = None
        
        # Initialize advanced scoring components.
        self.career_trajectory_analyzer = CareerTrajectoryAnalyzer()
        self.product_company_scorer = ProductCompanyScorer()
        self.retrieval_depth_scorer = RetrievalDepthScorer()
        self.evaluation_framework_scorer = EvaluationFrameworkScorer()
        
        # Initialize honeypot detection.
        self.honeypot_detector = HoneypotDetector() if enable_honeypot_detection else None
        
        self.embeddings_cache_dir = self.precomputer.cache_dir
        self.faiss_index_path = self.embeddings_cache_dir / 'precomputed_embeddings_faiss.index'
        self.use_precomputed = use_precomputed_embeddings
        
        # Track runtime telemetry for reporting.
        self.benchmark_telemetry = {}

        # Print resource utilization at startup.
        print("\n" + "="*70)
        print("OPTIMIZED RANKING ENGINE STARTUP")
        print("="*70)
        print(f"  CPU count: {_CPU_COUNT}")
        print(f"  PyTorch threads: {torch.get_num_threads()}")
        print(f"  PyTorch interop threads: {torch.get_num_interop_threads()}")
        print(f"  Available RAM: {_get_available_ram_gb():.1f} GB")
        print(f"  Current working directory: {Path.cwd()}")
        print(f"  Resolved embedding cache directory: {self.embeddings_cache_dir}")
        print(f"  Persisted FAISS index path: {self.faiss_index_path}")
        print("="*70 + "\n")
        
        self.precomputer._print_cache_diagnostics()
        
        self.candidates = []
        self.candidates_by_id = {}
        self.candidate_id_to_index = {}
        self.parsed_profiles = {}
        self.faiss_index = None
        self.candidate_embeddings = None
        self.candidate_ids = []
        self.jd_text = None
        self.jd_keywords = []
        self.jd_embedding = None
        self.candidates_jsonl_path = None
    
    def load_candidates(self, jsonl_path: str):
        """Load candidates from JSONL"""
        print(f"Loading candidates from {jsonl_path}...")
        
        self.candidates_jsonl_path = jsonl_path
        self.candidates = []
        self.candidates_by_id = {}
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    candidate = json.loads(line)
                    self.candidates.append(candidate)
                    self.candidates_by_id[candidate['candidate_id']] = candidate
                except json.JSONDecodeError as e:
                    if line_num < 10:  # Only warn for first 10
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.candidates)} candidates")
    
    def prepare_jd_text(self, jd: str):
        """
        Set the job description and extract structured JD signals.
        Parse requirements, preferences, and negative signals.
        """
        self.jd_text = jd.strip()
        
        # Use the structured JD parser.
        self.requirement_profile = self.recruiter_jd_parser.parse_jd(self.jd_text)
        
        # Legacy keywords for backward compatibility
        self.jd_keywords = self._extract_jd_keywords(self.jd_text)

        # Extract structured skill keywords for trust scoring
        self.jd_skill_keywords = self._extract_jd_skill_keywords(self.jd_text)
        # Augment with recruiter-parsed required/preferred terms
        self.jd_skill_keywords = list(dict.fromkeys(
            self.jd_skill_keywords +
            list(self.requirement_profile.required_keywords) +
            list(self.requirement_profile.preferred_keywords)
        ))
        print(f"  JD skill keywords: {self.jd_skill_keywords}")
        
        # Print extracted requirements for inspection.
        print("\n[JD Parsing Results]")
        print(f"  Required keywords: {', '.join(self.requirement_profile.required_keywords)[:100]}...")
        print(f"  Preferred keywords: {', '.join(self.requirement_profile.preferred_keywords)[:100]}...")
        print(f"  Experience range: {self.requirement_profile.target_experience_min}-{self.requirement_profile.target_experience_max} years")
        print(f"  Hands-on coding required: {self.requirement_profile.hands_on_coding}")
        print(f"  Leadership required: {self.requirement_profile.leadership_required}\n")

    def _extract_jd_keywords(self, text: str) -> List[str]:
        """Extract strong query terms from the job description."""
        text = text.lower()
        candidate_phrases = [
            'embeddings', 'retrieval', 'vector database', 'faiss', 'milvus', 'pinecone',
            'semantic search', 'ranking', 'ndcg', 'mrr', 'map', 'a/b testing',
            'online evaluation', 'production', 'scale', 'recommendation', 'rag',
            'startup', 'product', 'ml', 'python', 'llm', 'distributed'
        ]
        # Use word-boundary matching to avoid short-token collisions (e.g. 'map' in 'roadmap')
        keywords = [phrase for phrase in candidate_phrases if re.search(r"\b" + re.escape(phrase) + r"\b", text)]
        if not keywords:
            keywords = re.findall(r"\b[ a-z]{3,}\b", text)[:10]
        return list(dict.fromkeys(keywords))

    def _extract_jd_skill_keywords(self, jd_text: str) -> List[str]:
        """
        Extract skill-specific keywords from JD for use in skill trust
        and assessment scoring. More targeted than general BM25 keywords.
        """
        jd_lower = jd_text.lower()
        skill_terms = [
            "python", "pytorch", "tensorflow", "scikit-learn",
            "embeddings", "faiss", "milvus", "pinecone", "weaviate", "qdrant",
            "elasticsearch", "opensearch", "vector database",
            "ranking", "learning-to-rank", "ltr", "ndcg", "mrr", "map",
            "recommendation", "retrieval", "semantic search", "rag",
            "llm", "fine-tuning", "lora", "transformer",
            "machine learning", "deep learning", "nlp",
            "a/b testing", "experimentation", "production ml",
            "xgboost", "lightgbm", "spark", "distributed systems"
        ]
        # Word-boundary match for skills to avoid substring false-positives
        return [term for term in skill_terms if re.search(r"\b" + re.escape(term) + r"\b", jd_lower)]
    
    def _load_or_build_faiss_index(self):
        """Load cached embeddings and the FAISS index, or build them if needed."""
        self.candidate_ids = [c['candidate_id'] for c in self.candidates]

        if self.use_precomputed:
            try:
                print("[PERFORMANCE] Loading precomputed embeddings...")
                self.candidate_embeddings, self.candidate_ids, metadata = \
                    self.precomputer.load_precomputed_embeddings()
                # Use float32 consistently to avoid unnecessary copies.
                self.candidate_embeddings = self.candidate_embeddings.astype('float32', copy=False)
                self.candidate_id_to_index = {
                    cid: idx for idx, cid in enumerate(self.candidate_ids)
                }
                print("[PERFORMANCE] Precomputed embeddings ready ✓")

                if self.faiss_index_path.exists():
                    print(f"[PERFORMANCE] Loading persisted FAISS index from {self.faiss_index_path}...")
                    self.faiss_index = self.precomputer.load_faiss_index(self.faiss_index_path)
                    print("[PERFORMANCE] FAISS index loaded ✓")
                else:
                    print("[PERFORMANCE] Building FAISS index (will cache for future runs)...")
                    self.faiss_index = self.precomputer.build_faiss_index(
                        self.candidate_embeddings,
                        output_path=self.faiss_index_path
                    )
                    print(f"[PERFORMANCE] Saved FAISS index: {self.faiss_index_path}")

                return

            except FileNotFoundError as e:
                # PHASE 4: Never silently recompute - fail explicitly
                print(f"\nERROR: Precomputed embeddings not found!")
                print(f"Details: {e}")
                print(f"\nTo generate precomputed embeddings, run:")
                print(f"  python embedding_precompute.py <candidates_jsonl_path>")
                print(f"\nWill NOT silently recompute - embeddings cache is required.")
                raise

        # If precomputed embeddings are not being used, fail fast because ranking depends on them.
        print("[ERROR] Ranking engine requires precomputed embeddings (use_precomputed_embeddings=True)")
        raise RuntimeError(
            "Candidate embeddings must be precomputed. Run embedding_precompute.py first. "
            "Ranking must not generate candidate embeddings."
        )
    
    def rank_candidates_fast(
        self,
        top_k: int = 100,
        bm25_top_k: int = 3000,
        faiss_top_k: int = 1000,
        cross_encoder_top_k: int = 250
    ) -> Tuple[List[Dict], Any]:
        """
        Execute the full ranking pipeline.

        Pipeline:
        Stage 1: Load embeddings and the BM25 index
        Stage 2: BM25 retrieval
        Stage 3: FAISS semantic retrieval
        Stage 4: Cross-encoder reranking
        Stage 5: Feature scoring and final ranking
        """
        
        if not self.candidates or not self.jd_text:
            raise ValueError("Load candidates and JD first")
        
        # Record start times for pipeline telemetry.
        pipeline_start = datetime.now()
        stage_timings = {}
        
        print("\n" + "="*70)
        print("MULTI-STAGE RANKING PIPELINE")
        print("="*70)
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
        
        # Stage 1: Load/Build indices
        print("\n[Stage 1] Building retrieval indices...")
        t0 = datetime.now()
        self.bm25_retriever.build_index(self.candidates)
        self._load_or_build_faiss_index()
        stage_timings['stage1_setup'] = (datetime.now() - t0).total_seconds()
        print(f"  ✓ Complete ({stage_timings['stage1_setup']:.1f}s)")
        
        # Stage 2: BM25 retrieval (fast keyword match)
        print(f"\n[Stage 2] BM25 retrieval → Top {bm25_top_k}...")
        t0 = datetime.now()
        bm25_query = self.jd_text
        if getattr(self, 'jd_keywords', None):
            bm25_query += ' ' + ' '.join(self.jd_keywords[:20])
        bm25_ids, bm25_scores = self.bm25_retriever.retrieve(
            bm25_query,
            top_k=bm25_top_k
        )
        stage_timings['bm25'] = (datetime.now() - t0).total_seconds()
        print(f"  ✓ Retrieved {len(bm25_ids)} candidates ({stage_timings['bm25']:.1f}s)")
        
        # Stage 3: FAISS retrieval (semantic match from BM25 pool)
        print(f"\n[Stage 3] FAISS semantic retrieval from pool → Top {faiss_top_k}...")
        t0 = datetime.now()
        faiss_ids, faiss_scores = self._faiss_retrieve_from_pool(
            self.jd_text,
            bm25_ids,
            top_k=faiss_top_k
        )
        stage_timings['faiss'] = (datetime.now() - t0).total_seconds()
        print(f"  ✓ Retrieved {len(faiss_ids)} candidates ({stage_timings['faiss']:.1f}s)")
        
        # Build a map of FAISS candidates for quick lookup
        faiss_set = set(faiss_ids)
        faiss_score_map = {cid: score for cid, score in zip(faiss_ids, faiss_scores)}
        
        # Stage 3.5: Cross-encoder reranking.
        if self.cross_encoder_reranker and cross_encoder_top_k > 0:
            print(f"\n[Stage 3.5] Cross-Encoder reranking → Top {cross_encoder_top_k}...")
            t0 = datetime.now()
            try:
                ce_ids, ce_scores = self.cross_encoder_reranker.rerank(
                    self.jd_text,
                    faiss_ids,
                    self.candidates_by_id,
                    top_k=cross_encoder_top_k
                )
                stage_timings['cross_encoder'] = (datetime.now() - t0).total_seconds()
                
                # Update for next stage
                faiss_ids = ce_ids
                faiss_scores = ce_scores
                faiss_score_map = {cid: score for cid, score in zip(ce_ids, ce_scores)}
                faiss_set = set(ce_ids)
                
                self.cross_encoder_reranker.print_telemetry()
            except Exception as e:
                print(f"  ⚠ Cross-encoder reranking failed: {e}")
                print(f"  Continuing with FAISS results only")
                stage_timings['cross_encoder'] = 0.0
        else:
            stage_timings['cross_encoder'] = 0.0
        
        # Stage 4: Feature scoring and candidate ranking.
        print(f"\n[Stage 4] Feature scoring (Phases 3-6) → Top {top_k}...")
        t0 = datetime.now()
        
        scored_candidates = []
        seen_candidates = set()
        telemetry = {
            'skipped_candidates': 0,
            'invalid_candidates': 0,
            'duplicate_candidates': 0
        }
        
        for i, candidate_id in enumerate(faiss_ids):
            if i % 100 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  Scoring {i}/{len(faiss_ids)} ({elapsed:.1f}s)...")
            
            if candidate_id in seen_candidates:
                telemetry['duplicate_candidates'] += 1
                continue
            seen_candidates.add(candidate_id)
            
            # Lookup candidate by ID
            candidate = self.candidates_by_id.get(candidate_id)
            if not candidate:
                telemetry['skipped_candidates'] += 1
                continue
            
            try:
                # Parse profile
                if candidate_id not in self.parsed_profiles:
                    parsed = self.parser.parse_candidate(candidate)
                    self.parsed_profiles[candidate_id] = parsed
                else:
                    parsed = self.parsed_profiles[candidate_id]
                
                # Score the candidate using the full component set.
                raw_sem = faiss_score_map.get(candidate_id, 0.0)
                semantic_sim = 1.0 / (1.0 + math.exp(-raw_sem))
                components = self.feature_scorer.score_candidate(
                    candidate,
                    parsed,
                    semantic_similarity=semantic_sim,
                    advanced_scorer=self.advanced_scorer,
                    jd_skill_keywords=self.jd_skill_keywords,
                    requirement_profile=self.requirement_profile
                )
                
                # Add auxiliary component scores.
                career_traj_score = self.career_trajectory_analyzer.score(parsed, candidate)
                product_fit_score = self.product_company_scorer.score(parsed, candidate)
                retrieval_depth_score = self.retrieval_depth_scorer.score(candidate)
                eval_framework_score = self.evaluation_framework_scorer.score(candidate)
                
                # Apply honeypot detection.
                honeypot_penalty = 1.0
                if self.honeypot_detector:
                    risk_score = self.honeypot_detector.calculate_risk_score(parsed, candidate)
                    honeypot_penalty = self.honeypot_detector.get_penalty_multiplier(risk_score)

                auxiliary_bonus = (
                    career_traj_score * 0.02 +
                    product_fit_score * 0.02 +
                    retrieval_depth_score * 0.015
                )
                final_score = max((components.final_score + auxiliary_bonus) * honeypot_penalty, 0.0)

                # Apply an explicit cap to keep scores bounded.
                final_score = min(final_score, 1.0)

                # Apply additional disqualifiers (consulting-only etc.)
                if hasattr(self.feature_scorer, 'apply_disqualifying_factors'):
                    final_score = self.feature_scorer.apply_disqualifying_factors(
                        final_score,
                        parsed,
                        candidate
                    )
                
                scored_candidates.append({
                    'candidate_id': candidate_id,
                    'final_score': final_score,
                    'components': components,
                    'semantic_similarity': semantic_sim,
                    'parsed_profile': parsed,
                    'candidate_data': candidate,
                    'career_trajectory_score': career_traj_score,
                    'product_fit_score': product_fit_score,
                    'retrieval_depth_score': retrieval_depth_score,
                    'eval_framework_score': eval_framework_score,
                    'honeypot_penalty': honeypot_penalty
                })
            except Exception as e:
                telemetry['invalid_candidates'] += 1
                print(f"  ⚠ Failed to score candidate {candidate_id}: {e}")
                continue
        
        stage_timings['feature_scoring'] = (datetime.now() - t0).total_seconds()
        print(f"  ✓ Scored {len(scored_candidates)} candidates ({stage_timings['feature_scoring']:.1f}s)")
        if telemetry['skipped_candidates'] > 0 or telemetry['invalid_candidates'] > 0 or telemetry['duplicate_candidates'] > 0:
            print(f"    - Skipped (not found): {telemetry['skipped_candidates']}")
            print(f"    - Invalid (errors): {telemetry['invalid_candidates']}")
            print(f"    - Duplicates ignored: {telemetry['duplicate_candidates']}")

        # Do not rescale scores by pool max. Keep absolute scores so they are
        # comparable across runs and do not mask upstream weighting bugs.

        # Stage 5: Sort and finalize
        print(f"\n[Stage 5] Finalizing top {top_k}...")
        t0 = datetime.now()
        
        # Apply deterministic tie-breaking before ranking.
        # The challenge validator reads the CSV where scores are formatted to 4 decimals.
        # It considers candidates with the same formatted score as tied, so we MUST 
        # group them by that exact rounded value first, then tie-break by candidate_id.
        for scored in scored_candidates:
            scored['rounded_score'] = float(f"{scored['final_score']:.4f}")

        scored_candidates.sort(
            key=lambda x: (-x['rounded_score'], x['candidate_id'])
        )
        top_candidates = scored_candidates[:top_k]
        
        results = []
        for rank, scored in enumerate(top_candidates, 1):
            # Generate final reasoning for the candidate.
            reasoning_text = self._generate_reasoning(
                scored['candidate_data'],
                scored['parsed_profile'],
                scored['components'],
                scored.get('career_trajectory_score', 0.0),
                scored.get('product_fit_score', 0.0),
                scored.get('retrieval_depth_score', 0.0),
                scored.get('eval_framework_score', 0.0),
                scored.get('honeypot_penalty', 1.0)
            )
            result = {
                'rank': rank,
                'candidate_id': scored['candidate_id'],
                'final_score': float(scored['final_score']),
                'components': {
                    'title_relevance': float(scored['components'].title_relevance),
                    'skill_trust_score': float(scored['components'].skill_trust_score),
                    'assessment_score': float(scored['components'].assessment_score),
                    'technical_relevance': float(scored['components'].technical_relevance),
                    'production_experience': float(scored['components'].production_experience),
                    'profile_quality_multiplier': float(scored['components'].profile_quality_multiplier),
                    'experience_level_fit': float(scored['components'].experience_level_fit),
                    'education_score': float(scored['components'].education_score),
                    'behavioral_multiplier': float(scored['components'].behavioral_multiplier),
                    'semantic_similarity': float(scored['components'].semantic_similarity),
                    'evaluation_framework_score': float(scored['components'].evaluation_framework_score),
                    'product_mindset_score': float(scored['components'].product_mindset_score),
                },
                'reasoning': reasoning_text,
                'ranked_by': 'hybrid_bm25_faiss_crossencoder',
                'jd_alignment': {
                    'required_match': float(min(1.0, scored['components'].technical_relevance)),
                    'behavioral_trust': float(scored['components'].behavioral_multiplier),
                    'honeypot_penalty': float(1.0 - scored.get('honeypot_penalty', 1.0))
                }
            }
            results.append(result)
        
        stage_timings['finalize'] = (datetime.now() - t0).total_seconds()
        total_time = (datetime.now() - pipeline_start).total_seconds()
        stage_timings['total'] = total_time
        
        # Print runtime telemetry.
        print(f"\n[Stage 5] ✓ Complete ({stage_timings['finalize']:.1f}s)")
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"End: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total runtime: {total_time:.1f}s")
        print("\nStage Timings:")
        print(f"  Setup:           {stage_timings['stage1_setup']:.2f}s")
        print(f"  BM25 retrieval:  {stage_timings['bm25']:.2f}s")
        print(f"  FAISS retrieval: {stage_timings['faiss']:.2f}s")
        print(f"  Cross-Encoder:   {stage_timings['cross_encoder']:.2f}s")
        print(f"  Feature scoring: {stage_timings['feature_scoring']:.2f}s")
        print(f"  Finalize:        {stage_timings['finalize']:.2f}s")
        print(f"\nTop 5 Candidates:")
        for result in results[:5]:
            print(f"  {result['rank']}. {result['candidate_id']} ({result['final_score']:.4f})")
        print("\n[Feature Importance Analysis (Top 100 Averages)]")
        if top_candidates:
            num_candidates = len(top_candidates)
            # Weights must match ScoringComponents.final_score exactly.
            # Raw weights sum to 1.10, so the scorer normalizes by dividing by 1.10.
            # We replicate that normalization here so the printed contributions are accurate.
            _RAW_WEIGHTS = {
                'title_relevance': 0.25,
                'skill_trust_score': 0.22,
                'assessment_score': 0.18,
                'technical_relevance': 0.12,
                'production_experience': 0.08,
                'experience_level_fit': 0.06,
                'education_score': 0.03,
                'evaluation_framework_score': 0.03,
                'product_mindset_score': 0.03,
                'semantic_similarity': 0.10,
            }
            _SUM_WEIGHTS = sum(_RAW_WEIGHTS.values())  # 1.10
            _W = {k: v / _SUM_WEIGHTS for k, v in _RAW_WEIGHTS.items()}

            avg_contributions = {
                'title_relevance': sum(c['components'].title_relevance * _W['title_relevance'] for c in top_candidates) / num_candidates,
                'skill_trust_score': sum(c['components'].skill_trust_score * _W['skill_trust_score'] for c in top_candidates) / num_candidates,
                'assessment_score': sum(c['components'].assessment_score * _W['assessment_score'] for c in top_candidates) / num_candidates,
                'technical_relevance': sum(c['components'].technical_relevance * _W['technical_relevance'] for c in top_candidates) / num_candidates,
                'production_experience': sum(c['components'].production_experience * _W['production_experience'] for c in top_candidates) / num_candidates,
                'experience_level_fit': sum(c['components'].experience_level_fit * _W['experience_level_fit'] for c in top_candidates) / num_candidates,
                'education_score': sum(c['components'].education_score * _W['education_score'] for c in top_candidates) / num_candidates,
                'evaluation_framework_score': sum(c['components'].evaluation_framework_score * _W['evaluation_framework_score'] for c in top_candidates) / num_candidates,
                'product_mindset_score': sum(c['components'].product_mindset_score * _W['product_mindset_score'] for c in top_candidates) / num_candidates,
                'semantic_similarity': sum(c['components'].semantic_similarity * _W['semantic_similarity'] for c in top_candidates) / num_candidates,
                # Additive adjustments (not part of normalized base; shown separately)
                'behavioral_adjustment': sum((c['components'].behavioral_multiplier - 1.0) * 0.15 for c in top_candidates) / num_candidates,
                'career_trajectory_score': sum(c.get('career_trajectory_score', 0.0) * 0.02 for c in top_candidates) / num_candidates,
                'product_fit_score': sum(c.get('product_fit_score', 0.0) * 0.02 for c in top_candidates) / num_candidates,
            }

            # Print sorted by contribution
            sorted_contribs = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
            for feature, value in sorted_contribs:
                print(f"  {feature}: {value:.4f}")
        
        print("="*70 + "\n")
        
        # Store telemetry for later inspection.
        self.benchmark_telemetry = stage_timings
        
        return results, scored_candidates
    
    def _faiss_retrieve_from_pool(
        self,
        query_text: str,
        pool_ids: List[str],
        top_k: int = 500
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k candidates from a specific pool using precomputed embeddings."""

        if self.precomputer.model is None:
            self.precomputer.load_model()

        query_embedding = self.precomputer.model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1).astype('float32', copy=False)

        # Build an index array for the pool candidates.
        pool_indices = np.array([
            self.candidate_id_to_index[pool_id]
            for pool_id in pool_ids
            if pool_id in self.candidate_id_to_index
        ], dtype=np.int64)

        if len(pool_indices) == 0:
            return [], []

        if self.candidate_embeddings is None:
            raise ValueError("Candidate embeddings are required for FAISS retrieval.")

        # Compute similarity scores in vectorized form.
        pool_embeddings = self.candidate_embeddings[pool_indices]
        similarities = np.dot(pool_embeddings, query_embedding.T).flatten()

        # Select the highest-scoring candidates.
        top_indices_in_pool = np.argsort(similarities)[-top_k:][::-1]
        retrieved_ids = [pool_ids[int(i)] for i in top_indices_in_pool]
        retrieved_scores = [float(similarities[i]) for i in top_indices_in_pool]
        return retrieved_ids, retrieved_scores

    def _count_jd_matches(self, candidate_raw: Dict[str, Any], parsed_profile: ParsedProfile) -> Tuple[int, int, int]:
        """Count JD required, preferred, and negative keyword matches in candidate text."""
        if not self.requirement_profile:
            return 0, 0, 0

        candidate_text = self.feature_scorer._collect_candidate_text(candidate_raw, parsed_profile)
        # Normalize text for robust matching
        candidate_text = self.feature_scorer._normalize_text(candidate_text)

        def _match_set(keyword_set):
            count = 0
            for kw in keyword_set:
                kw_norm = self.feature_scorer._normalize_text(str(kw))
                if not kw_norm:
                    continue
                if re.search(r"\b" + re.escape(kw_norm) + r"\b", candidate_text):
                    count += 1
            return count

        required = _match_set(self.requirement_profile.required_keywords)
        preferred = _match_set(self.requirement_profile.preferred_keywords)
        negatives = _match_set(self.requirement_profile.negative_signals)

        return required, preferred, negatives

    def _generate_reasoning(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        components: ScoringComponents,
        career_trajectory_score: float = 0.0,
        product_fit_score: float = 0.0,
        retrieval_depth_score: float = 0.0,
        eval_framework_score: float = 0.0,
        honeypot_penalty: float = 1.0
    ) -> str:
        profile = candidate_raw.get("profile", {})
        skills = candidate_raw.get("skills", [])
        redrob = candidate_raw.get("redrob_signals", {})

        title = parsed_profile.current_title or parsed_profile.most_recent_role_title
        company = parsed_profile.current_company or parsed_profile.most_recent_role_company
        years = parsed_profile.years_experience

        # Build fact-based sentence 1: title + years + company
        if years and title and company:
            s1 = f"{years:.0f}yr {title} at {company}"
        elif years and title:
            s1 = f"{years:.0f}yr {title}"
        else:
            s1 = title or "Candidate"

        # Build sentence 2: top signals
        signals = []

        if components.assessment_score >= 0.6:
            top_assessed = sorted(
                redrob.get("skill_assessment_scores", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:2]
            if top_assessed:
                assessed_str = ", ".join(
                    f"{k}={v:.0f}" for k, v in top_assessed
                )
                signals.append(f"verified assessments: {assessed_str}")

        if components.skill_trust_score >= 0.6:
            trusted = sorted(
                parsed_profile.top_skill_trust_scores.items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            top_skills = [k for k, v in trusted if v >= 0.5]
            if top_skills:
                signals.append(
                    "strong skills: " + ", ".join(top_skills)
                )

        if components.title_relevance >= 0.8:
            signals.append("direct title match")
        elif components.title_relevance >= 0.5:
            signals.append("relevant engineering background")
        else:
            signals.append("strong verified skills despite title mismatch")

        if product_fit_score >= 0.35:
            signals.append("product/startup experience")
        if retrieval_depth_score >= 0.35:
            signals.append("deep retrieval/embeddings experience")
        if eval_framework_score >= 0.35:
            signals.append("evaluation/metrics strength")

        if parsed_profile.interview_completion_rate < 0.5:
            signals.append(
                f"ghost risk: {parsed_profile.interview_completion_rate:.0%} interview completion"
            )
        elif parsed_profile.interview_completion_rate >= 0.9:
            signals.append("high interview completion rate")

        if parsed_profile.offer_acceptance_rate >= 0.7:
            signals.append("strong offer acceptance history")
        elif 0 <= parsed_profile.offer_acceptance_rate < 0.3:
            signals.append("low offer acceptance rate")

        if redrob.get('open_to_work_flag', True):
            signals.append("available/open to work")
        else:
            signals.append("not currently open to work")

        required_hits, preferred_hits, neg_hits = self._count_jd_matches(candidate_raw, parsed_profile)
        if self.requirement_profile:
            req_total = len(self.requirement_profile.required_keywords)
            pref_total = len(self.requirement_profile.preferred_keywords)
            if req_total > 0:
                signals.append(
                    f"matched {required_hits}/{req_total} required JD terms"
                )
            if pref_total > 0:
                signals.append(
                    f"matched {preferred_hits}/{pref_total} preferred JD terms"
                )
            if neg_hits > 0:
                signals.append(f"{neg_hits} negative JD signals present")

        if honeypot_penalty < 0.98:
            signals.append("honeypot risk penalty applied")

        s2 = "; ".join(signals[:4]) + "." if signals else "Technical fit to JD."
        s3 = ""

        if self.requirement_profile and self.requirement_profile.hands_on_coding:
            s3 = "Hands-on coding and production delivery are prioritized."

        if self.requirement_profile and self.requirement_profile.leadership_required:
            leadership_note = "Leadership and mentorship experience is also valued." 
            s3 = (s3 + " " + leadership_note).strip()

        reasoning_sentences = [f"{s1}.", s2]
        if s3.strip():
            reasoning_sentences.append(s3)

        return " ".join(re.sub(r'\s+', ' ', sentence).strip() for sentence in reasoning_sentences)

    def save_results(self, results: List[Dict], output_dir: str = None):
        """
        Save ranking results with deterministic tie-breaking.
        
        Note: Tie-breaking is already applied in rank_candidates_fast via 
        sort key: (-final_score, candidate_id). This just writes the results.
        """
        
        import pandas as pd
        
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[2] / 'ranking_output'
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        
        # CSV
        csv_data = []
        for result in results[:100]:  # Ensure exactly 100 rows
            csv_data.append({
                'candidate_id': result['candidate_id'],
                'rank': result['rank'],
                'score': f"{result['final_score']:.4f}",
                'reasoning': result.get('reasoning', '')
            })
        
        csv_df = pd.DataFrame(csv_data, columns=['candidate_id', 'rank', 'score', 'reasoning'])
        csv_path = os.path.join(output_dir, 'submission.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        print(f"  Rows: {len(csv_df)}")
        print(f"  Unique candidates: {csv_df['candidate_id'].nunique()}")
        print(f"  Unique ranks: {csv_df['rank'].nunique()}")

        xlsx_path = os.path.join(output_dir, 'submission.xlsx')
        try:
            csv_df.to_excel(xlsx_path, index=False, engine='openpyxl')
            print(f"Saved XLSX: {xlsx_path}")
        except Exception as exc:
            print(f"Failed to save XLSX: {exc}")

        # JSON
        json_path = os.path.join(output_dir, 'ranking_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(results[:100], f, indent=2)
        print(f"Saved JSON: {json_path}")
