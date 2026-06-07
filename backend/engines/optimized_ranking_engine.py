"""
Optimized Multi-Stage Ranking Engine
Enhanced 5-minute pipeline: BM25 (3000) → FAISS (1000) → CrossEncoder (250) → Scoring (100)
Uses precomputed embeddings for speed.
OPTIMIZATIONS: Threading config, adaptive batching, strict cache-first, memory mapping, resource telemetry
PHASES: 1-CrossEncoder, 2-RecruiterJD, 3-AdvancedScoring, 4-Honeypot, 5-Behavioral, 6-Rebalance,
         7-TieBreaking, 8-Reasoning, 9-Benchmarking, 10-ConfigurableEmbeddings
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os
import faiss
import torch

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

# Configure FAISS threading (torch threading already configured in embedding_precompute)
faiss.omp_set_num_threads(_CPU_COUNT)


class OptimizedRankingEngine:
    """Fast multi-stage ranking: BM25 → FAISS → CrossEncoder → Features → Top 100"""
    
    def __init__(
        self,
        embeddings_cache_dir: str = './embeddings_cache',
        use_precomputed_embeddings: bool = True,
        embedding_model: str = 'BAAI/bge-small-en-v1.5',  # Phase 10: Configurable
        enable_cross_encoder: bool = True,  # Phase 1: Toggle cross-encoder
        enable_honeypot_detection: bool = True  # Phase 4: Toggle honeypot detection
    ):
        self.parser = CandidateProfileParser()
        self.feature_scorer = FeatureScorer(self.parser)
        self.advanced_scorer = AdvancedScorer(self.parser)
        self.bm25_retriever = BM25Retriever()
        
        # Phase 10: Configurable embedding model
        self.embedding_model_name = embedding_model
        self.precomputer = EmbeddingPrecomputer(
            model_name=embedding_model,
            cache_dir=embeddings_cache_dir
        )
        
        # Phase 1: Cross-encoder reranker
        self.cross_encoder_reranker = CrossEncoderReranker() if enable_cross_encoder else None
        
        # Phase 2: Recruiter-centric JD parser
        self.recruiter_jd_parser = RecruiterCentricJDParser()
        self.requirement_profile = None
        
        # Phase 3: Advanced scoring components
        self.career_trajectory_analyzer = CareerTrajectoryAnalyzer()
        self.product_company_scorer = ProductCompanyScorer()
        self.retrieval_depth_scorer = RetrievalDepthScorer()
        self.evaluation_framework_scorer = EvaluationFrameworkScorer()
        
        # Phase 4: Honeypot detection
        self.honeypot_detector = HoneypotDetector() if enable_honeypot_detection else None
        
        self.embeddings_cache_dir = self.precomputer.cache_dir
        self.faiss_index_path = self.embeddings_cache_dir / 'precomputed_embeddings_faiss.index'
        self.use_precomputed = use_precomputed_embeddings
        
        # Phase 9: Benchmarking telemetry
        self.benchmark_telemetry = {}

        # PHASE 8: Print resource utilization at startup
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
        Set job description and extract JD signals (Phase 2: Recruiter-centric).
        Parse requirements, preferences, and negative signals.
        """
        self.jd_text = jd.strip()
        
        # Phase 2: Use recruiter-centric parser
        self.requirement_profile = self.recruiter_jd_parser.parse_jd(self.jd_text)
        
        # Legacy keywords for backward compatibility
        self.jd_keywords = self._extract_jd_keywords(self.jd_text)
        
        # Print extracted requirements (Phase 2 telemetry)
        print("\n[Phase 2] JD Parsing Results:")
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
        keywords = [phrase for phrase in candidate_phrases if phrase in text]
        if not keywords:
            keywords = re.findall(r"\b[ a-z]{3,}\b", text)[:10]
        return list(dict.fromkeys(keywords))
    
    def _load_or_build_faiss_index(self):
        """Load cached embeddings and FAISS index, or build them once (PHASE 4-5: Strict cache-first, PHASE 6: Memory mapping)."""
        self.candidate_ids = [c['candidate_id'] for c in self.candidates]

        if self.use_precomputed:
            try:
                print("[PERFORMANCE] Loading precomputed embeddings (CACHE FIRST)...")
                self.candidate_embeddings, self.candidate_ids, metadata = \
                    self.precomputer.load_precomputed_embeddings()
                # PHASE 6: Use float32 consistently, avoid unnecessary copies
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

        # If precomputed embeddings are not being used, load model but DON'T compute embeddings in ranking
        # (PHASE 5: Ranking should only embed JD once, not candidate embeddings)
        print("[ERROR] Ranking engine requires precomputed embeddings (use_precomputed_embeddings=True)")
        raise RuntimeError(
            "Candidate embeddings must be precomputed. Run embedding_precompute.py first. "
            "Ranking must not generate candidate embeddings."
        )
    
    def rank_candidates_fast(
        self,
        top_k: int = 100,
        bm25_top_k: int = 3000,  # Phase 1: Increased from 2000
        faiss_top_k: int = 1000,  # Phase 1: Increased from 500
        cross_encoder_top_k: int = 250  # Phase 1: New cross-encoder reranking stage
    ) -> Tuple[List[Dict], Any]:
        """
        Execute enhanced multi-stage ranking pipeline with all phases.
        
        Pipeline:
        Stage 1: Load embeddings + BM25 index
        Stage 2: BM25 retrieval → 3000 (increased for coverage)
        Stage 3: FAISS semantic retrieval → 1000 (increased)
        Stage 4: Cross-Encoder reranking → 250 (Phase 1: New)
        Stage 5: Feature scoring with all components → 100
        
        Phases integrated:
        - Phase 1: Cross-encoder reranking
        - Phase 2: Recruiter-centric JD parsing
        - Phase 3: Advanced scoring components
        - Phase 4: Honeypot detection
        - Phase 5: Behavioral signals (in feature scorer)
        - Phase 6: Rebalanced scoring weights
        - Phase 8: Improved reasoning
        - Phase 9: Benchmarking telemetry
        - Phase 10: Configurable embeddings
        """
        
        if not self.candidates or not self.jd_text:
            raise ValueError("Load candidates and JD first")
        
        # Phase 9: Benchmarking - record start times
        pipeline_start = datetime.now()
        stage_timings = {}
        
        print("\n" + "="*70)
        print("ENHANCED MULTI-STAGE RANKING PIPELINE (Phases 1-10)")
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
        
        # PHASE 1: Stage 3.5 - Cross-Encoder Reranking
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
        
        # Stage 4: Feature scoring with all new components
        print(f"\n[Stage 4] Feature scoring (Phases 3-6) → Top {top_k}...")
        t0 = datetime.now()
        
        scored_candidates = []
        for i, candidate_id in enumerate(faiss_ids):
            if i % 100 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  Scoring {i}/{len(faiss_ids)} ({elapsed:.1f}s)...")
            
            # Lookup candidate by ID
            candidate = self.candidates_by_id.get(candidate_id)
            if not candidate:
                continue
            
            # Parse profile
            if candidate_id not in self.parsed_profiles:
                parsed = self.parser.parse_candidate(candidate)
                self.parsed_profiles[candidate_id] = parsed
            else:
                parsed = self.parsed_profiles[candidate_id]
            
            # Phase 3: Score with all components
            semantic_sim = faiss_score_map.get(candidate_id, 0.0)
            components = self.feature_scorer.score_candidate(
                candidate,
                parsed,
                semantic_similarity=semantic_sim,
                advanced_scorer=self.advanced_scorer
            )
            
            # Phase 3: Add new component scores
            career_traj_score = self.career_trajectory_analyzer.score(parsed, candidate)
            product_fit_score = self.product_company_scorer.score(parsed, candidate)
            retrieval_depth_score = self.retrieval_depth_scorer.score(candidate)
            eval_framework_score = self.evaluation_framework_scorer.score(candidate)
            
            # Phase 4: Honeypot detection
            honeypot_penalty = 1.0
            if self.honeypot_detector:
                risk_score = self.honeypot_detector.calculate_risk_score(parsed, candidate)
                honeypot_penalty = self.honeypot_detector.get_penalty_multiplier(risk_score)
            
            # Phase 6: Rebalanced scoring with new weights
            base_score = (
                components.technical_relevance * 0.20 +
                components.production_experience * 0.15 +
                components.semantic_similarity * 0.25 +
                components.behavioral_engagement * 0.05 +
                components.experience_level_fit * 0.05 +
                career_traj_score * 0.05 +
                product_fit_score * 0.05 +
                retrieval_depth_score * 0.10 +
                eval_framework_score * 0.10
            )
            
            # Apply profile quality as multiplier and honeypot penalty
            final_score = base_score * components.profile_quality_multiplier * honeypot_penalty
            
            # Apply additional disqualifiers
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
        
        stage_timings['feature_scoring'] = (datetime.now() - t0).total_seconds()
        print(f"  ✓ Scored {len(scored_candidates)} candidates ({stage_timings['feature_scoring']:.1f}s)")
        
        # Stage 5: Sort and finalize
        print(f"\n[Stage 5] Finalizing top {top_k}...")
        t0 = datetime.now()
        
        # Phase 7: Apply deterministic tie-breaking before ranking
        # Sort by: (score DESC, candidate_id ASC)
        scored_candidates.sort(
            key=lambda x: (-x['final_score'], x['candidate_id'])
        )
        top_candidates = scored_candidates[:top_k]
        
        results = []
        for rank, scored in enumerate(top_candidates, 1):
            # Phase 8: Improved reasoning
            reasoning_text = self._generate_improved_reasoning(
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
                    'technical_relevance': float(scored['components'].technical_relevance),
                    'production_experience': float(scored['components'].production_experience),
                    'profile_quality_multiplier': float(scored['components'].profile_quality_multiplier),
                    'behavioral_engagement': float(scored['components'].behavioral_engagement),
                    'experience_level_fit': float(scored['components'].experience_level_fit),
                    'evaluation_framework_score': float(scored['components'].evaluation_framework_score),
                    'product_mindset_score': float(scored['components'].product_mindset_score),
                    'semantic_similarity': float(scored['components'].semantic_similarity),
                    'career_trajectory_score': float(scored.get('career_trajectory_score', 0.0)),
                    'product_fit_score': float(scored.get('product_fit_score', 0.0)),
                    'retrieval_depth_score': float(scored.get('retrieval_depth_score', 0.0)),
                    'eval_framework_score': float(scored.get('eval_framework_score', 0.0))
                },
                'reasoning': reasoning_text
            }
            results.append(result)
        
        stage_timings['finalize'] = (datetime.now() - t0).total_seconds()
        total_time = (datetime.now() - pipeline_start).total_seconds()
        stage_timings['total'] = total_time
        
        # Phase 9: Print benchmarking telemetry
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
        print("="*70 + "\n")
        
        # Phase 9: Store telemetry
        self.benchmark_telemetry = stage_timings
        
        return results, scored_candidates
    
    def _faiss_retrieve_from_pool(
        self,
        query_text: str,
        pool_ids: List[str],
        top_k: int = 500
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k from a specific pool using precomputed candidate embeddings (PHASE 7: Vectorized operations)."""

        if self.precomputer.model is None:
            self.precomputer.load_model()

        query_embedding = self.precomputer.model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1).astype('float32', copy=False)

        # PHASE 7: Vectorized pool indexing (avoid Python loops for numerical work)
        pool_indices = np.array([
            self.candidate_id_to_index[pool_id]
            for pool_id in pool_ids
            if pool_id in self.candidate_id_to_index
        ], dtype=np.int64)

        if len(pool_indices) == 0:
            return [], []

        if self.candidate_embeddings is None:
            raise ValueError("Candidate embeddings are required for FAISS retrieval.")

        # PHASE 7: Vectorized dot product (numpy operation, not Python loop)
        pool_embeddings = self.candidate_embeddings[pool_indices]
        similarities = np.dot(pool_embeddings, query_embedding.T).flatten()

        # PHASE 7: Vectorized top-k selection
        top_indices_in_pool = np.argsort(similarities)[-top_k:][::-1]
        retrieved_ids = [pool_ids[int(i)] for i in top_indices_in_pool]
        retrieved_scores = [float(similarities[i]) for i in top_indices_in_pool]
        return retrieved_ids, retrieved_scores

    def _generate_improved_reasoning(
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
        """
        Phase 8: Generate factual, candidate-specific reasoning.
        
        Rules:
        - Reference actual data: years, titles, skills
        - Use different templates for different candidates
        - Max 2 sentences
        - Never hallucinate
        - Call out concerns if present
        """
        profile = candidate_raw.get('profile', {})
        skills = [s.get('name', '').strip() for s in candidate_raw.get('skills', []) if s.get('name')]
        top_skills = ', '.join(skills[:3]) if skills else ''
        
        company = parsed_profile.current_company or parsed_profile.most_recent_role_company or ''
        title = parsed_profile.current_title or parsed_profile.most_recent_role_title or ''
        years_exp = parsed_profile.years_experience or 0
        
        factual_signals = []
        
        # Build fact-based first sentence
        if years_exp > 0 and title:
            first_sentence = f"{years_exp:.0f}-year {'AI/ML ' if any(x in title.lower() for x in ['ml', 'ai', 'data']) else ''}engineer with {title.lower()} background"
        elif title and company:
            first_sentence = f"{'AI/ML engineer' if any(x in title.lower() for x in ['ml', 'ai']) else title.title()} at {company}"
        else:
            first_sentence = "Strong candidate profile"
        
        # Build specifics based on actual scores
        if retrieval_depth_score >= 0.5:
            factual_signals.append("proven retrieval and vector DB expertise")
        
        if eval_framework_score >= 0.5:
            factual_signals.append("strong ranking evaluation and metrics knowledge")
        
        if product_fit_score >= 0.6:
            factual_signals.append("product company and startup background")
        
        if career_trajectory_score >= 0.6:
            factual_signals.append("strong career progression in ML/ranking roles")
        
        if components.production_experience >= 0.7:
            factual_signals.append("demonstrated production deployment experience")
        
        if components.behavioral_engagement >= 0.6:
            factual_signals.append("high recruiter engagement signals")
        
        # Add concerns if present
        if honeypot_penalty < 0.9:
            factual_signals.append("profile quality concerns noted")
        
        if not factual_signals:
            # Generic fallback
            if top_skills:
                factual_signals.append(f"skills in {top_skills}")
            else:
                factual_signals.append("technical fit to job requirements")
        
        second_sentence = '; '.join(factual_signals[:2]).capitalize() + '.'
        
        return f"{first_sentence}. {second_sentence}"

    def save_results(self, results: List[Dict], output_dir: str = './ranking_output'):
        """
        Phase 7: Save ranking results with tie-breaking compliance.
        
        Note: Tie-breaking is already applied in rank_candidates_fast via 
        sort key: (-final_score, candidate_id). This just writes the results.
        """
        
        import pandas as pd
        
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
        
        # JSON
        json_path = os.path.join(output_dir, 'ranking_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(results[:100], f, indent=2)
        print(f"Saved JSON: {json_path}")
