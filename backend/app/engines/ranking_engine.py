"""
Main Ranking Engine
Orchestrates the complete ranking pipeline: retrieval → feature scoring → final ranking.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

from .candidate_profile_parser import CandidateProfileParser, ParsedProfile
from .feature_scorer import FeatureScorer, ScoringComponents
from .embedding_retrieval import EmbeddingRetriever, BM25Retriever


@dataclass
class RankedCandidate:
    """Ranked candidate with scores and reasoning"""
    candidate_id: str
    rank: int
    final_score: float
    components: ScoringComponents
    semantic_similarity: float
    reasoning: Dict[str, Any]


class RankingEngine:
    """Main ranking engine orchestrator"""
    
    def __init__(
        self,
        embedding_model: str = 'BAAI/bge-small-en-v1.5',
        use_bm25_prefilter: bool = True,
        cache_dir: str = './ranking_cache'
    ):
        self.parser = CandidateProfileParser()
        self.scorer = FeatureScorer(self.parser)
        self.embedding_retriever = EmbeddingRetriever(
            model_name=embedding_model,
            cache_dir=cache_dir
        )
        self.bm25_retriever = BM25Retriever()
        self.use_bm25_prefilter = use_bm25_prefilter
        
        self.candidates = None
        self.parsed_profiles = {}
        self.jd_text = None
    
    def load_candidates(self, jsonl_path: str):
        """Load candidate data from JSONL file"""
        print(f"Loading candidates from {jsonl_path}...")
        
        self.candidates = []
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    candidate = json.loads(line)
                    self.candidates.append(candidate)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.candidates)} candidates")
    
    def prepare_jd_text(self, jd: str):
        """Set the job description text for ranking"""
        self.jd_text = jd
    
    def parse_all_profiles(self):
        """Parse all candidate profiles"""
        print("Parsing candidate profiles...")
        
        for candidate in tqdm(self.candidates, desc="Parsing profiles"):
            try:
                parsed = self.parser.parse_candidate(candidate)
                self.parsed_profiles[candidate['candidate_id']] = parsed
            except Exception as e:
                print(f"Error parsing candidate {candidate['candidate_id']}: {e}")
    
    def rank_candidates(
        self,
        top_k: int = 100,
        retrieval_top_k: int = 3000,
        include_explainability: bool = True
    ) -> Tuple[List[RankedCandidate], pd.DataFrame]:
        """Execute complete ranking pipeline
        
        Args:
            top_k: Number of top candidates to return
            retrieval_top_k: Number of candidates to retrieve before scoring
            include_explainability: Whether to generate detailed explanations
            
        Returns:
            Tuple of (ranked_candidates, CSV dataframe)
        """
        
        if not self.candidates:
            raise ValueError("No candidates loaded. Call load_candidates() first.")
        if not self.jd_text:
            raise ValueError("No JD text provided. Call prepare_jd_text() first.")
        
        print("\n" + "="*60)
        print("RANKING PIPELINE START")
        print("="*60)
        
        # Stage 1: Build retrieval indices
        print("\n[Stage 1] Building retrieval indices...")
        self.embedding_retriever.build_index(self.candidates, use_cache=True)
        if self.use_bm25_prefilter:
            self.bm25_retriever.build_index(self.candidates)
        
        # Stage 2: Initial retrieval
        print("\n[Stage 2] Initial retrieval...")
        retrieved_ids, retrieved_scores = self.embedding_retriever.retrieve(
            self.jd_text,
            top_k=retrieval_top_k
        )
        print(f"Retrieved {len(retrieved_ids)} candidates")
        
        # Stage 3: Parse and score retrieved candidates
        print("\n[Stage 3] Scoring candidates...")
        candidates_to_score = [
            c for c in self.candidates
            if c['candidate_id'] in retrieved_ids
        ]
        
        scored_candidates = []
        similarity_map = {cid: score for cid, score in zip(retrieved_ids, retrieved_scores)}
        
        for candidate in tqdm(candidates_to_score, desc="Scoring"):
            candidate_id = candidate['candidate_id']
            
            # Parse profile
            if candidate_id not in self.parsed_profiles:
                parsed_profile = self.parser.parse_candidate(candidate)
                self.parsed_profiles[candidate_id] = parsed_profile
            else:
                parsed_profile = self.parsed_profiles[candidate_id]
            
            # Score components
            semantic_sim = similarity_map.get(candidate_id, 0.0)
            components = self.scorer.score_candidate(
                candidate,
                parsed_profile,
                semantic_similarity=semantic_sim
            )
            
            # Apply disqualifying factors
            final_score = components.final_score
            final_score = self.scorer.apply_disqualifying_factors(
                final_score,
                parsed_profile,
                candidate
            )
            
            scored_candidates.append({
                'candidate_id': candidate_id,
                'final_score': final_score,
                'components': components,
                'semantic_similarity': semantic_sim,
                'parsed_profile': parsed_profile,
                'candidate_data': candidate
            })
        
        # Stage 4: Sort and select top-k
        print("\n[Stage 4] Final ranking...")
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        # Stage 5: Generate explainability
        print("\n[Stage 5] Generating explanations...")
        ranked_candidates = []
        
        for rank, scored in enumerate(top_candidates, 1):
            reasoning = self._generate_reasoning(
                scored['candidate_data'],
                scored['parsed_profile'],
                scored['components'],
                scored['final_score']
            ) if include_explainability else {}
            
            ranked = RankedCandidate(
                candidate_id=scored['candidate_id'],
                rank=rank,
                final_score=scored['final_score'],
                components=scored['components'],
                semantic_similarity=scored['semantic_similarity'],
                reasoning=reasoning
            )
            ranked_candidates.append(ranked)
        
        # Generate CSV
        print("\n[Stage 6] Generating submission CSV...")
        csv_df = self._generate_csv(ranked_candidates)
        
        print("\n" + "="*60)
        print("RANKING PIPELINE COMPLETE")
        print(f"Top candidate: {ranked_candidates[0].candidate_id} (score: {ranked_candidates[0].final_score:.3f})")
        print("="*60 + "\n")
        
        return ranked_candidates, csv_df
    
    def _generate_reasoning(
        self,
        candidate: Dict[str, Any],
        parsed_profile: ParsedProfile,
        components: ScoringComponents,
        final_score: float
    ) -> Dict[str, Any]:
        """Generate detailed reasoning for a ranked candidate"""
        
        reasoning = {
            'strengths': [],
            'concerns': [],
            'key_facts': [],
            'disqualifiers': []
        }
        
        profile = candidate['profile']
        career_history = candidate['career_history']
        redrob = candidate['redrob_signals']
        
        # Strengths
        if components.technical_relevance > 0.8:
            reasoning['strengths'].append(
                f"Strong technical match ({components.technical_relevance:.0%}) for retrieval/ranking systems"
            )
        
        if components.production_experience > 0.8:
            reasoning['strengths'].append(
                f"Proven {parsed_profile.years_experience:.1f} years of production ML experience"
            )
        
        if parsed_profile.github_activity_score > 30:
            reasoning['strengths'].append(
                f"Active GitHub contributor (activity score: {parsed_profile.github_activity_score:.0f})"
            )
        
        if redrob['interview_completion_rate'] > 0.7:
            reasoning['strengths'].append(
                f"Reliable candidate ({redrob['interview_completion_rate']:.0%} interview completion)"
            )
        
        # Extract relevant skills
        relevant_skills = self.parser.extract_relevant_skills(candidate)
        if relevant_skills['retrieval']:
            reasoning['strengths'].append(
                f"Retrieval expertise: {', '.join(relevant_skills['retrieval'][:3])}"
            )
        
        # Career facts
        if career_history:
            most_recent = career_history[0]
            reasoning['key_facts'].append(f"Current/most recent: {most_recent['title']} at {most_recent['company']}")
            reasoning['key_facts'].append(f"Experience: {parsed_profile.years_experience:.1f} years total")
            
            if most_recent['duration_months'] >= 12:
                reasoning['key_facts'].append(
                    f"Current role duration: {most_recent['duration_months']} months"
                )
        
        if profile['years_of_experience'] >= 5:
            reasoning['key_facts'].append(f"Education: {len(candidate.get('education', []))} degree(s)")
        
        # Concerns
        if components.profile_quality < 0.7:
            reasoning['concerns'].append("Profile quality concerns (inconsistencies or flag detected)")
        
        if parsed_profile.is_consulting_only:
            reasoning['concerns'].append(
                "Career primarily at consulting firms (TCS/Infosys/Wipro/etc) without product company experience"
            )
        
        if parsed_profile.years_since_last_coding > 1.0:
            reasoning['concerns'].append(
                f"Limited recent coding ({parsed_profile.years_since_last_coding:.1f}+ years inactive)"
            )
        
        if redrob['notice_period_days'] > 60:
            reasoning['concerns'].append(f"Notice period: {redrob['notice_period_days']} days")
        
        if not redrob['open_to_work_flag']:
            reasoning['concerns'].append("Not marked as open to work")
        
        if len(parsed_profile.timeline_issues) > 0:
            reasoning['concerns'].append(f"Timeline issues: {', '.join(parsed_profile.timeline_issues[:2])}")
        
        # Red flags
        red_flags = self.parser.detect_red_flags(parsed_profile, candidate)
        if red_flags['keyword_stuffing']:
            reasoning['disqualifiers'].append("Potential keyword stuffing (many skills, low endorsements)")
        if red_flags['consulting_only']:
            reasoning['disqualifiers'].append("Consulting-only background without product experience")
        if red_flags['missing_recent_code']:
            reasoning['disqualifiers'].append("No evidence of recent code production")
        
        return reasoning
    
    def _generate_csv(self, ranked_candidates: List[RankedCandidate]) -> pd.DataFrame:
        """Generate submission CSV"""
        
        csv_data = []
        
        for ranked in ranked_candidates:
            row = {
                'candidate_id': ranked.candidate_id,
                'rank': ranked.rank,
                'score': f"{ranked.final_score:.4f}",
                'reasoning': self._format_reasoning_for_csv(ranked.reasoning)
            }
            csv_data.append(row)
        
        return pd.DataFrame(csv_data)
    
    def _format_reasoning_for_csv(self, reasoning: Dict[str, Any]) -> str:
        """Format reasoning for CSV output"""
        
        parts = []
        
        if reasoning.get('strengths'):
            parts.append("Strengths: " + "; ".join(reasoning['strengths'][:2]))
        
        if reasoning.get('concerns'):
            parts.append("Concerns: " + "; ".join(reasoning['concerns'][:1]))
        
        if reasoning.get('key_facts'):
            parts.append("Facts: " + "; ".join(reasoning['key_facts'][:2]))
        
        return " | ".join(parts) if parts else "No additional details"
    
    def save_results(
        self,
        ranked_candidates: List[RankedCandidate],
        csv_df: pd.DataFrame,
        output_dir: str = './ranking_results'
    ):
        """Save ranking results to files
        
        Args:
            ranked_candidates: List of ranked candidates
            csv_df: DataFrame for CSV output
            output_dir: Output directory for results
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON results
        detailed_results = []
        for ranked in ranked_candidates:
            result = {
                'candidate_id': ranked.candidate_id,
                'rank': ranked.rank,
                'score': float(ranked.final_score),
                'components': {
                    'technical_relevance': float(ranked.components.technical_relevance),
                    'production_experience': float(ranked.components.production_experience),
                    'profile_quality': float(ranked.components.profile_quality),
                    'behavioral_engagement': float(ranked.components.behavioral_engagement),
                    'experience_level_fit': float(ranked.components.experience_level_fit),
                    'semantic_similarity': float(ranked.components.semantic_similarity)
                },
                'reasoning': ranked.reasoning
            }
            detailed_results.append(result)
        
        detailed_path = os.path.join(output_dir, 'ranking_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Saved detailed results to {detailed_path}")
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'submission.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved submission CSV to {csv_path}")
