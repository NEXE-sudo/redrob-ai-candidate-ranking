"""
Feature Scoring Module
Implements the 5-component scoring system from the ranking strategy.
"""

import re
from typing import Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from .candidate_profile_parser import ParsedProfile, CandidateProfileParser


@dataclass
class ScoringComponents:
    """Individual component scores"""
    technical_relevance: float
    production_experience: float
    profile_quality_multiplier: float  # NOW A MULTIPLIER (was additive)
    behavioral_engagement: float
    experience_level_fit: float
    evaluation_framework_score: float = 0.0
    product_mindset_score: float = 0.0
    semantic_similarity: float = 0.0  # Will be set from embeddings
    
    @property
    def final_score(self) -> float:
        """Calculate weighted final score with profile_quality as multiplier
        
        Formula:
        base_score = (technical * 0.35 + production * 0.25 + engagement * 0.15 +
                      experience * 0.10 + eval_framework * 0.10 + product_mindset * 0.05 +
                      semantic * 0.05)
        final_score = base_score * profile_quality_multiplier
        
        This ensures suspicious profiles cannot rank highly through keyword matching.
        """
        # Calculate base score (before profile quality multiplier)
        base_score = (
            self.technical_relevance * 0.35 +
            self.production_experience * 0.25 +
            self.behavioral_engagement * 0.15 +
            self.experience_level_fit * 0.10 +
            self.evaluation_framework_score * 0.10 +
            self.product_mindset_score * 0.05 +
            self.semantic_similarity * 0.05
        )
        
        # Apply profile_quality as MULTIPLIER (not additive)
        final_score = base_score * self.profile_quality_multiplier
        
        return min(max(final_score, 0.0), 1.0)


class FeatureScorer:
    """Score candidates on 5 dimensions"""
    
    # Keywords organized by tier
    TIER_1_KEYWORDS = {
        'embeddings', 'retrieval', 'vector_db', 'vector database', 'ranking',
        'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'semantic search',
        'bge', 'sentence transformers', 'opensearch', 'dense retrieval'
    }
    
    TIER_2_KEYWORDS = {
        'llm', 'fine-tuning', 'lora', 'qlora', 'learning-to-rank', 'ltr',
        'xgboost', 'lambdarank', 'evaluation framework', 'ndcg', 'mrr', 'map',
        'production ml', 'inference', 'model deployment'
    }
    
    TIER_3_KEYWORDS = {
        'python', 'ml', 'ai', 'nlp', 'information retrieval', 'search',
        'recommendation', 'machine learning', 'data science'
    }
    
    # Scale metrics
    SCALE_KEYWORDS = {
        '100k', '1m', '10m', '100m', '1b', 'million', 'billion',
        'real-time', 'qps', 'latency', 'throughput'
    }
    
    PRODUCTION_KEYWORDS = {
        'deployed', 'shipped', 'live', 'production', 'user-facing',
        'production-grade', 'production system'
    }
    
    def __init__(self, parser: CandidateProfileParser = None):
        self.parser = parser or CandidateProfileParser()
    
    def score_candidate(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        semantic_similarity: float = 0.0,
        reference_date: datetime = None,
        advanced_scorer = None  # Optional AdvancedScorer instance
    ) -> ScoringComponents:
        """Score candidate on all dimensions including new components"""
        
        if reference_date is None:
            reference_date = datetime.now()
        
        technical_relevance = self._score_technical_relevance(candidate_raw, parsed_profile)
        production_experience = self._score_production_experience(parsed_profile, candidate_raw)
        profile_quality_multiplier = self._score_profile_quality_multiplier(parsed_profile, candidate_raw)
        behavioral_engagement = self._score_behavioral_engagement(candidate_raw, reference_date)
        experience_level_fit = self._score_experience_level_fit(parsed_profile)
        
        # New components (use advanced_scorer if provided)
        evaluation_framework_score = 0.0
        product_mindset_score = 0.0
        
        if advanced_scorer:
            evaluation_framework_score = advanced_scorer.score_evaluation_framework(candidate_raw)
            product_mindset_score = advanced_scorer.score_product_mindset(candidate_raw, parsed_profile)
        
        components = ScoringComponents(
            technical_relevance=technical_relevance,
            production_experience=production_experience,
            profile_quality_multiplier=profile_quality_multiplier,
            behavioral_engagement=behavioral_engagement,
            experience_level_fit=experience_level_fit,
            evaluation_framework_score=evaluation_framework_score,
            product_mindset_score=product_mindset_score,
            semantic_similarity=semantic_similarity
        )
        
        return components
    
    def _score_technical_relevance(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> float:
        """Score 1A + 1B + 1C: Keywords, Scale, Recency"""
        
        # 1A: Keyword score
        keyword_score = self._keyword_match_score(candidate_raw, parsed_profile)
        
        # 1B: Scale signal
        scale_score = self._scale_signal_score(candidate_raw)
        
        # Combine: 70% keywords + 30% scale
        technical_base = keyword_score * 0.7 + scale_score * 0.3
        
        # 1C: Apply recency penalty/bonus
        recency_penalty = self._calculate_recency_penalty(parsed_profile)
        technical_relevance = technical_base * (1.0 - recency_penalty)
        
        return min(max(technical_relevance, 0.0), 1.0)
    
    def _keyword_match_score(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> float:
        """Count tier-weighted keywords in profile and career text"""
        
        # Concatenate all relevant text
        profile_text = (
            candidate_raw['profile'].get('summary', '').lower() + ' ' +
            candidate_raw['profile'].get('headline', '').lower() + ' ' +
            candidate_raw['profile'].get('current_title', '').lower()
        )
        
        career_text = ' '.join([
            role['title'].lower() + ' ' + role['description'].lower()
            for role in candidate_raw['career_history']
        ])
        
        all_text = profile_text + ' ' + career_text
        
        # Count skill names too
        skill_text = ' '.join([s['name'].lower() for s in candidate_raw['skills']])
        all_text = all_text + ' ' + skill_text
        
        # Count keywords by tier
        tier1_count = sum(1 for keyword in self.TIER_1_KEYWORDS if keyword in all_text)
        tier2_count = sum(1 for keyword in self.TIER_2_KEYWORDS if keyword in all_text)
        tier3_count = sum(1 for keyword in self.TIER_3_KEYWORDS if keyword in all_text)
        
        # Expected counts (from JD analysis)
        tier1_expected = 4  # embeddings, retrieval, vector DB, ranking
        tier2_expected = 5  # LLM, fine-tuning, eval, etc
        tier3_expected = 5  # Python, ML, AI, etc
        
        # Normalized score
        keyword_score = (
            0.4 * min(tier1_count / max(tier1_expected, 1), 1.0) +
            0.35 * min(tier2_count / max(tier2_expected, 1), 1.0) +
            0.25 * min(tier3_count / max(tier3_expected, 1), 1.0)
        )
        
        return min(max(keyword_score, 0.0), 1.0)
    
    def _scale_signal_score(self, candidate_raw: Dict[str, Any]) -> float:
        """Check for production evidence and scale metrics"""
        
        score = 0.0
        
        # Check career descriptions for production evidence
        career_text = ' '.join([
            role['description'].lower()
            for role in candidate_raw['career_history']
        ])
        
        # Production deployment
        if any(keyword in career_text for keyword in self.PRODUCTION_KEYWORDS):
            score += 0.4
        
        # Scale metrics
        if any(metric in career_text for metric in self.SCALE_KEYWORDS):
            score += 0.4
        
        # Company size (most recent)
        most_recent_role = candidate_raw['career_history'][0] if candidate_raw['career_history'] else None
        if most_recent_role:
            company_size = most_recent_role['company_size']
            size_numeric = CandidateProfileParser.COMPANY_SIZE_NUMERIC.get(company_size, 0)
            if size_numeric >= 500:
                score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_recency_penalty(self, parsed_profile: ParsedProfile) -> float:
        """Apply penalty for stale experience"""
        
        penalty = 0.0
        
        # If currently in role for 3+ months, no penalty
        if parsed_profile.most_recent_role_ended_months_ago == 0.0:
            penalty = 0.0
        # Recently ended (0-6 months)
        elif parsed_profile.most_recent_role_ended_months_ago <= 6:
            penalty = 0.0
        # Moderately old (6-18 months)
        elif parsed_profile.most_recent_role_ended_months_ago <= 18:
            penalty = 0.1
        # Older (18-36 months)
        elif parsed_profile.most_recent_role_ended_months_ago <= 36:
            penalty = 0.2
        # Very old
        else:
            penalty = 0.3
        
        return penalty
    
    def _score_production_experience(
        self,
        parsed_profile: ParsedProfile,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Score based on career depth and consistency"""
        
        # 2A: Career depth
        ml_months = parsed_profile.career_depth_months.get('ml_months', 0)
        production_months = parsed_profile.career_depth_months.get('production_months', 0)
        
        # Need at least 12 months ML + production
        deep_experience_score = 0.0
        if ml_months >= 12:
            deep_experience_score += 0.5
        if ml_months >= 24:
            deep_experience_score += 0.3
        if production_months >= 12:
            deep_experience_score += 0.2
        
        deep_experience_score = min(deep_experience_score, 1.0)
        
        # 2B: Role consistency check
        current_title = parsed_profile.current_title.lower()
        is_tech_focused = any(
            term in current_title 
            for term in CandidateProfileParser.ENGINEERING_TITLES
        )
        
        consistency_penalty = 0.0
        
        # Check for title-chaser pattern (jumping for seniority)
        titles = [role['title'].lower() for role in candidate_raw['career_history']]
        seniority_jumps = 0
        for title in titles:
            if any(level in title for level in ['principal', 'staff', 'senior', 'manager']):
                seniority_jumps += 1
        
        if seniority_jumps > len(titles) * 0.5:  # More than 50% high-level titles
            consistency_penalty += 0.1
        
        # Check for narrative consistency
        if not is_tech_focused and parsed_profile.years_experience >= 5:
            consistency_penalty += 0.15
        
        production_experience = (deep_experience_score * 0.8 + 0.2) * (1.0 - consistency_penalty)
        
        return min(max(production_experience, 0.0), 1.0)
    
    def _score_profile_quality_multiplier(
        self,
        parsed_profile: ParsedProfile,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Score profile consistency, realism as MULTIPLIER (0.2 to 1.0)
        
        This is now a multiplier applied to the base score, ensuring that
        suspicious profiles (even with good keywords) cannot rank highly.
        
        Returns:
            Multiplier 0.2-1.0 (default 1.0 for clean profiles)
        """
        
        multiplier = 1.0
        
        # Timeline analysis - DEDUCTION
        if len(parsed_profile.timeline_issues) > 0:
            multiplier -= min(len(parsed_profile.timeline_issues) * 0.15, 0.3)
        
        # Skill realism - MAJOR RED FLAG
        if parsed_profile.skill_counts > 50:
            multiplier -= 0.25  # Strong penalty for likely padding
        
        # Check for unrealistic skill combinations - MAJOR RED FLAG
        skill_names = [s.lower() for s in parsed_profile.skill_names]
        if self._check_unrealistic_skills(skill_names):
            multiplier -= 0.30  # Strong penalty for suspicious profiles
        
        # Profile completeness - BONUS/PENALTY
        if parsed_profile.profile_completeness < 40:
            multiplier -= 0.2  # Very incomplete = suspicious
        elif parsed_profile.profile_completeness < 60:
            multiplier -= 0.1  # Incomplete
        elif parsed_profile.profile_completeness >= 80:
            multiplier += 0.05  # Bonus for very complete
        
        # Verification signals - BONUS (trust signals)
        redrob = candidate_raw['redrob_signals']
        verify_count = sum([
            redrob.get('verified_email', False),
            redrob.get('verified_phone', False),
            redrob.get('linkedin_connected', False)
        ])
        multiplier += verify_count * 0.10  # Bonus per verification
        
        # Ensure multiplier is in valid range [0.2, 1.0]
        # Floor at 0.2 so even bad profiles aren't completely eliminated
        # (disqualifiers handled separately)
        multiplier = min(max(multiplier, 0.2), 1.0)
        
        return multiplier
    
    def _check_unrealistic_skills(self, skill_names: list) -> bool:
        """Detect unrealistic skill combinations"""
        
        # Example: [Photoshop, FAISS, Kubernetes, Accounting] is suspicious
        design_skills = {'photoshop', 'figma', 'sketch', 'adobe xd'}
        infra_skills = {'kubernetes', 'docker', 'terraform', 'cloudformation'}
        non_tech_skills = {'accounting', 'seo', 'sales', 'marketing', 'project management'}
        ai_ml_skills = {'faiss', 'milvus', 'embeddings', 'ranking', 'nlp', 'deep learning'}
        
        has_design = any(s in skill_names for s in design_skills)
        has_infra = any(s in skill_names for s in infra_skills)
        has_non_tech = any(s in skill_names for s in non_tech_skills)
        has_ai_ml = any(s in skill_names for s in ai_ml_skills)
        
        # Suspicious combo: all design + infra + non-tech + AI/ML
        suspicious_combos = sum([has_design, has_infra, has_non_tech, has_ai_ml])
        
        return suspicious_combos >= 3 and len(skill_names) > 30
    
    def _score_behavioral_engagement(
        self,
        candidate_raw: Dict[str, Any],
        reference_date: datetime
    ) -> float:
        """Score availability and engagement signals"""
        
        redrob = candidate_raw['redrob_signals']
        score = 1.0
        
        # Availability: open_to_work flag
        if not redrob.get('open_to_work_flag', True):
            score *= 0.5
        
        # Notice period
        notice_days = redrob.get('notice_period_days', 30)
        if notice_days <= 30:
            pass  # Full score
        elif notice_days <= 60:
            score *= 0.95
        else:
            score *= min(0.8, max(0.3, 1.0 - (notice_days - 60) / 100))
        
        # Engagement signals
        response_rate = redrob.get('recruiter_response_rate', 0.0)
        if response_rate > 0.5:
            score *= 1.02
        
        saved_recruiters = redrob.get('saved_by_recruiters_30d', 0)
        if saved_recruiters > 5:
            score *= 1.02
        
        # Market activity
        search_appearance = redrob.get('search_appearance_30d', 0)
        if search_appearance > 100:
            score *= 1.02
        
        # GitHub activity
        github_score = redrob.get('github_activity_score', -1)
        if github_score > 30:
            score *= 1.05
        
        # Profile completeness
        if redrob.get('profile_completeness_score', 0) < 60:
            score *= 0.85
        
        return min(max(score, 0.0), 1.0)
    
    def _score_experience_level_fit(self, parsed_profile: ParsedProfile) -> float:
        """Score experience in target band (5-9 years)"""
        
        exp = parsed_profile.years_experience
        
        if exp < 3:
            return 0.4  # Too junior
        elif exp < 5:
            return 0.7  # Below band
        elif exp <= 9:
            return 1.0  # Perfect band
        elif exp <= 12:
            return 0.95  # Slightly over
        elif exp <= 15:
            return 0.85  # Over band
        else:
            return 0.7  # Way over (stagnation risk)
    
    def apply_disqualifying_factors(
        self,
        final_score: float,
        parsed_profile: ParsedProfile,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Apply hard disqualifiers that drastically reduce score"""
        
        multiplier = 1.0
        
        redrob = candidate_raw['redrob_signals']
        
        # Pure research background (no production deployment)
        career_text = ' '.join([
            role['description'].lower()
            for role in candidate_raw['career_history']
        ])
        
        if 'research' in ' '.join([role['title'].lower() for role in candidate_raw['career_history']]) * 3:
            if not any(keyword in career_text for keyword in CandidateProfileParser.PRODUCTION_KEYWORDS):
                multiplier *= 0.1
        
        # Pure consulting background (TCS/Infosys etc)
        if parsed_profile.is_consulting_only and parsed_profile.years_experience >= 5:
            multiplier *= 0.1
        
        # No recent code (18+ months ago)
        if parsed_profile.years_since_last_coding > 1.5:
            multiplier *= 0.3
        
        # Not available / not open to work
        if not redrob.get('open_to_work_flag', False) and redrob.get('recruiter_response_rate', 0) < 0.05:
            multiplier *= 0.2
        
        return final_score * multiplier
