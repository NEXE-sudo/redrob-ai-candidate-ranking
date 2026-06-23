"""
Advanced Ranking Components - Phase 3+
New scoring dimensions: career trajectory, product company, retrieval depth, evaluation depth.
"""

from typing import Dict, Any
from datetime import datetime
import re

from .candidate_profile_parser import ParsedProfile


class CareerTrajectoryAnalyzer:
    """Score positive career progression trajectories"""
    
    POSITIVE_PROGRESSIONS = [
        # ML Engineer progressions
        ('junior ml', 'ml engineer', 'senior ml'),
        ('machine learning', 'ml engineer', 'senior ml engineer'),
        ('data scientist', 'ml engineer', 'senior ml engineer'),
        
        # Ranking/Retrieval progressions
        ('ranking engineer', 'senior ranking engineer', 'staff ranking engineer'),
        ('retrieval engineer', 'senior retrieval engineer', 'staff ranking engineer'),
        ('search engineer', 'senior search engineer', 'staff engineer'),
        
        # Recommendation progressions
        ('recommendation engineer', 'senior recommendation engineer', 'staff engineer'),
        ('ml engineer', 'search engineer', 'ranking engineer'),
        
        # NLP progressions
        ('nlp engineer', 'senior nlp engineer', 'staff engineer'),
    ]
    
    def score(self, parsed_profile: ParsedProfile, candidate: Dict[str, Any]) -> float:
        """Score career trajectory (0-1)"""
        if not parsed_profile.career_history or len(parsed_profile.career_history) < 2:
            return 0.5  # Neutral for short careers
        
        score = 0.0
        
        # Check for positive progressions
        titles = [role.get('title', '').lower() for role in parsed_profile.career_history]
        
        for progression in self.POSITIVE_PROGRESSIONS:
            # Check if any two titles match this progression (in order)
            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    if self._progression_matches(titles[i], titles[j], progression):
                        score += 0.3
                        break
        
        # Penalize lateral moves or moves to lower levels
        for i in range(len(titles) - 1):
            if self._is_downgrade(titles[i], titles[i + 1]):
                score -= 0.15
        
        return min(max(score, 0.0), 1.0)
    
    def _progression_matches(self, title1: str, title2: str, progression: tuple) -> bool:
        """Check if two titles match a progression"""
        for prog_title in progression:
            if prog_title.lower() in title1:
                for later_title in progression:
                    if prog_title != later_title and later_title.lower() in title2:
                        return True
        return False
    
    def _is_downgrade(self, title1: str, title2: str) -> bool:
        """Check if move is a downgrade in seniority"""
        senior_words = {'senior', 'staff', 'principal', 'lead'}
        junior_words = {'junior', 'associate', 'intern'}
        
        has_senior_1 = any(w in title1 for w in senior_words)
        has_senior_2 = any(w in title2 for w in senior_words)
        has_junior_2 = any(w in title2 for w in junior_words)
        
        return has_senior_1 and (has_junior_2 or not has_senior_2)


class ProductCompanyScorer:
    """Score experience at product-focused companies"""
    
    PRODUCT_KEYWORDS = {
        'product', 'startup', 'scale-up', 'saas', 'tech',
        'marketplace', 'platform', 'app', 'software',
        'ai company', 'ml company'
    }
    
    PRODUCT_COMPANIES = {
        'google', 'meta', 'amazon', 'microsoft', 'apple',
        'openai', 'anthropic', 'perplexity', 'mistral',
        'stripe', 'airbnb', 'uber', 'lyft', 'doordash',
        'spotify', 'linkedin', 'pinterest', 'dropbox',
        'databricks', 'hugging face', 'together', 'anyscale'
    }
    
    def score(self, parsed_profile: ParsedProfile, candidate: Dict[str, Any]) -> float:
        """Score product company experience (0-1)"""
        if not parsed_profile.career_history:
            return 0.0
        
        score = 0.0
        
        for role in parsed_profile.career_history:
            company = role.get('company', '').lower()
            
            # Check against known product companies
            if any(pc in company for pc in self.PRODUCT_COMPANIES):
                score += 0.2
            
            # Check for product company signals
            elif any(kw in company for kw in self.PRODUCT_KEYWORDS):
                score += 0.1
        
        return min(max(score, 0.0), 1.0)


class RetrievalDepthScorer:
    """Score depth of retrieval/vector DB experience"""
    
    RETRIEVAL_TECH = {
        'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant',
        'elasticsearch', 'opensearch', 'vector', 'embedding',
        'dense retrieval', 'semantic search', 'rag'
    }
    
    def score(self, candidate: Dict[str, Any]) -> float:
        """Score retrieval technology depth (0-1)"""
        score = 0.0
        
        # Check skills
        for skill in candidate.get('skills', []):
            skill_name = skill.get('name', '').lower()
            if any(tech in skill_name for tech in self.RETRIEVAL_TECH):
                score += 0.15
        
        # Check career history
        for role in candidate.get('career_history', []):
            description = (role.get('description', '') + ' ' + role.get('title', '')).lower()
            for tech in self.RETRIEVAL_TECH:
                if tech in description:
                    score += 0.2
                    break  # Only count once per role
        
        return min(max(score, 0.0), 1.0)


class EvaluationFrameworkScorer:
    """Score depth of ML evaluation and metrics experience"""
    
    EVAL_KEYWORDS = {
        'ndcg', 'mrr', 'map', 'eval', 'evaluation',
        'benchmark', 'metric', 'offline evaluation',
        'online evaluation', 'a/b test', 'a/b testing',
        'experiment', 'statistical significance',
        'ranking metric', 'retrieval metric'
    }
    
    def score(self, candidate: Dict[str, Any]) -> float:
        """Score evaluation framework depth (0-1)"""
        score = 0.0
        
        # Check skills
        for skill in candidate.get('skills', []):
            skill_name = skill.get('name', '').lower()
            if any(keyword in skill_name for keyword in self.EVAL_KEYWORDS):
                score += 0.15
        
        # Check career history
        for role in candidate.get('career_history', []):
            description = (role.get('description', '') + ' ' + role.get('title', '')).lower()
            for keyword in self.EVAL_KEYWORDS:
                if keyword in description:
                    score += 0.2
                    break
        
        return min(max(score, 0.0), 1.0)


class HoneypotDetector:
    """
    Detect suspicious profiles that may be honeypots or low-quality.
    Apply penalty multiplier to final score.
    """
    
    def calculate_risk_score(self, parsed_profile: ParsedProfile, candidate: Dict[str, Any]) -> float:
        """
        Calculate honeypot risk (0-1).
        Higher score = higher risk = larger penalty.
        """
        risk = 0.0
        
        # Check for impossible timelines (overlapping employment)
        if self._has_overlapping_employment(parsed_profile):
            risk += 0.25
        
        # Check for unrealistic experience claims
        if self._has_unrealistic_experience(parsed_profile):
            risk += 0.20
        
        # Check for suspicious skill distribution
        if self._has_suspicious_skills(candidate):
            risk += 0.15
        
        # Check for career inconsistencies
        if self._has_career_inconsistency(parsed_profile):
            risk += 0.15
        
# Check for excessive expert skills / zero-duration mastery claims
        if self._has_excessive_expert_skills(candidate):
            risk += 0.20

        # Check for expert/mastery skills with zero reported duration
        if self._has_mastery_without_duration(candidate):
            risk += 0.15
        
        # Check for generic profile
        if self._is_generic_profile(parsed_profile, candidate):
            risk += 0.05
        
        return min(max(risk, 0.0), 1.0)
    
    def get_penalty_multiplier(self, risk_score: float) -> float:
        """
        Convert risk score to penalty multiplier.
        0.0 risk = 1.0 multiplier (no penalty)
        0.5 risk = 0.5 multiplier (50% penalty)
        1.0 risk = 0.1 multiplier (90% penalty)
        """
        return 1.0 - (risk_score * 0.9)
    
    def _has_overlapping_employment(self, parsed: ParsedProfile) -> bool:
        """Check for overlapping employment dates"""
        if not parsed.career_history or len(parsed.career_history) < 2:
            return False
        
        for i in range(len(parsed.career_history) - 1):
            role1_end = parsed.career_history[i].get('end_date')
            role2_start = parsed.career_history[i + 1].get('start_date')
            
            if role1_end and role2_start:
                if role1_end > role2_start:
                    return True
        
        return False
    
    def _has_unrealistic_experience(self, parsed: ParsedProfile) -> bool:
        """Check for unrealistic experience claims"""
        if not parsed.total_years_experience:
            return False
        
        # Career history should roughly match total years
        if len(parsed.career_history) > parsed.total_years_experience * 2:
            return True  # Too many roles for claimed years
        
        return False
    
    def _has_suspicious_skills(self, candidate: Dict[str, Any]) -> bool:
        """Check for suspicious skill distribution"""
        skills = candidate.get('skills', [])
        
        if not skills:
            return True  # No skills is suspicious
        
        # All skills are "expert" level (common in honeypots)
        expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
        if len(skills) > 5 and expert_count / len(skills) > 0.8:
            return True
        
        # Too many unrelated skills
        if len(skills) > 50:
            return True
        
        return False
    
    def _has_career_inconsistency(self, parsed: ParsedProfile) -> bool:
        """Check for career inconsistencies"""
        if not parsed.career_history:
            return False
        
        # Huge title changes
        titles = [r.get('title', '').lower() for r in parsed.career_history]
        for i in range(len(titles) - 1):
            if titles[i] and titles[i + 1]:
                # Very different roles might indicate inconsistency
                if len(set(titles[i].split()) & set(titles[i + 1].split())) == 0:
                    if i > 0:  # Allow one switch early in career
                        return True
        
        return False
    
    def _is_generic_profile(self, parsed: ParsedProfile, candidate: Dict[str, Any]) -> bool:
        """Check for generic/low-quality profile"""
        summary = parsed.summary or ""
        headline = parsed.headline or ""
        
        if len(summary) < 20 and len(headline) < 10:
            return True
        
        if not candidate.get('skills'):
            return True
        
        return False
    
    def _has_excessive_expert_skills(self, candidate: Dict[str, Any]) -> bool:
        """Check for excessive expert-level skills (common in honeypots)"""
        skills = candidate.get('skills', [])
        if not skills:
            return False

        expert_count = sum(1 for s in skills if s.get('proficiency') in ['expert', 'mastery'])
        return len(skills) > 5 and expert_count / len(skills) > 0.8

    def _has_mastery_without_duration(self, candidate: Dict[str, Any]) -> bool:
        """Check for expert/mastery skills declared with zero duration."""
        zero_duration_mastery = 0
        for skill in candidate.get('skills', []):
            proficiency = skill.get('proficiency', '').lower()
            duration = skill.get('duration_months', 0)
            if proficiency in ['expert', 'mastery'] and duration == 0:
                zero_duration_mastery += 1
        return zero_duration_mastery >= 2
