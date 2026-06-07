"""
Feature Scoring Module
Implements the 5-component scoring system from the ranking strategy.
"""

import re
from typing import Dict, Any, Tuple, List
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from .candidate_profile_parser import ParsedProfile, CandidateProfileParser


@dataclass
class ScoringComponents:
    title_relevance: float          # NEW: decisive anti-stuffer signal
    skill_trust_score: float        # NEW: endorsement+duration weighted
    assessment_score: float         # NEW: platform-verified
    technical_relevance: float      # existing keyword/scale score
    production_experience: float    # existing
    profile_quality_multiplier: float
    experience_level_fit: float
    education_score: float          # NEW
    evaluation_framework_score: float
    product_mindset_score: float
    semantic_similarity: float
    behavioral_multiplier: float    # CHANGED: now a multiplier not additive

    @property
    def final_score(self) -> float:
        """
        Base score built from structured signals, then multiplied by
        behavioral multiplier and profile quality.

        Weights designed to make title + trust skills + assessments
        decisive so that keyword stuffers cannot rank highly.
        """
        base = (
            self.title_relevance        * 0.30 +
            self.skill_trust_score      * 0.25 +
            self.assessment_score       * 0.20 +
            self.experience_level_fit   * 0.08 +
            self.education_score        * 0.05 +
            self.technical_relevance    * 0.05 +
            self.production_experience  * 0.04 +
            self.evaluation_framework_score * 0.02 +
            self.semantic_similarity    * 0.01
        )

        final = base * self.profile_quality_multiplier * self.behavioral_multiplier
        return min(max(final, 0.0), 1.0)


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
        advanced_scorer=None,
        jd_skill_keywords: List[str] = None
    ) -> ScoringComponents:

        if reference_date is None:
            reference_date = datetime.now()

        if jd_skill_keywords is None:
            jd_skill_keywords = []

        title_relevance = self._score_title_relevance(parsed_profile)
        skill_trust = self._score_skill_trust(
            candidate_raw, parsed_profile, jd_skill_keywords
        )
        assessment_score = self._score_skill_assessments(
            candidate_raw, jd_skill_keywords
        )
        technical_relevance = self._score_technical_relevance(
            candidate_raw, parsed_profile
        )
        production_experience = self._score_production_experience(
            parsed_profile, candidate_raw
        )
        profile_quality_multiplier = self._score_profile_quality_multiplier(
            parsed_profile, candidate_raw
        )
        experience_level_fit = self._score_experience_level_fit(parsed_profile)
        education_score = self._score_education(parsed_profile)
        behavioral_multiplier = self._compute_behavioral_multiplier(
            candidate_raw, parsed_profile, reference_date
        )

        evaluation_framework_score = 0.0
        product_mindset_score = 0.0
        if advanced_scorer:
            evaluation_framework_score = advanced_scorer.score_evaluation_framework(
                candidate_raw
            )
            product_mindset_score = advanced_scorer.score_product_mindset(
                candidate_raw, parsed_profile
            )

        return ScoringComponents(
            title_relevance=title_relevance,
            skill_trust_score=skill_trust,
            assessment_score=assessment_score,
            technical_relevance=technical_relevance,
            production_experience=production_experience,
            profile_quality_multiplier=profile_quality_multiplier,
            experience_level_fit=experience_level_fit,
            education_score=education_score,
            evaluation_framework_score=evaluation_framework_score,
            product_mindset_score=product_mindset_score,
            semantic_similarity=semantic_similarity,
            behavioral_multiplier=behavioral_multiplier
        )
    
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

    def _score_title_relevance(
        self,
        parsed_profile: ParsedProfile
    ) -> float:
        """
        Score how relevant the candidate's current title is to the JD role.
        This is the decisive anti-keyword-stuffer signal.
        An HR Manager scores 0.0 regardless of listed skills.
        An ML Engineer scores 1.0.
        Check both current_title and most_recent_role_title.
        """
        TITLE_TIERS = {
            "tier_1": [
                "ml engineer", "machine learning engineer", "ai engineer",
                "ranking engineer", "search engineer", "recommendation engineer",
                "nlp engineer", "research scientist", "applied scientist",
                "data scientist", "computer vision engineer", "deep learning engineer",
                "retrieval engineer", "recsys engineer", "principal engineer",
                "staff engineer", "senior engineer"
            ],
            "tier_2": [
                "software engineer", "data engineer", "backend engineer",
                "platform engineer", "full stack engineer", "infrastructure engineer",
                "analytics engineer", "ml ops engineer", "mlops", "devops engineer"
            ],
            "tier_3": [
                "data analyst", "business analyst", "product manager",
                "technical lead", "engineering manager", "solutions architect"
            ]
        }

        TITLE_TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.5, "tier_3": 0.2, "tier_4": 0.0}

        titles_to_check = [
            parsed_profile.current_title.lower(),
            parsed_profile.most_recent_role_title.lower()
        ]

        best_score = 0.0
        for title in titles_to_check:
            for tier, keywords in TITLE_TIERS.items():
                if any(kw in title for kw in keywords):
                    tier_score = TITLE_TIER_SCORES[tier]
                    if tier_score > best_score:
                        best_score = tier_score
                    break

        return best_score

    def _score_skill_trust(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        jd_skill_keywords: List[str]
    ) -> float:
        """
        Score skills weighted by proficiency, duration_months, and endorsements.
        Raw keyword presence without backing (endorsements=0, duration=0)
        contributes near-zero. This defeats keyword stuffing.

        jd_skill_keywords: list of lowercase skill terms from the JD
        """
        if not candidate_raw.get("skills"):
            return 0.0

        PROFICIENCY_MAP = {
            "beginner": 0.25, "intermediate": 0.5,
            "advanced": 0.75, "expert": 1.0
        }

        total_trust = 0.0
        matched_count = 0

        for skill in candidate_raw["skills"]:
            name = skill.get("name", "").lower()
            is_relevant = any(jd_kw in name or name in jd_kw
                              for jd_kw in jd_skill_keywords)
            if not is_relevant:
                continue

            prof = PROFICIENCY_MAP.get(skill.get("proficiency", "beginner"), 0.25)
            dur = min(skill.get("duration_months", 0) / 36.0, 1.0)
            end = min(skill.get("endorsements", 0) / 20.0, 1.0)

            trust = prof * 0.4 + dur * 0.4 + end * 0.2
            total_trust += trust
            matched_count += 1

        if matched_count == 0:
            return 0.0

        return min(total_trust / 5.0, 1.0)

    def _score_skill_assessments(
        self,
        candidate_raw: Dict[str, Any],
        jd_skill_keywords: List[str]
    ) -> float:
        """
        Score platform-verified skill assessment results.
        This is the hardest signal to fake — an HR Manager will not have
        a Python or FAISS assessment score.
        Returns 0.0 if no relevant assessments exist.
        """
        assessments = candidate_raw.get("redrob_signals", {}).get(
            "skill_assessment_scores", {}
        )
        if not assessments:
            return 0.0

        relevant_scores = []
        for skill_name, score in assessments.items():
            skill_lower = skill_name.lower()
            if any(jd_kw in skill_lower or skill_lower in jd_kw
                   for jd_kw in jd_skill_keywords):
                relevant_scores.append(score / 100.0)

        if not relevant_scores:
            return 0.0

        return min(sum(relevant_scores) / len(relevant_scores), 1.0)

    def _score_education(self, parsed_profile: ParsedProfile) -> float:
        """Score education institution tier."""
        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.7,
            "tier_3": 0.4,
            "tier_4": 0.2,
            "unknown": 0.3
        }
        return tier_scores.get(parsed_profile.education_tier, 0.3)

    def _compute_behavioral_multiplier(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        reference_date: datetime
    ) -> float:
        """
        Compute behavioral multiplier (0.4 to 1.3 range).
        Applied multiplicatively to the base score, not additively.

        Key signals:
        - open_to_work_flag: hard availability gate
        - interview_completion_rate: ghost risk
        - offer_acceptance_rate: pipeline waste risk
        - recruiter_response_rate: engagement signal
        - notice_period_days: time-to-start
        - last_active_date: recency of platform activity
        """
        redrob = candidate_raw.get("redrob_signals", {})
        multiplier = 1.0

        if not redrob.get("open_to_work_flag", True):
            multiplier *= 0.5

        completion_rate = parsed_profile.interview_completion_rate
        if completion_rate < 0.4:
            multiplier *= 0.6
        elif completion_rate < 0.7:
            multiplier *= 0.85
        elif completion_rate >= 0.9:
            multiplier *= 1.05

        acceptance = parsed_profile.offer_acceptance_rate
        if acceptance == -1:
            pass
        elif acceptance < 0.3:
            multiplier *= 0.75
        elif acceptance >= 0.7:
            multiplier *= 1.05

        response_rate = redrob.get("recruiter_response_rate", 0.5)
        if response_rate >= 0.65:
            multiplier *= 1.08
        elif response_rate < 0.15:
            multiplier *= 0.85

        notice = redrob.get("notice_period_days", 30)
        if notice <= 15:
            multiplier *= 1.05
        elif notice <= 30:
            multiplier *= 1.02
        elif notice > 90:
            multiplier *= 0.88

        last_active_str = redrob.get("last_active_date", "")
        if last_active_str:
            try:
                last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
                days_inactive = (reference_date - last_active).days
                if days_inactive > 180:
                    multiplier *= 0.80
                elif days_inactive > 90:
                    multiplier *= 0.90
                elif days_inactive <= 30:
                    multiplier *= 1.05
            except ValueError:
                pass

        github = redrob.get("github_activity_score", -1)
        if github >= 70:
            multiplier *= 1.08
        elif github >= 40:
            multiplier *= 1.04

        saved = redrob.get("saved_by_recruiters_30d", 0)
        if saved >= 10:
            multiplier *= 1.05
        elif saved >= 5:
            multiplier *= 1.02

        return min(max(multiplier, 0.4), 1.3)

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
        
        # 2A: Career depth and production engineering experience
        ml_months = parsed_profile.career_depth_months.get('ml_months', 0)
        production_months = parsed_profile.career_depth_months.get('production_months', 0)
        engineering_months = parsed_profile.career_depth_months.get('engineering_months', 0)
        
        # Score experience with ML, production, and engineering together
        deep_experience_score = min(
            0.4 +
            min(ml_months / 24, 1.0) * 0.25 +
            min(production_months / 24, 1.0) * 0.20 +
            min(engineering_months / 36, 1.0) * 0.15,
            1.0
        )
        
        # Bonus for sustained scale exposure
        if parsed_profile.most_recent_company_size in ['1001-5000', '5001-10000', '10001+']:
            deep_experience_score += 0.05
        
        # 2B: Role consistency / practical delivery signal
        current_title = parsed_profile.current_title.lower()
        is_tech_focused = any(
            term in current_title
            for term in CandidateProfileParser.ENGINEERING_TITLES
        )
        
        consistency_penalty = 0.0
        titles = [role['title'].lower() for role in candidate_raw['career_history']]
        seniority_jumps = sum(
            1 for title in titles
            if any(level in title for level in ['principal', 'staff', 'senior', 'manager'])
        )
        if titles and seniority_jumps > len(titles) * 0.6:
            consistency_penalty += 0.1
        
        if not is_tech_focused and parsed_profile.years_experience >= 5:
            consistency_penalty += 0.15
        
        # Reduce score if the profile is mostly consulting without production signals
        career_text = ' '.join([role['description'].lower() for role in candidate_raw['career_history']])
        if parsed_profile.is_consulting_only and 'production' not in career_text:
            consistency_penalty += 0.15
        
        production_experience = max(0.0, deep_experience_score - consistency_penalty)
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
            multiplier -= min(len(parsed_profile.timeline_issues) * 0.12, 0.3)
        
        # Skill realism - MAJOR RED FLAG
        if parsed_profile.skill_counts > 45:
            multiplier -= 0.25  # Strong penalty for likely padding
        elif parsed_profile.skill_counts > 35:
            multiplier -= 0.10  # Suspiciously high skill count
        
        # Check for unrealistic skill combinations - MAJOR RED FLAG
        skill_names = [s.lower() for s in parsed_profile.skill_names]
        if self._check_unrealistic_skills(skill_names):
            multiplier -= 0.30  # Strong penalty for suspicious profiles
        
        # Profile completeness - BONUS/PENALTY
        if parsed_profile.profile_completeness < 40:
            multiplier -= 0.25  # Very incomplete = suspicious
        elif parsed_profile.profile_completeness < 60:
            multiplier -= 0.12  # Incomplete
        elif parsed_profile.profile_completeness >= 80:
            multiplier += 0.05  # Bonus for very complete
        
        # Verification signals - BONUS (trust signals)
        redrob = candidate_raw['redrob_signals']
        verify_count = sum([
            redrob.get('verified_email', False),
            redrob.get('verified_phone', False),
            redrob.get('linkedin_connected', False)
        ])
        multiplier += min(verify_count, 3) * 0.08
        
        # Reward strong GitHub/recruiter signals for trust
        if redrob.get('github_activity_score', 0) >= 50:
            multiplier += 0.05
        if redrob.get('recruiter_response_rate', 0) >= 0.5:
            multiplier += 0.04
        
        # Ensure multiplier is in valid range [0.2, 1.0]
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
        
        career_titles = [role['title'].lower() for role in candidate_raw['career_history']]
        research_title_count = sum(1 for t in career_titles if 'research' in t)
        is_research_heavy = research_title_count >= max(1, len(career_titles) * 0.5)
        if is_research_heavy:
            if not any(keyword in career_text for keyword in CandidateProfileParser.PRODUCTION_KEYWORDS):
                multiplier *= 0.3
        
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
