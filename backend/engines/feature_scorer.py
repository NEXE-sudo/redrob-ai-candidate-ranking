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
from .recruiter_jd_parser import RequirementProfile


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
        # Use explicit weights but normalize them so they sum to 1.0.
        # This prevents accidental drift when individual weights are adjusted.
        weights = {
            'title_relevance': 0.25,
            'skill_trust_score': 0.22,
            'assessment_score': 0.18,
            'technical_relevance': 0.12,
            'production_experience': 0.08,
            'experience_level_fit': 0.06,
            'education_score': 0.03,
            'evaluation_framework_score': 0.03,
            'product_mindset_score': 0.03,
            'semantic_similarity': 0.10
        }

        base_raw = (
            self.title_relevance        * weights['title_relevance'] +
            self.skill_trust_score      * weights['skill_trust_score'] +
            self.assessment_score       * weights['assessment_score'] +
            self.technical_relevance    * weights['technical_relevance'] +
            self.production_experience  * weights['production_experience'] +
            self.experience_level_fit   * weights['experience_level_fit'] +
            self.education_score        * weights['education_score'] +
            self.evaluation_framework_score * weights['evaluation_framework_score'] +
            self.product_mindset_score   * weights['product_mindset_score'] +
            self.semantic_similarity    * weights['semantic_similarity']
        )

        sum_weights = sum(weights.values()) or 1.0
        base = base_raw / sum_weights

        # Phase 2: Fix Score Distortion (bounded additive instead of multiplicative)
        # Shift multipliers to act as +/- bonuses relative to 1.0 baseline
        quality_adjustment = (self.profile_quality_multiplier - 1.0) * 0.10
        behavioral_adjustment = (self.behavioral_multiplier - 1.0) * 0.15
        
        final = base + quality_adjustment + behavioral_adjustment
        return min(max(final, 0.0), 1.0)

    @property
    def behavioral_engagement(self) -> float:
        """Alias for backward compatibility with tests and reporting."""
        return self.behavioral_multiplier

    @property
    def profile_quality(self) -> float:
        """Alias for backward compatibility with tests and reporting."""
        return self.profile_quality_multiplier


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
        jd_skill_keywords: List[str] = None,
        requirement_profile: RequirementProfile = None
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
            candidate_raw, parsed_profile, requirement_profile
        )
        production_experience = self._score_production_experience(
            parsed_profile, candidate_raw, requirement_profile
        )
        profile_quality_multiplier = self._score_profile_quality_multiplier(
            parsed_profile, candidate_raw, requirement_profile
        )
        experience_level_fit = self._score_experience_level_fit(parsed_profile, requirement_profile)
        education_score = self._score_education(parsed_profile)
        behavioral_multiplier = self._compute_behavioral_multiplier(
            candidate_raw, parsed_profile, reference_date, requirement_profile
        )

        evaluation_framework_score = 0.0
        product_mindset_score = 0.0
        if advanced_scorer:
            evaluation_framework_score = advanced_scorer.score_evaluation_framework(
                candidate_raw
            )
            product_mindset_score = advanced_scorer.score_startup_product_mindset(
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
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile = None
    ) -> float:
        """Score 1A + 1B + 1C: Keywords, Scale, JD fit, Recency"""
        
        # 1A: Keyword score
        keyword_score = self._keyword_match_score(candidate_raw, parsed_profile)
        
        # 1B: Scale signal
        scale_score = self._scale_signal_score(candidate_raw)
        
        if requirement_profile:
            jd_requirement_score = self._score_jd_requirement_match(
                candidate_raw, parsed_profile, requirement_profile
            )
            hands_on_score = self._score_hands_on_coding(
                candidate_raw, parsed_profile, requirement_profile
            )
            leadership_score = self._score_leadership_fit(
                candidate_raw, parsed_profile, requirement_profile
            )
            technical_base = (
                keyword_score * 0.35 +
                scale_score * 0.15 +
                jd_requirement_score * 0.30 +
                hands_on_score * 0.12 +
                leadership_score * 0.08
            )
        else:
            technical_base = keyword_score * 0.5 + scale_score * 0.25
        
        # 1C: Apply recency penalty/bonus
        recency_penalty = self._calculate_recency_penalty(parsed_profile)
        technical_relevance = technical_base * (1.0 - recency_penalty)
        
        return min(max(technical_relevance, 0.0), 1.0)

    def _collect_candidate_text(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> str:
        """Collect all candidate text fields for JD matching."""
        profile_text = (
            candidate_raw.get('profile', {}).get('summary', '').lower() + ' ' +
            candidate_raw.get('profile', {}).get('headline', '').lower() + ' ' +
            candidate_raw.get('profile', {}).get('current_title', '').lower() + ' ' +
            candidate_raw.get('profile', {}).get('location', '').lower()
        )
        career_text = ' '.join([
            role.get('title', '').lower() + ' ' + role.get('description', '').lower()
            for role in candidate_raw.get('career_history', [])
        ])
        skill_text = ' '.join([s.get('name', '').lower() for s in candidate_raw.get('skills', [])])
        cert_text = ' '.join([c.get('name', '').lower() for c in candidate_raw.get('certifications', [])])
        language_text = ' '.join([l.get('language', '').lower() for l in candidate_raw.get('languages', [])])
        return ' '.join([profile_text, career_text, skill_text, cert_text, language_text]).strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent keyword matching."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _score_jd_requirement_match(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile
    ) -> float:
        """Score JD requirement fit using soft keyword matching and semantic similarity.
        
        ROBUSTNESS: Use partial/fuzzy matching instead of hard hit counts to
        reduce brittleness and overfitting to exact keyword lists.
        """
        if not requirement_profile:
            return 0.0

        candidate_text = self._normalize_text(self._collect_candidate_text(candidate_raw, parsed_profile))
        required_keywords = requirement_profile.required_keywords
        preferred_keywords = requirement_profile.preferred_keywords

        # Soft matching: reward substring/partial matches
        def soft_match_score(keywords, text):
            """Return a soft match ratio for keyword lists."""
            if not keywords:
                return 0.0

            hits = 0.0
            text_words = set(text.split())
            for kw in keywords:
                kw_norm = self._normalize_text(kw)
                kw_words = kw_norm.split()
                if kw_norm in text:
                    hits += 1.0
                    continue

                if any(word in text for word in kw_words if len(word) >= 3):
                    hits += 0.7
                    continue

                intersection = len(set(kw_words) & text_words)
                if kw_words and intersection / len(kw_words) >= 0.6:
                    hits += 0.5
                    continue

                # Reward partial semantic signal from individual term overlap
                if any(word in text for word in kw_words):
                    hits += 0.35

            return min(hits / len(keywords), 1.0)

        required_score = soft_match_score(required_keywords, candidate_text)
        preferred_score = soft_match_score(preferred_keywords, candidate_text)

        negative_penalty = self._score_negative_jd_signals(
            candidate_raw, parsed_profile, requirement_profile
        )

        # Balance required vs. preferred; soften thresholds
        score = required_score * 0.60 + preferred_score * 0.30
        score = max(score - negative_penalty, 0.0)

        # Softer fallback: reward if candidate shows strong domain signals
        if score < 0.35:
            domain_keywords = ['embeddings', 'retrieval', 'ranking', 'semantic', 'production', 'ml', 'ai']
            domain_hits = sum(1 for kw in domain_keywords if kw in candidate_text)
            score = max(score, min(0.35 + domain_hits * 0.03, 0.5))

        # If no required terms are present, use domain signal and JD-awareness
        if required_score < 0.2 and preferred_score >= 0.4:
            score = max(score, 0.28)

        return min(score, 1.0)

    def _score_negative_jd_signals(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile
    ) -> float:
        """Penalty for JD negative signals that match the candidate."""
        penalty = 0.0
        negatives = requirement_profile.negative_signals
        redrob = candidate_raw.get('redrob_signals', {})

        if not negatives:
            return 0.0

        if 'consulting-only' in negatives or 'consulting career' in negatives:
            if parsed_profile.is_consulting_only:
                penalty += 0.25

        if 'pure research' in negatives or 'academic research' in negatives:
            if self._has_research_without_production(candidate_raw):
                penalty += 0.20

        if 'no production' in negatives or 'no shipped' in negatives:
            if not any(keyword in self._collect_candidate_text(candidate_raw, parsed_profile)
                       for keyword in self.PRODUCTION_KEYWORDS):
                penalty += 0.20

        if 'long notice period' in negatives:
            if redrob.get('notice_period_days', 0) > 60:
                penalty += 0.15

        return min(penalty, 0.5)

    def _has_research_without_production(self, candidate_raw: Dict[str, Any]) -> bool:
        career_text = ' '.join([
            role.get('title', '').lower() + ' ' + role.get('description', '').lower()
            for role in candidate_raw.get('career_history', [])
        ])
        if 'research' in career_text and not any(keyword in career_text for keyword in self.PRODUCTION_KEYWORDS):
            return True
        return False

    def _score_hands_on_coding(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile
    ) -> float:
        """Score a candidate's hands-on coding signal when the JD requires it."""
        if not requirement_profile or not requirement_profile.hands_on_coding:
            return 0.0

        text = self._collect_candidate_text(candidate_raw, parsed_profile)
        hands_on_terms = [
            'hands-on', 'hands on', 'coding', 'implementation',
            'write code', 'development', 'engineer', 'build',
            'ship', 'productization', 'technical implementation'
        ]
        if any(term in text for term in hands_on_terms):
            return 1.0
        return 0.45

    def _score_leadership_fit(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile
    ) -> float:
        """Score leadership signals when the JD explicitly requires them."""
        if not requirement_profile or not requirement_profile.leadership_required:
            return 0.0

        title_text = (
            parsed_profile.current_title + ' ' + parsed_profile.most_recent_role_title
        ).lower()
        leadership_terms = [
            'lead', 'manage', 'mentor', 'leadership', 'team',
            'principal', 'staff', 'head', 'director'
        ]
        if any(term in title_text for term in leadership_terms):
            return 1.0

        text = self._collect_candidate_text(candidate_raw, parsed_profile)
        if any(term in text for term in leadership_terms):
            return 0.85
        return 0.60

    def _normalize_title(self, title: str) -> str:
        """Normalize titles for robust matching."""
        title = (title or '').lower()
        title = re.sub(r"[^a-z0-9\s]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    def add_rank_aggregation_method(self) -> str:
        """Return method hint for rank aggregation as alternative to score-based ranking.
        
        ROBUSTNESS: Provide option to use rank aggregation (Borda count, Spearman) 
        instead of single weighted sum. Reduces impact of single dimension.
        """
        return "rank_aggregation_available"

    def _score_title_relevance(
        self,
        parsed_profile: ParsedProfile
    ) -> float:
        """
        Score how relevant the candidate's current title is to the JD role.
        This is the decisive anti-keyword-stuffer signal.
        """
        titles_to_check = [
            self._normalize_title(parsed_profile.current_title),
            self._normalize_title(parsed_profile.most_recent_role_title)
        ]

        title_patterns = [
            (1.0, [r'\b(ml|machine learning|ai|artificial intelligence)\b.*\b(engineer|scientist|researcher|specialist)\b']),
            (1.0, [r'\b(ranking|retrieval|search|recommendation|recsys)\b.*\b(engineer|scientist|developer)\b']),
            (1.0, [r'\b(data scientist|applied scientist|research scientist|ml engineer|ai engineer|nlp engineer|deep learning engineer|computer vision engineer)\b']),
            (0.85, [r'\b(staff|principal|lead)\b.*\b(engineer|scientist|developer)\b']),
            (0.75, [r'\b(software engineer|backend engineer|platform engineer|full stack engineer|data engineer|mlops engineer|devops engineer)\b']),
            (0.65, [r'\b(technical lead|tech lead|engineering manager|manager|director)\b']),
            (0.35, [r'\b(data analyst|business analyst|product manager|program manager|project manager)\b']),
            (0.0, [r'\b(hr manager|recruiter|talent acquisition|people operations)\b']),
        ]

        best_score = 0.0
        for title in titles_to_check:
            if not title:
                continue
            for score, patterns in title_patterns:
                if any(re.search(pattern, title) for pattern in patterns):
                    best_score = max(best_score, score)

        if best_score == 0.0 and any(
            keyword in title
            for title in titles_to_check
            for keyword in ['engineer', 'scientist', 'developer', 'architect', 'data']
        ):
            return 0.5

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

        normalized_jd_keywords = [self._normalize_text(jd_kw) for jd_kw in jd_skill_keywords]
        for skill in candidate_raw["skills"]:
            name = self._normalize_text(skill.get("name", ""))
            is_relevant = any(jd_kw in name or name in jd_kw
                              for jd_kw in normalized_jd_keywords)
            if not is_relevant:
                continue

            prof = PROFICIENCY_MAP.get(skill.get("proficiency", "beginner"), 0.25)
            dur = min(skill.get("duration_months", 0) / 36.0, 1.0)
            end = min(skill.get("endorsements", 0) / 20.0, 1.0)

            trust = prof * 0.4 + dur * 0.4 + end * 0.2
            total_trust += trust
            matched_count += 1

        if matched_count == 0:
            # Fallback to overall trust when JD-specific matches are missing.
            all_skills_trust = []
            for skill in candidate_raw["skills"]:
                prof = PROFICIENCY_MAP.get(skill.get("proficiency", "beginner"), 0.25)
                dur = min(skill.get("duration_months", 0) / 36.0, 1.0)
                end = min(skill.get("endorsements", 0) / 20.0, 1.0)
                all_skills_trust.append(prof * 0.4 + dur * 0.4 + end * 0.2)
            if not all_skills_trust:
                return 0.0
            mean_trust = sum(all_skills_trust) / len(all_skills_trust)
            fallback_factor = 0.7 if not normalized_jd_keywords else 0.55
            return min(mean_trust * fallback_factor, 1.0)

        mean_trust = total_trust / matched_count
        coverage_bonus = min(matched_count / 6.0, 1.0) * 0.15
        return min(mean_trust * 0.8 + coverage_bonus, 1.0)

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

        normalized_jd_keywords = [self._normalize_text(jd_kw) for jd_kw in jd_skill_keywords]
        relevant_scores = []
        for skill_name, score in assessments.items():
            skill_lower = self._normalize_text(skill_name)
            if any(jd_kw in skill_lower or skill_lower in jd_kw
                   for jd_kw in normalized_jd_keywords):
                relevant_scores.append(score / 100.0)

        if not relevant_scores:
            all_scores = [v / 100.0 for v in assessments.values()]
            if not all_scores:
                return 0.0
            return min(sum(all_scores) / len(all_scores) * 0.6, 1.0)

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
        reference_date: datetime,
        requirement_profile: RequirementProfile = None
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

        views = redrob.get("profile_views_received_30d", 0)
        if views >= 20:
            multiplier *= 1.05
        elif views >= 10:
            multiplier *= 1.02

        search_appearances = redrob.get("search_appearance_30d", 0)
        if search_appearances >= 20:
            multiplier *= 1.05
        elif search_appearances >= 10:
            multiplier *= 1.02

        applications = redrob.get("applications_submitted_30d", 0)
        if applications >= 3:
            multiplier *= 1.04
        elif applications >= 1:
            multiplier *= 1.02

        avg_response = redrob.get("avg_response_time_hours", 24)
        if avg_response <= 24:
            multiplier *= 1.05
        elif avg_response <= 48:
            multiplier *= 1.02
        elif avg_response > 96:
            multiplier *= 0.95

        candidate_location = candidate_raw.get("profile", {}).get("location", "").lower()
        if requirement_profile and requirement_profile.relocation_required:
            if redrob.get('willing_to_relocate', False):
                multiplier *= 1.03

        if redrob.get("recruiter_response_rate", 0) >= 0.5:
            multiplier *= 1.02
            if requirement_profile and requirement_profile.location_preferences:
                if any(loc in candidate_location for loc in requirement_profile.location_preferences):
                    multiplier *= 1.03

        return min(max(multiplier, 0.4), 1.3)

    def _keyword_match_score(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> float:
        """Score keywords using soft n-gram matching instead of exact hits.
        
        ROBUSTNESS: Replace brittle exact keyword matching with softer scoring
        that rewards partial matches and nearby words (n-grams).
        """
        
        # Concatenate all relevant text
        profile_text = (
            candidate_raw.get('profile', {}).get('summary', '').lower() + ' ' +
            candidate_raw.get('profile', {}).get('headline', '').lower() + ' ' +
            candidate_raw.get('profile', {}).get('current_title', '').lower()
        )
        
        career_text = ' '.join([
            role['title'].lower() + ' ' + role['description'].lower()
            for role in candidate_raw['career_history']
        ])
        
        all_text = profile_text + ' ' + career_text
        skill_text = ' '.join([s['name'].lower() for s in candidate_raw['skills']])
        all_text = all_text + ' ' + skill_text
        
        # Soft counting: penalize keyword position but reward partial matches
        def soft_keyword_count(keywords, text, boost=1.0):
            """Count with soft matching: reward substring matches, not just exact."""
            count = 0.0
            for kw in keywords:
                # Exact match: full credit
                if kw in text:
                    count += 1.0 * boost
                # Substring match (e.g. "embedding" in "embeddings"): 0.6 credit
                else:
                    # Check if keyword is substring of nearby text
                    kw_short = kw.replace(' ', '')
                    if any(kw_short in word for word in text.split() if len(word) >= len(kw_short)):
                        count += 0.6 * boost
            return count
        
        tier1_count = soft_keyword_count(self.TIER_1_KEYWORDS, all_text, boost=1.0)
        tier2_count = soft_keyword_count(self.TIER_2_KEYWORDS, all_text, boost=0.8)
        tier3_count = soft_keyword_count(self.TIER_3_KEYWORDS, all_text, boost=0.6)
        
        # Softer expectation thresholds to reduce overfitting
        tier1_expected = 3.5
        tier2_expected = 4.0
        tier3_expected = 5.0
        
        # Use sigmoid-like saturation instead of hard cap
        tier1_sat = min(tier1_count / tier1_expected, 1.0) if tier1_expected > 0 else 0.0
        tier2_sat = min(tier2_count / tier2_expected, 1.0) if tier2_expected > 0 else 0.0
        tier3_sat = min(tier3_count / tier3_expected, 1.0) if tier3_expected > 0 else 0.0
        
        keyword_score = (
            0.40 * tier1_sat +
            0.35 * tier2_sat +
            0.25 * tier3_sat
        )
        
        # Phase 4: Keyword Density Penalty (Anti-stuffing)
        # If the candidate has many keyword hits but very few total words, penalize them.
        total_words = len(all_text.split())
        total_hits = tier1_count + tier2_count + tier3_count
        
        if total_words > 0:
            density = total_hits / total_words
            # A normal profile rarely exceeds 5-8% density for these specific ML/ranking keywords.
            if density > 0.15:
                # Extreme keyword stuffing -> heavy penalty
                keyword_score *= 0.5
            elif density > 0.08:
                # High density -> mild penalty
                keyword_score *= 0.8
                
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
            company_size = most_recent_role.get('company_size', '')
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
        candidate_raw: Dict[str, Any],
        requirement_profile: RequirementProfile = None
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
        elif parsed_profile.current_company_size in ['1001-5000', '5001-10000', '10001+']:
            deep_experience_score += 0.04
        elif parsed_profile.current_company_size == '501-1000':
            deep_experience_score += 0.02
        elif parsed_profile.current_company_size == '201-500':
            deep_experience_score += 0.01

        if parsed_profile.current_industry and any(
            keyword in parsed_profile.current_industry.lower()
            for keyword in ['technology', 'software', 'ai', 'ml', 'internet', 'ecommerce', 'search', 'saas', 'product']
        ):
            deep_experience_score += 0.03
        
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
        candidate_raw: Dict[str, Any],
        requirement_profile: RequirementProfile = None
    ) -> float:
        """Score profile consistency, realism as MULTIPLIER (0.7 to 1.0)
        
        ROBUSTNESS: Changed range to [0.7, 1.0] to prevent profile quality
        from dominating scoring. Even suspicious profiles can still rank if
        their technical signals are strong.
        
        Returns:
            Multiplier 0.7-1.0 (softer penalties than before)
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
        redrob = candidate_raw.get('redrob_signals', {})
        verify_count = sum([
            redrob.get('verified_email', False),
            redrob.get('verified_phone', False),
            redrob.get('linkedin_connected', False)
        ])
        multiplier += min(verify_count, 3) * 0.05

        # Reward strong GitHub/recruiter signals for trust
        if redrob.get('github_activity_score', 0) >= 50:
            multiplier += 0.03
        if redrob.get('recruiter_response_rate', 0) >= 0.5:
            multiplier += 0.02
            if requirement_profile and requirement_profile.location_preferences:
                if parsed_profile.country and any(
                    loc in parsed_profile.country.lower()
                    for loc in requirement_profile.location_preferences
                ):
                    multiplier += 0.03

        # Certifications and language signals
        if parsed_profile.certifications_count > 0:
            multiplier += min(parsed_profile.certifications_count, 4) * 0.01
        if parsed_profile.languages_count >= 2:
            multiplier += 0.02
        if parsed_profile.has_english_proficiency:
            multiplier += 0.01

        # Reasonable expected salary is a marginal reliability signal
        if parsed_profile.expected_salary_range_inr_lpa > 0:
            if 5 <= parsed_profile.expected_salary_range_inr_lpa <= 60:
                multiplier += 0.01
            elif parsed_profile.expected_salary_range_inr_lpa > 80:
                multiplier -= 0.02
            elif parsed_profile.expected_salary_range_inr_lpa < 2:
                multiplier -= 0.01

        # Recent signup age can indicate signal maturity
        if parsed_profile.days_since_signup >= 180:
            multiplier += 0.01
        elif parsed_profile.days_since_signup >= 30:
            multiplier += 0.005

        # Connections and endorsements are soft trust signals
        if parsed_profile.connection_count >= 500:
            multiplier += 0.03
        elif parsed_profile.connection_count >= 100:
            multiplier += 0.015

        if parsed_profile.endorsements_received >= 50:
            multiplier += 0.03
        elif parsed_profile.endorsements_received >= 20:
            multiplier += 0.015

        # Match work mode when JD requires remote/hybrid
        if requirement_profile and requirement_profile.location_preferences:
            if any(term in requirement_profile.location_preferences for term in ['remote', 'hybrid']):
                if parsed_profile.preferred_work_mode in ['remote', 'hybrid']:
                    multiplier *= 1.03
                elif parsed_profile.preferred_work_mode and parsed_profile.preferred_work_mode not in ['remote', 'hybrid', 'onsite', 'on-site', 'office']:
                    multiplier *= 0.98

        # Ensure multiplier is in valid range [0.7, 1.0] per robustness design
        multiplier = min(max(multiplier, 0.7), 1.0)
        
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
    
    def _score_experience_level_fit(
        self,
        parsed_profile: ParsedProfile,
        requirement_profile: RequirementProfile = None
    ) -> float:
        """Score experience in a target band with JD context"""
        
        exp = parsed_profile.years_experience
        if requirement_profile:
            min_target = requirement_profile.target_experience_min
            max_target = requirement_profile.target_experience_max
        else:
            min_target = 5
            max_target = 9

        if exp < min_target:
            return max(0.3, 0.5 + 0.5 * (exp / max(min_target, 1)))
        elif exp <= max_target:
            return 1.0
        elif exp <= max_target + 3:
            return 0.95
        elif exp <= max_target + 6:
            return 0.85
        else:
            return 0.7
    
    def apply_disqualifying_factors(
        self,
        final_score: float,
        parsed_profile: ParsedProfile,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Apply hard disqualifiers that drastically reduce score"""
        
        multiplier = 1.0
        
        redrob = candidate_raw.get('redrob_signals', {})
        
        # Pure research background (no production deployment)
        career_text = ' '.join([
            role['description'].lower()
            for role in candidate_raw.get('career_history', [])
        ])
        
        career_titles = [role['title'].lower() for role in candidate_raw['career_history']]
        research_title_count = sum(1 for t in career_titles if 'research' in t)
        is_research_heavy = research_title_count >= max(1, len(career_titles) * 0.5)
        if is_research_heavy:
            if not any(keyword in career_text for keyword in CandidateProfileParser.PRODUCTION_KEYWORDS):
                multiplier *= 0.3
        
        # Pure consulting background (TCS/Infosys etc)
        if parsed_profile.is_consulting_only and parsed_profile.years_experience >= 5:
            multiplier *= 0.75

        # No recent code (18+ months ago)
        if parsed_profile.years_since_last_coding > 1.5:
            multiplier *= 0.7
        # Not available / not open to work
        if not redrob.get('open_to_work_flag', False) and redrob.get('recruiter_response_rate', 0) < 0.05:
            multiplier *= 0.3
        
        return final_score * multiplier
