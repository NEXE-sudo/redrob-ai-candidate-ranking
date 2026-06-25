"""
Candidate Profile Parser
Extracts structured information from raw candidate profiles for scoring.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ParsedProfile:
    """Structured extracted profile data"""
    candidate_id: str
    years_experience: float
    current_title: str
    current_company: str
    summary: str
    headline: str
    company_type: str  # "product", "consulting", "startup", "other"
    is_consulting_only: bool
    profile_completeness: float
    skill_names: List[str]
    skill_counts: int
    skill_endorsements_mean: float
    career_history: List[Dict[str, Any]]
    career_depth_months: Dict[str, int]
    timeline_issues: List[str]
    most_recent_role_title: str
    total_years_experience: float
    most_recent_role_company: str
    most_recent_role_ended_months_ago: float
    most_recent_company_size: str
    years_since_last_coding: float
    has_github: bool
    github_activity_score: float
    education_tier: str           # "tier_1", "tier_2", "tier_3", "tier_4", "unknown"
    has_skill_assessments: bool   # True if any skill_assessment_scores exist
    assessment_relevant_score: float  # 0.0-1.0, avg of JD-relevant assessments
    interview_completion_rate: float  # from redrob_signals
    offer_acceptance_rate: float      # from redrob_signals, -1 means no history
    avg_response_time_hours: float    # from redrob_signals
    current_industry: str
    current_company_size: str
    country: str
    certifications_count: int
    certification_names: List[str]
    languages_count: int
    language_names: List[str]
    has_english_proficiency: bool
    connection_count: int
    endorsements_received: int
    expected_salary_range_inr_lpa: float
    preferred_work_mode: str
    signup_date: str
    days_since_signup: float
    top_skill_trust_scores: dict      # {skill_name: trust_score} for top 10 skills


class CandidateProfileParser:
    """Parse and validate candidate profiles"""
    
    # Consulting company keywords
    CONSULTING_COMPANIES = {
        'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
        'deloitte', 'pwc', 'kpmg', 'ey', 'ernst & young', 'heidrick & struggles',
        'mckinsey', 'bain', 'bcg', 'goldman sachs', 'morgan stanley'
    }
    
    # Company size mapping
    COMPANY_SIZE_NUMERIC = {
        "1-10": 5,
        "11-50": 30,
        "51-200": 125,
        "201-500": 350,
        "501-1000": 750,
        "1001-5000": 3000,
        "5001-10000": 7500,
        "10001+": 50000
    }
    
    # Production-related keywords
    PRODUCTION_KEYWORDS = {
        'production', 'deployed', 'shipped', 'live', 'real-time', 'qps',
        'scale', 'million', 'billion', 'latency', 'throughput', 'production-grade'
    }
    
    # Engineering-related titles
    ENGINEERING_TITLES = {
        'engineer', 'ml engineer', 'ai engineer', 'data engineer', 'ml researcher',
        'ai researcher', 'scientist', 'technical lead', 'tech lead', 'architect',
        'data scientist', 'applied scientist', 'research scientist',
        'research engineer', 'applied researcher', 'machine learning', 'deep learning'
    }
    
    # Skills to look for
    RETRIEVAL_SKILLS = {
        'elasticsearch', 'milvus', 'pinecone', 'weaviate', 'qdrant', 'faiss',
        'opensearch', 'vector database', 'vector db', 'semantic search', 'rag',
        'retrieval'
    }
    
    EMBEDDING_SKILLS = {
        'embeddings', 'sentence transformers', 'bge', 'e5', 'openai embeddings',
        'embedding model', 'dense retrieval', 'semantic', 'transformer'
    }
    
    RANKING_SKILLS = {
        'ranking', 'learning-to-rank', 'ltr', 'xgboost', 'lambdarank',
        'ndcg', 'mrr', 'map', 'evaluation metric'
    }
    
    LLM_FINE_TUNING_SKILLS = {
        'lora', 'qlora', 'peft', 'fine-tuning llm', 'fine-tuning', 'prompt',
        'instruction tuning', 'rlhf'
    }
    
    def parse_candidate(self, candidate_raw: Dict[str, Any], reference_date: datetime = None) -> ParsedProfile:
        """Parse raw candidate JSON into structured format"""
        
        if reference_date is None:
            reference_date = datetime.now()
        
        profile = candidate_raw.get('profile', {})
        career_history = candidate_raw.get('career_history', [])
        skills = candidate_raw.get('skills', [])
        redrob_signals = candidate_raw.get('redrob_signals', {})
        
        # Basic info
        candidate_id = candidate_raw.get('candidate_id', '')
        years_experience = profile.get('years_of_experience', 0.0)
        current_title = profile.get('current_title', '')
        current_company = profile.get('current_company', '')
        current_company_size = profile.get('current_company_size', '')
        current_industry = profile.get('current_industry', '')
        country = profile.get('country', '')
        summary = profile.get('summary', '')
        headline = profile.get('headline', '')
        
        # Determine company type
        company_type, is_consulting_only = self._classify_company(career_history)
        
        # Profile completeness
        profile_completeness = redrob_signals.get('profile_completeness_score', 0.0)
        
        # Skill analysis
        skill_names = [s['name'] for s in skills]
        skill_counts = len(skills)
        endorsements = [s['endorsements'] for s in skills]
        skill_endorsements_mean = sum(endorsements) / len(endorsements) if endorsements else 0

        # Certifications and languages
        certifications = candidate_raw.get('certifications', [])
        certifications_count = len(certifications)
        certification_names = [c.get('name', '').lower() for c in certifications]

        languages = candidate_raw.get('languages', [])
        languages_count = len(languages)
        language_names = [l.get('language', '').lower() for l in languages]
        has_english_proficiency = any(
            l.get('language', '').lower() == 'english' and
            l.get('proficiency', '').lower() in ['professional', 'full professional', 'native', 'bilingual', 'fluent']
            for l in languages
        )
        
        # Career depth
        career_depth_months = self._analyze_career_depth(career_history)
        
        # Timeline validation
        timeline_issues = self._check_timeline_consistency(career_history)
        
        # Most recent role
        most_recent_role = career_history[0] if career_history else None
        most_recent_role_title = most_recent_role.get('title', '') if most_recent_role else ""
        most_recent_role_company = most_recent_role.get('company', '') if most_recent_role else ""
        most_recent_company_size = most_recent_role.get('company_size', '') if most_recent_role else ""
        
        # Calculate months since last role ended (or current if still in role)
        if most_recent_role and most_recent_role.get('is_current'):
            most_recent_role_ended_months_ago = 0.0
        elif most_recent_role:
            end_date_str = most_recent_role.get('end_date', '')
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                    most_recent_role_ended_months_ago = (reference_date - end_date).days / 30.44
                except ValueError:
                    most_recent_role_ended_months_ago = 999.0
            else:
                most_recent_role_ended_months_ago = 999.0  # Unknown
        else:
            most_recent_role_ended_months_ago = 999.0
        
        # Years since last code (proxy: if title contains engineer and ended <18mo ago)
        is_engineer = any(eng_title in current_title.lower() for eng_title in self.ENGINEERING_TITLES)
        if is_engineer and most_recent_role_ended_months_ago < 18:
            years_since_last_coding = most_recent_role_ended_months_ago / 12.0
        else:
            years_since_last_coding = 999.0  # Assume not coding
        
        # GitHub signal
        github_score = redrob_signals.get('github_activity_score', -1)
        has_github = github_score >= 0
        github_activity_score = max(0, github_score)
        
        # Education tier — take the highest tier found across all education entries
        # tier_1 > tier_2 > tier_3 > tier_4 > unknown
        TIER_ORDER = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]
        best_tier = "unknown"
        for edu in candidate_raw.get("education", []):
            t = edu.get("tier", "unknown")
            try:
                if TIER_ORDER.index(t) < TIER_ORDER.index(best_tier):
                    best_tier = t
            except ValueError:
                pass
        education_tier = best_tier

        # Skill assessment scores
        assessments = redrob_signals.get("skill_assessment_scores", {})
        has_skill_assessments = len(assessments) > 0

        # assessment_relevant_score is computed in feature_scorer with JD context
        # set to 0.0 here, will be overridden during scoring
        assessment_relevant_score = 0.0

        # Behavioral reliability signals
        interview_completion_rate = redrob_signals.get("interview_completion_rate", 0.5)
        raw_acceptance = redrob_signals.get("offer_acceptance_rate", -1)
        offer_acceptance_rate = raw_acceptance  # keep -1 as sentinel for no history
        avg_response_time_hours = redrob_signals.get("avg_response_time_hours", 24.0)
        connection_count = redrob_signals.get('connection_count', 0)
        endorsements_received = redrob_signals.get('endorsements_received', 0)
        expected_salary_range = redrob_signals.get('expected_salary_range_inr_lpa', 0.0)
        expected_salary_range_inr_lpa = self._normalize_salary_range(expected_salary_range)
        preferred_work_mode = str(redrob_signals.get('preferred_work_mode', '')).lower()
        signup_date = redrob_signals.get('signup_date', '')

        # GitHub signal
        github_score = redrob_signals.get('github_activity_score', -1)
        has_github = github_score >= 0
        github_activity_score = max(0, github_score)

        days_since_signup = 999.0
        if signup_date:
            try:
                signup_dt = datetime.strptime(signup_date, '%Y-%m-%d')
                days_since_signup = (reference_date - signup_dt).days
            except ValueError:
                days_since_signup = 999.0

        # Skill trust scores — compute per skill
        # trust = proficiency_weight * 0.4 + duration_weight * 0.4 + endorsement_weight * 0.2
        PROFICIENCY_MAP = {
            "beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0
        }
        skill_trust = {}
        for s in skills:
            name = s.get("name", "")
            prof = PROFICIENCY_MAP.get(s.get("proficiency", "beginner"), 0.25)
            dur = min(s.get("duration_months", 0) / 36.0, 1.0)
            end = min(s.get("endorsements", 0) / 20.0, 1.0)
            trust = prof * 0.4 + dur * 0.4 + end * 0.2
            skill_trust[name.lower()] = trust
        top_skill_trust_scores = skill_trust
        
        return ParsedProfile(
            candidate_id=candidate_id,
            years_experience=years_experience,
            current_title=current_title,
            current_company=current_company,
            company_type=company_type,
            is_consulting_only=is_consulting_only,
            profile_completeness=profile_completeness,
            skill_names=skill_names,
            skill_counts=skill_counts,
            skill_endorsements_mean=skill_endorsements_mean,
            career_history=career_history,
            career_depth_months=career_depth_months,
            timeline_issues=timeline_issues,
            most_recent_role_title=most_recent_role_title,
            most_recent_role_company=most_recent_role_company,
            most_recent_role_ended_months_ago=most_recent_role_ended_months_ago,
            most_recent_company_size=most_recent_company_size,
            years_since_last_coding=years_since_last_coding,
            has_github=has_github,
            github_activity_score=github_activity_score,
            education_tier=education_tier,
            has_skill_assessments=has_skill_assessments,
            assessment_relevant_score=assessment_relevant_score,
            interview_completion_rate=interview_completion_rate,
            offer_acceptance_rate=offer_acceptance_rate,
            avg_response_time_hours=avg_response_time_hours,
            current_industry=current_industry,
            current_company_size=current_company_size,
            country=country,
            certifications_count=certifications_count,
            certification_names=certification_names,
            languages_count=languages_count,
            language_names=language_names,
            has_english_proficiency=has_english_proficiency,
            connection_count=connection_count,
            endorsements_received=endorsements_received,
            expected_salary_range_inr_lpa=expected_salary_range_inr_lpa,
            preferred_work_mode=preferred_work_mode,
            signup_date=signup_date,
            days_since_signup=days_since_signup,
            total_years_experience=years_experience,
            summary=summary,
            headline=headline,
            top_skill_trust_scores=top_skill_trust_scores
        )

    def _normalize_salary_range(self, salary_range: Any) -> float:
        """Normalize salary range values to a single annual LPA number."""
        if isinstance(salary_range, dict):
            min_val = salary_range.get('min')
            max_val = salary_range.get('max')
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                return (min_val + max_val) / 2.0
            if isinstance(min_val, (int, float)):
                return float(min_val)
            if isinstance(max_val, (int, float)):
                return float(max_val)
            return 0.0
        if isinstance(salary_range, (int, float)):
            return float(salary_range)
        if isinstance(salary_range, str):
            try:
                return float(salary_range)
            except ValueError:
                return 0.0
        return 0.0

    def _classify_company(self, career_history: List[Dict]) -> Tuple[str, bool]:
        """Determine company type and if consulting-only"""
        
        company_types = []
        is_consulting_only = True
        
        for role in career_history:
            company = role['company'].lower()
            industry = role['industry'].lower()
            company_size = role['company_size']
            
            # Check if consulting
            is_consulting = any(
                consulting_co in company 
                for consulting_co in self.CONSULTING_COMPANIES
            ) or 'consulting' in industry or 'services' in industry
            
            if is_consulting:
                company_types.append('consulting')
            else:
                is_consulting_only = False
                
                # Classify non-consulting
                if any(x in company for x in ['startup', 'ai', 'tech', 'software']):
                    company_types.append('startup')
                elif any(x in industry for x in ['technology', 'software', 'tech', 'ai', 'ml']):
                    company_types.append('product')
                else:
                    company_types.append('other')
        
        if not company_types:
            company_type = 'unknown'
        else:
            company_type = company_types[0]  # Most recent
        
        return company_type, is_consulting_only
    
    def _analyze_career_depth(self, career_history: List[Dict]) -> Dict[str, int]:
        """Count months in relevant roles"""
        
        depth = {
            'total_months': 0,
            'engineering_months': 0,
            'ml_months': 0,
            'production_months': 0
        }
        
        for role in career_history:
            title = role['title'].lower()
            duration = role['duration_months']
            description = role['description'].lower()
            
            depth['total_months'] += duration
            
            # Engineering roles
            if any(eng_title in title for eng_title in self.ENGINEERING_TITLES):
                depth['engineering_months'] += duration
            
            # ML/AI roles
            if any(ml_term in title for ml_term in ['ml', 'ai', 'machine learning', 'data scientist']):
                depth['ml_months'] += duration
            
            # Production experience
            if any(keyword in description for keyword in self.PRODUCTION_KEYWORDS):
                depth['production_months'] += duration
        
        return depth
    
    def _check_timeline_consistency(self, career_history: List[Dict]) -> List[str]:
        """Check for timeline gaps, overlaps, and inconsistencies"""
        
        issues = []
        
        if not career_history:
            return issues
        
        # Sort by end date descending (most recent first)
        sorted_history = sorted(
            career_history,
            key=lambda x: x['end_date'] or '2099-12-31',
            reverse=True
        )
        
        for i in range(len(sorted_history) - 1):
            current_role = sorted_history[i]
            next_role = sorted_history[i + 1]
            
            try:
                current_end = current_role.get('end_date')
                next_start = next_role.get('start_date')
                
                if current_end and next_start:
                    current_end_date = datetime.strptime(current_end, '%Y-%m-%d')
                    next_start_date = datetime.strptime(next_start, '%Y-%m-%d')
                    
                    # Check for overlap
                    if current_end_date > next_start_date:
                        issues.append(
                            f"Overlapping roles: {current_role.get('company', '')} ends after {next_role.get('company', '')} starts"
                        )
                    
                    # Check for large gaps
                    gap_days = (current_end_date - next_start_date).days
                    if gap_days > 365:
                        issues.append(
                            f"Large gap ({gap_days} days) between {current_role.get('company', '')} and {next_role.get('company', '')}"
                        )
            except (ValueError, TypeError):
                issues.append(f"Could not parse dates for role at {current_role['company']}")
                continue
        
        # Check for very short roles
        for role in career_history:
            if role['duration_months'] < 1 and not role['is_current']:
                issues.append(f"Very short role ({role['duration_months']} months) at {role['company']}")
        
        return issues

    def detect_red_flags(self, parsed_profile: ParsedProfile, candidate_raw: Dict[str, Any]) -> Dict[str, bool]:
        """Detect obvious red flags or suspicious profile signals."""
        flags = {
            'timeline_issues': bool(parsed_profile.timeline_issues),
            'heavy_skill_padding': parsed_profile.skill_counts > 60,
            'low_profile_completeness': parsed_profile.profile_completeness < 40,
            'consulting_only_senior': parsed_profile.is_consulting_only and parsed_profile.years_experience >= 5,
            'stale_coding_experience': parsed_profile.years_since_last_coding > 1.5,
            'not_open_to_work': not candidate_raw.get('redrob_signals', {}).get('open_to_work_flag', True),
            'low_response_rate': candidate_raw.get('redrob_signals', {}).get('recruiter_response_rate', 1.0) < 0.1,
            'research_without_production': False,
        }

        career_text = ' '.join([
            (role.get('description', '') + ' ' + role.get('title', '')).lower()
            for role in candidate_raw.get('career_history', [])
        ])

        if 'research' in career_text and not any(keyword in career_text for keyword in self.PRODUCTION_KEYWORDS):
            flags['research_without_production'] = True

        return flags
