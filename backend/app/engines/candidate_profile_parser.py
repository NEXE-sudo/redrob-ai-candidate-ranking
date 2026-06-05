"""
Candidate Profile Parser
Extracts structured information from raw candidate profiles for scoring.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ParsedProfile:
    """Structured extracted profile data"""
    candidate_id: str
    years_experience: float
    current_title: str
    current_company: str
    company_type: str  # "product", "consulting", "startup", "other"
    is_consulting_only: bool
    profile_completeness: float
    skill_names: List[str]
    skill_counts: int
    skill_endorsements_mean: float
    career_depth_months: Dict[str, int]
    timeline_issues: List[str]
    most_recent_role_title: str
    most_recent_role_company: str
    most_recent_role_ended_months_ago: float
    most_recent_company_size: str
    years_since_last_coding: float
    has_github: bool
    github_activity_score: float


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
        'ai researcher', 'scientist', 'technical lead', 'tech lead', 'architect'
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
        
        profile = candidate_raw['profile']
        career_history = candidate_raw['career_history']
        skills = candidate_raw['skills']
        redrob_signals = candidate_raw['redrob_signals']
        
        # Basic info
        candidate_id = candidate_raw['candidate_id']
        years_experience = profile['years_of_experience']
        current_title = profile['current_title']
        current_company = profile['current_company']
        
        # Determine company type
        company_type, is_consulting_only = self._classify_company(career_history)
        
        # Profile completeness
        profile_completeness = redrob_signals['profile_completeness_score']
        
        # Skill analysis
        skill_names = [s['name'] for s in skills]
        skill_counts = len(skills)
        endorsements = [s['endorsements'] for s in skills]
        skill_endorsements_mean = sum(endorsements) / len(endorsements) if endorsements else 0
        
        # Career depth
        career_depth_months = self._analyze_career_depth(career_history)
        
        # Timeline validation
        timeline_issues = self._check_timeline_consistency(career_history)
        
        # Most recent role
        most_recent_role = career_history[0] if career_history else None
        most_recent_role_title = most_recent_role['title'] if most_recent_role else ""
        most_recent_role_company = most_recent_role['company'] if most_recent_role else ""
        most_recent_company_size = most_recent_role['company_size'] if most_recent_role else ""
        
        # Calculate months since last role ended (or current if still in role)
        if most_recent_role['is_current']:
            most_recent_role_ended_months_ago = 0.0
        else:
            end_date_str = most_recent_role['end_date']
            if end_date_str:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                most_recent_role_ended_months_ago = (reference_date - end_date).days / 30.44
            else:
                most_recent_role_ended_months_ago = 999  # Unknown
        
        # Years since last code (proxy: if title contains engineer and ended <18mo ago)
        is_engineer = any(eng_title in current_title.lower() for eng_title in self.ENGINEERING_TITLES)
        if is_engineer and most_recent_role_ended_months_ago < 18:
            years_since_last_coding = most_recent_role_ended_months_ago / 12.0
        else:
            years_since_last_coding = 999.0  # Assume not coding
        
        # GitHub signal
        has_github = redrob_signals['github_activity_score'] >= 0
        github_activity_score = max(0, redrob_signals['github_activity_score'])
        
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
            career_depth_months=career_depth_months,
            timeline_issues=timeline_issues,
            most_recent_role_title=most_recent_role_title,
            most_recent_role_company=most_recent_role_company,
            most_recent_role_ended_months_ago=most_recent_role_ended_months_ago,
            most_recent_company_size=most_recent_company_size,
            years_since_last_coding=years_since_last_coding,
            has_github=has_github,
            github_activity_score=github_activity_score
        )
    
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
            
            current_end = current_role['end_date']
            next_start = next_role['start_date']
            
            if current_end and next_start:
                current_end_date = datetime.strptime(current_end, '%Y-%m-%d')
                next_start_date = datetime.strptime(next_start, '%Y-%m-%d')
                
                # Check for overlap
                if current_end_date > next_start_date:
                    issues.append(f"Overlapping roles: {current_role['company']} ends after {next_role['company']} starts")
                
                # Check for large gaps
                gap_days = (current_end_date - next_start_date).days
                if gap_days > 365:
                    issues.append(f"Large gap ({gap_days} days) between {current_role['company']} and {next_role['company']}")
        
        # Check for very short roles
        for role in career_history:
            if role['duration_months'] < 1 and not role['is_current']:
                issues.append(f"Very short role ({role['duration_months']} months) at {role['company']}")
        
        return issues
