"""Pydantic schemas for API requests/responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============ Job Schemas ============

class JobRequirementExtract(BaseModel):
    """Extracted and parsed job requirements."""
    
    role: str
    seniority: Optional[str] = None
    must_have: List[str] = []
    good_to_have: List[str] = []
    soft_skills: List[str] = []
    expanded_requirements: Dict[str, List[str]] = {}
    ambiguity_detected: List[Dict[str, Any]] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class JobCreate(BaseModel):
    """Create job request."""
    
    title: str
    description: str
    created_by: str = "system"


class JobUpdate(BaseModel):
    """Update job request."""
    
    title: Optional[str] = None
    description: Optional[str] = None
    must_have: Optional[List[str]] = None
    good_to_have: Optional[List[str]] = None
    soft_skills: Optional[List[str]] = None


class JobResponse(BaseModel):
    """Job response."""
    
    id: str
    title: str
    description: str
    role_seniority: Optional[str]
    must_have: List[str]
    good_to_have: List[str]
    soft_skills: List[str]
    expanded_requirements: Dict[str, List[str]]
    ambiguity_detected: List[Dict[str, Any]]
    requirement_confidence: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============ Candidate Schemas ============

class SkillInfo(BaseModel):
    """Candidate skill."""
    
    name: str
    proficiency: Optional[str] = None  # beginner, intermediate, advanced, expert
    years_of_experience: Optional[int] = None
    endorsements: Optional[int] = 0


class ExperienceInfo(BaseModel):
    """Candidate work experience."""
    
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    current: bool = False
    description: Optional[str] = None
    technologies: List[str] = []


class ProjectInfo(BaseModel):
    """Candidate project."""
    
    name: str
    description: Optional[str] = None
    technologies: List[str] = []
    link: Optional[str] = None
    github_link: Optional[str] = None


class EducationInfo(BaseModel):
    """Candidate education."""
    
    institution: str
    degree: str
    field: Optional[str] = None
    graduation_year: Optional[int] = None


class ActivitySignal(BaseModel):
    """Behavioral activity signal."""
    
    github_contributions: int = 0
    portfolio_updates: int = 0
    open_source_repos: int = 0
    community_posts: int = 0
    last_activity_date: Optional[str] = None


class CandidateCreate(BaseModel):
    """Create candidate request."""
    
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    profile_url: Optional[str] = None
    skills: List[SkillInfo] = []
    experience: List[ExperienceInfo] = []
    projects: List[ProjectInfo] = []
    education: List[EducationInfo] = []
    certifications: List[str] = []
    activity_signals: Optional[ActivitySignal] = None
    github_profile: Optional[str] = None


class CandidateResponse(BaseModel):
    """Candidate response."""
    
    id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    profile_url: Optional[str]
    skills: List[SkillInfo]
    experience: List[ExperienceInfo]
    projects: List[ProjectInfo]
    education: List[EducationInfo]
    certifications: List[str]
    activity_signals: ActivitySignal
    technical_signal_score: float
    career_signal_score: float
    behavioral_signal_score: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============ Ranking Schemas ============

class RankingComponentScores(BaseModel):
    """Individual component scores for ranking."""
    
    technical_match: float
    experience_match: float
    project_relevance: float
    behavior_signal: float
    semantic_similarity: float


class LLMEvaluation(BaseModel):
    """LLM-based evaluation of candidate."""
    
    overall_score: float = Field(ge=0, le=100)
    recommendation: str  # Strong Hire, Good Fit, Consider, Not Fit
    strengths: List[str]
    weaknesses: List[str]
    reasoning: str


class RankingExplanation(BaseModel):
    """Explainability for ranking."""
    
    overall_fit_percentage: float
    component_scores: RankingComponentScores
    strengths: List[str]
    weaknesses: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    key_insights: List[str]
    reasoning: str


class RankingResponse(BaseModel):
    """Ranking response."""
    
    job_id: str
    candidate_id: str
    candidate_name: str
    rank: Optional[int]
    final_score: float
    component_scores: RankingComponentScores
    llm_evaluation: Optional[LLMEvaluation] = None
    explanation: RankingExplanation
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RankingListResponse(BaseModel):
    """List of rankings for a job."""
    
    job_id: str
    job_title: str
    total_candidates: int
    rankings: List[RankingResponse]


# ============ Query Schemas ============

class AmbiguityCheckResponse(BaseModel):
    """Response for ambiguity detection."""
    
    has_ambiguity: bool
    possible_interpretations: List[str]
    clarification_needed: bool
    recommended_questions: List[str]


class CopilotQuery(BaseModel):
    """Natural language query for recruiter copilot."""
    
    query: str
    job_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class CopilotResponse(BaseModel):
    """Response from recruiter copilot."""
    
    response: str
    candidates: Optional[List[Dict[str, Any]]] = None
    reasoning: Optional[str] = None
