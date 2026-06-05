"""Database models for jobs, candidates, and rankings."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.base import Base


class Job(Base):
    """Job requirement model."""
    
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    role_seniority = Column(String(100), nullable=True)  # Junior, Mid, Senior, Lead
    must_have = Column(JSON, nullable=False, default=[])
    good_to_have = Column(JSON, nullable=False, default=[])
    soft_skills = Column(JSON, nullable=False, default=[])
    expanded_requirements = Column(JSON, nullable=False, default=[])
    ambiguity_detected = Column(JSON, nullable=False, default=[])
    requirement_confidence = Column(Float, nullable=False, default=0.0)
    embedding = Column(JSON, nullable=True)  # Store embedding for retrieval
    parsed_details = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    rankings = relationship("Ranking", back_populates="job", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_job_created_at', 'created_at'),
        Index('idx_job_title', 'title'),
    )


class Candidate(Base):
    """Candidate profile model."""
    
    __tablename__ = "candidates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True, unique=True)
    phone = Column(String(20), nullable=True)
    profile_url = Column(String(500), nullable=True)
    
    # Profile data (structured)
    skills = Column(JSON, nullable=False, default=[])
    experience = Column(JSON, nullable=False, default=[])
    projects = Column(JSON, nullable=False, default=[])
    education = Column(JSON, nullable=False, default=[])
    certifications = Column(JSON, nullable=False, default=[])
    career_progression = Column(JSON, nullable=False, default=[])
    
    # Behavioral signals
    activity_signals = Column(JSON, nullable=False, default={})
    github_profile = Column(String(500), nullable=True)
    open_source_contributions = Column(Integer, default=0)
    portfolio_updates = Column(Integer, default=0)
    community_activity = Column(Integer, default=0)
    
    # Embeddings
    profile_embedding = Column(JSON, nullable=True)
    skills_embedding = Column(JSON, nullable=True)
    experience_embedding = Column(JSON, nullable=True)
    
    # Scores (cached)
    technical_signal_score = Column(Float, default=0.0)
    career_signal_score = Column(Float, default=0.0)
    behavioral_signal_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source = Column(String(100), nullable=False, default="manual")  # upload, linkedin, etc.
    
    # Relationships
    rankings = relationship("Ranking", back_populates="candidate", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_candidate_created_at', 'created_at'),
        Index('idx_candidate_name', 'name'),
        Index('idx_candidate_email', 'email'),
    )


class Ranking(Base):
    """Candidate ranking for a specific job."""
    
    __tablename__ = "rankings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False, index=True)
    
    # Individual component scores
    technical_match_score = Column(Float, default=0.0)
    experience_match_score = Column(Float, default=0.0)
    project_relevance_score = Column(Float, default=0.0)
    behavior_signal_score = Column(Float, default=0.0)
    semantic_similarity_score = Column(Float, default=0.0)
    
    # Final weighted score
    final_score = Column(Float, default=0.0, index=True)
    rank = Column(Integer, nullable=True, index=True)
    
    # LLM evaluation
    llm_overall_score = Column(Float, nullable=True)
    llm_recommendation = Column(String(100), nullable=True)  # Strong Hire, Good Fit, Consider, Not Fit
    llm_strengths = Column(JSON, nullable=False, default=[])
    llm_weaknesses = Column(JSON, nullable=False, default=[])
    llm_reasoning = Column(Text, nullable=True)
    
    # Explainability
    explanation = Column(JSON, nullable=False, default={})
    matched_skills = Column(JSON, nullable=False, default=[])
    missing_skills = Column(JSON, nullable=False, default=[])
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="rankings")
    candidate = relationship("Candidate", back_populates="rankings")
    
    __table_args__ = (
        Index('idx_ranking_job_final_score', 'job_id', 'final_score'),
        Index('idx_ranking_job_candidate', 'job_id', 'candidate_id'),
    )


class ProcessingLog(Base):
    """Log for tracking job/candidate processing."""
    
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(50), nullable=False)  # "job" or "candidate"
    entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    operation = Column(String(100), nullable=False)  # "parse", "embed", "rank", etc.
    status = Column(String(50), nullable=False)  # "pending", "completed", "failed"
    error_message = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_log_entity', 'entity_type', 'entity_id'),
        Index('idx_log_created_at', 'created_at'),
    )
