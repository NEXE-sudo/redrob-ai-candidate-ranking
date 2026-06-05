"""API routes for candidates."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import json
from app.db.session import get_db
from app.models.database import Candidate
from app.models.schemas import CandidateCreate, CandidateResponse, SkillInfo, ExperienceInfo
from app.engines.candidate_intelligence import CandidateIntelligenceService
from app.engines.embedding_retrieval import EmbeddingRetrievalService

router = APIRouter(prefix="/api/candidates", tags=["candidates"])

candidate_intel = CandidateIntelligenceService()
embedding_service = EmbeddingRetrievalService()


@router.post("/", response_model=CandidateResponse)
async def create_candidate(
    candidate_data: CandidateCreate,
    db: AsyncSession = Depends(get_db)
) -> CandidateResponse:
    """Create a new candidate profile."""
    
    # Convert schemas to JSON-serializable format
    skills_data = [s.dict() for s in candidate_data.skills] if candidate_data.skills else []
    experience_data = [e.dict() for e in candidate_data.experience] if candidate_data.experience else []
    projects_data = [p.dict() for p in candidate_data.projects] if candidate_data.projects else []
    education_data = [e.dict() for e in candidate_data.education] if candidate_data.education else []
    activity_data = candidate_data.activity_signals.dict() if candidate_data.activity_signals else {}
    
    # Compute intelligence scores
    candidate_dict = {
        "name": candidate_data.name,
        "skills": skills_data,
        "experience": experience_data,
        "projects": projects_data,
        "education": education_data,
        "activity_signals": activity_data,
    }
    
    technical_score = candidate_intel.compute_technical_signal_score(candidate_dict)
    career_score = candidate_intel.compute_career_signal_score(candidate_dict)
    behavior_score = candidate_intel.compute_behavioral_signal_score(candidate_dict)
    
    # Create candidate record
    db_candidate = Candidate(
        name=candidate_data.name,
        email=candidate_data.email,
        phone=candidate_data.phone,
        profile_url=candidate_data.profile_url,
        skills=skills_data,
        experience=experience_data,
        projects=projects_data,
        education=education_data,
        certifications=candidate_data.certifications,
        activity_signals=activity_data,
        github_profile=candidate_data.github_profile,
        technical_signal_score=technical_score,
        career_signal_score=career_score,
        behavioral_signal_score=behavior_score,
    )
    
    db.add(db_candidate)
    await db.commit()
    await db.refresh(db_candidate)
    
    # Add to embedding index
    try:
        embedding_service.add_candidate_to_index(str(db_candidate.id), candidate_dict)
    except Exception as e:
        print(f"Warning: Could not add candidate to embedding index: {e}")
    
    return CandidateResponse.from_orm(db_candidate)


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db)
) -> CandidateResponse:
    """Get candidate profile."""
    
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    db_candidate = result.scalars().first()
    
    if not db_candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return CandidateResponse.from_orm(db_candidate)


@router.get("/", response_model=List[CandidateResponse])
async def list_candidates(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0
) -> List[CandidateResponse]:
    """List all candidates."""
    
    result = await db.execute(
        select(Candidate).order_by(Candidate.created_at.desc()).limit(limit).offset(offset)
    )
    candidates = result.scalars().all()
    
    return [CandidateResponse.from_orm(c) for c in candidates]


@router.get("/{candidate_id}/insights", response_model=dict)
async def get_candidate_insights(
    candidate_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Get intelligence insights about a candidate."""
    
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    db_candidate = result.scalars().first()
    
    if not db_candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # Build candidate dict for analysis
    candidate_dict = {
        "name": db_candidate.name,
        "skills": db_candidate.skills,
        "experience": db_candidate.experience,
        "projects": db_candidate.projects,
        "education": db_candidate.education,
        "activity_signals": db_candidate.activity_signals,
    }
    
    # Extract domains
    domains = candidate_intel.extract_domain_expertise(candidate_dict)
    
    return {
        "candidate_id": str(candidate_id),
        "name": db_candidate.name,
        "technical_signal_score": db_candidate.technical_signal_score,
        "career_signal_score": db_candidate.career_signal_score,
        "behavioral_signal_score": db_candidate.behavioral_signal_score,
        "domains_of_expertise": domains,
        "insights": {
            "total_experience_years": sum(
                max(0, (int(e.get("end_date", "2024")[:4]) - int(e.get("start_date", "2020")[:4])))
                if isinstance(e, dict) else 0
                for e in candidate_dict.get("experience", [])
            ),
            "number_of_projects": len(candidate_dict.get("projects", [])),
            "number_of_skills": len(candidate_dict.get("skills", [])),
            "leadership_potential": "Leadership" in domains,
        }
    }
