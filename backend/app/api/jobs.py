"""API routes for jobs."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
from app.db.session import get_db
from app.models.database import Job, Ranking, Candidate
from app.models.schemas import JobCreate, JobResponse, JobRequirementExtract, AmbiguityCheckResponse
from app.engines.requirement_engine import RequirementUnderstandingService

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

requirement_service = RequirementUnderstandingService()


@router.post("/", response_model=JobResponse)
async def create_job(
    job: JobCreate,
    db: AsyncSession = Depends(get_db)
) -> JobResponse:
    """Create a new job posting."""
    
    # Parse requirements
    requirements = requirement_service.parse_requirements(job.description)
    
    # Create job record
    db_job = Job(
        title=job.title,
        description=job.description,
        role_seniority=requirements.seniority,
        must_have=requirements.must_have,
        good_to_have=requirements.good_to_have,
        soft_skills=requirements.soft_skills,
        expanded_requirements=requirements.expanded_requirements,
        ambiguity_detected=requirements.ambiguity_detected,
        requirement_confidence=requirements.confidence,
        parsed_details=requirements.dict(),
        created_by=job.created_by,
    )
    
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    
    return JobResponse.from_orm(db_job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db)
) -> JobResponse:
    """Get job details."""
    
    result = await db.execute(select(Job).where(Job.id == job_id))
    db_job = result.scalars().first()
    
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse.from_orm(db_job)


@router.get("/", response_model=List[JobResponse])
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0
) -> List[JobResponse]:
    """List all jobs."""
    
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
    )
    jobs = result.scalars().all()
    
    return [JobResponse.from_orm(job) for job in jobs]


@router.post("/{job_id}/check-ambiguity", response_model=AmbiguityCheckResponse)
async def check_ambiguity(
    job_id: str,
    db: AsyncSession = Depends(get_db)
) -> AmbiguityCheckResponse:
    """Check for ambiguities in job requirements."""
    
    result = await db.execute(select(Job).where(Job.id == job_id))
    db_job = result.scalars().first()
    
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    ambiguities = db_job.ambiguity_detected
    
    # Generate clarification questions
    clarification_needed = len(ambiguities) > 0
    recommended_questions = []
    
    if "experience" in str(ambiguities).lower():
        recommended_questions.append("How many years of experience are required?")
    
    if db_job.requirement_confidence < 0.6:
        clarification_needed = True
        recommended_questions.append("Could you provide more specific role requirements?")
    
    possible_interpretations = [
        a.get("term") for a in ambiguities if isinstance(a, dict)
    ]
    
    return AmbiguityCheckResponse(
        has_ambiguity=len(ambiguities) > 0,
        possible_interpretations=possible_interpretations,
        clarification_needed=clarification_needed,
        recommended_questions=recommended_questions,
    )


@router.get("/{job_id}/rankings", response_model=dict)
async def get_job_rankings(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 50
) -> dict:
    """Get top candidates for a job."""
    
    # Get job
    result = await db.execute(select(Job).where(Job.id == job_id))
    db_job = result.scalars().first()
    
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get rankings
    ranking_result = await db.execute(
        select(Ranking)
        .where(Ranking.job_id == job_id)
        .order_by(Ranking.final_score.desc())
        .limit(limit)
    )
    rankings = ranking_result.scalars().all()
    
    total_candidates = await db.execute(
        func.count(Ranking.id).select_from(Ranking).where(Ranking.job_id == job_id)
    )
    total = total_candidates.scalar()
    
    return {
        "job_id": str(job_id),
        "job_title": db_job.title,
        "total_candidates": total,
        "rankings": rankings,
    }
