"""API routes for rankings."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List
import uuid
from app.db.session import get_db
from app.models.database import Job, Candidate, Ranking
from app.models.schemas import RankingResponse, RankingListResponse, CopilotQuery, CopilotResponse
from app.engines.ranking import RankingService
from app.engines.embedding_retrieval import EmbeddingRetrievalService
from app.engines.llm_recruiter import LLMRecruiterService

router = APIRouter(prefix="/api/rankings", tags=["rankings"])

ranking_service = RankingService()
embedding_service = EmbeddingRetrievalService()
llm_service = LLMRecruiterService()


@router.post("/evaluate-job/{job_id}")
async def evaluate_job(
    job_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Evaluate all candidates for a job and generate rankings.
    
    This will:
    1. Retrieve similar candidates using embeddings
    2. Compute multi-factor scores
    3. Evaluate top candidates with LLM
    4. Store rankings
    """
    
    # Get job
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    db_job = job_result.scalars().first()
    
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_requirements = {
        "role": db_job.title,
        "seniority": db_job.role_seniority,
        "must_have": db_job.must_have,
        "good_to_have": db_job.good_to_have,
        "soft_skills": db_job.soft_skills,
    }
    
    # Retrieve similar candidates using embeddings
    job_description = f"{db_job.title} {db_job.description}"
    similar_candidates = embedding_service.retrieve_similar_candidates(
        job_description, 
        top_k=100
    )
    
    if not similar_candidates:
        # If no embeddings, get all candidates
        candidate_result = await db.execute(select(Candidate))
        all_candidates = candidate_result.scalars().all()
    else:
        # Get candidate IDs from similar results
        candidate_ids = [cand_id for cand_id, _ in similar_candidates]
        candidate_result = await db.execute(
            select(Candidate).where(Candidate.id.in_(candidate_ids))
        )
        all_candidates = candidate_result.scalars().all()
    
    rankings_created = 0
    
    # Compute rankings for each candidate
    for candidate in all_candidates:
        candidate_dict = {
            "name": candidate.name,
            "skills": candidate.skills,
            "experience": candidate.experience,
            "projects": candidate.projects,
            "education": candidate.education,
            "certifications": candidate.certifications,
            "activity_signals": candidate.activity_signals,
        }
        
        # Compute scores
        scores = ranking_service.compute_ranking_scores(candidate_dict, job_requirements)
        
        # Compute explainability
        explanation_data = ranking_service.compute_explainability(
            candidate_dict, job_requirements, scores
        )
        
        # Create or update ranking
        existing_ranking = await db.execute(
            select(Ranking).where(
                and_(Ranking.job_id == job_id, Ranking.candidate_id == candidate.id)
            )
        )
        db_ranking = existing_ranking.scalars().first()
        
        if db_ranking:
            # Update existing ranking
            db_ranking.technical_match_score = scores["technical_match"]
            db_ranking.experience_match_score = scores["experience_match"]
            db_ranking.project_relevance_score = scores["project_relevance"]
            db_ranking.behavior_signal_score = scores["behavior_signal"]
            db_ranking.semantic_similarity_score = scores["semantic_similarity"]
            db_ranking.final_score = scores["final_score"]
            db_ranking.explanation = explanation_data
            db_ranking.matched_skills = explanation_data.get("matched_skills", [])
            db_ranking.missing_skills = explanation_data.get("missing_skills", [])
        else:
            # Create new ranking
            db_ranking = Ranking(
                job_id=job_id,
                candidate_id=candidate.id,
                technical_match_score=scores["technical_match"],
                experience_match_score=scores["experience_match"],
                project_relevance_score=scores["project_relevance"],
                behavior_signal_score=scores["behavior_signal"],
                semantic_similarity_score=scores["semantic_similarity"],
                final_score=scores["final_score"],
                explanation=explanation_data,
                matched_skills=explanation_data.get("matched_skills", []),
                missing_skills=explanation_data.get("missing_skills", []),
            )
            db.add(db_ranking)
        
        rankings_created += 1
    
    # Get rankings ordered by score
    ranking_result = await db.execute(
        select(Ranking)
        .where(Ranking.job_id == job_id)
        .order_by(Ranking.final_score.desc())
    )
    all_rankings = ranking_result.scalars().all()
    
    # Assign rank numbers
    for rank, ranking in enumerate(all_rankings, 1):
        ranking.rank = rank
    
    await db.commit()
    
    # Evaluate top candidates with LLM
    top_k = min(5, len(all_rankings))
    for i, ranking in enumerate(all_rankings[:top_k]):
        candidate = await db.get(Candidate, ranking.candidate_id)
        
        candidate_dict = {
            "name": candidate.name,
            "skills": candidate.skills,
            "experience": candidate.experience,
            "projects": candidate.projects,
            "education": candidate.education,
            "activity_signals": candidate.activity_signals,
        }
        
        preliminary_scores = {
            "technical_match": ranking.technical_match_score,
            "experience_match": ranking.experience_match_score,
            "project_relevance": ranking.project_relevance_score,
            "behavior_signal": ranking.behavior_signal_score,
            "semantic_similarity": ranking.semantic_similarity_score,
            "final_score": ranking.final_score,
        }
        
        try:
            llm_eval = await llm_service.evaluate_candidate(
                candidate_dict, job_requirements, preliminary_scores
            )
            
            ranking.llm_overall_score = llm_eval.get("overall_score", 0)
            ranking.llm_recommendation = llm_eval.get("recommendation", "Unable to evaluate")
            ranking.llm_strengths = llm_eval.get("strengths", [])
            ranking.llm_weaknesses = llm_eval.get("weaknesses", [])
            ranking.llm_reasoning = llm_eval.get("reasoning", "")
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
    
    await db.commit()
    
    return {
        "job_id": str(job_id),
        "job_title": db_job.title,
        "rankings_created": rankings_created,
        "message": f"Evaluated {rankings_created} candidates"
    }


@router.get("/{job_id}", response_model=RankingListResponse)
async def get_rankings(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 50
) -> RankingListResponse:
    """Get rankings for a job."""
    
    # Get job
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    db_job = job_result.scalars().first()
    
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
    
    # Get candidates
    ranking_responses = []
    for ranking in rankings:
        candidate = await db.get(Candidate, ranking.candidate_id)
        
        ranking_response = RankingResponse(
            job_id=str(ranking.job_id),
            candidate_id=str(ranking.candidate_id),
            candidate_name=candidate.name if candidate else "Unknown",
            rank=ranking.rank,
            final_score=ranking.final_score,
            component_scores={
                "technical_match": ranking.technical_match_score,
                "experience_match": ranking.experience_match_score,
                "project_relevance": ranking.project_relevance_score,
                "behavior_signal": ranking.behavior_signal_score,
                "semantic_similarity": ranking.semantic_similarity_score,
            },
            explanation=ranking.explanation,
        )
        ranking_responses.append(ranking_response)
    
    return RankingListResponse(
        job_id=str(job_id),
        job_title=db_job.title,
        total_candidates=len(rankings),
        rankings=ranking_responses,
    )


@router.post("/copilot/query", response_model=CopilotResponse)
async def copilot_query(
    query_data: CopilotQuery,
    db: AsyncSession = Depends(get_db)
) -> CopilotResponse:
    """
    Natural language query for recruiter copilot.
    
    Supports queries like:
    - Why is Candidate A ranked above Candidate B?
    - Show candidates missing only one skill
    - Find candidates with strong leadership potential
    """
    
    query = query_data.query.lower()
    
    # Route to appropriate handler
    if "why is" in query or "ranked above" in query or "compare" in query:
        response = await handle_comparison_query(query, query_data.job_id, db)
    elif "missing" in query and "skill" in query:
        response = await handle_skill_gap_query(query, query_data.job_id, db)
    elif "leadership" in query or "growth" in query or "potential" in query:
        response = await handle_potential_query(query, query_data.job_id, db)
    else:
        response = await handle_general_query(query, query_data.job_id, db)
    
    return response


async def handle_comparison_query(query: str, job_id: str, db: AsyncSession) -> CopilotResponse:
    """Handle candidate comparison queries."""
    
    if not job_id:
        return CopilotResponse(response="Please provide a job ID for comparison")
    
    # Get top 2 candidates
    ranking_result = await db.execute(
        select(Ranking)
        .where(Ranking.job_id == job_id)
        .order_by(Ranking.final_score.desc())
        .limit(2)
    )
    rankings = ranking_result.scalars().all()
    
    if len(rankings) < 2:
        return CopilotResponse(response="Not enough candidates to compare")
    
    cand1 = await db.get(Candidate, rankings[0].candidate_id)
    cand2 = await db.get(Candidate, rankings[1].candidate_id)
    
    response = f"""
**{cand1.name}** is ranked above **{cand2.name}** because:

1. **Overall Score**: {rankings[0].final_score:.1f}% vs {rankings[1].final_score:.1f}%

2. **Technical Match**: {rankings[0].technical_match_score:.1f}% vs {rankings[1].technical_match_score:.1f}%

3. **Experience Match**: {rankings[0].experience_match_score:.1f}% vs {rankings[1].experience_match_score:.1f}%

4. **Key Strengths of {cand1.name}**:
{chr(10).join(f"   - {s}" for s in rankings[0].explanation.get("strengths", [])[:3])}

5. **Areas for {cand2.name}**:
{chr(10).join(f"   - {s}" for s in rankings[1].explanation.get("weaknesses", [])[:3])}
"""
    
    return CopilotResponse(response=response.strip(), candidates=[
        {"name": cand1.name, "score": rankings[0].final_score},
        {"name": cand2.name, "score": rankings[1].final_score},
    ])


async def handle_skill_gap_query(query: str, job_id: str, db: AsyncSession) -> CopilotResponse:
    """Handle skill gap queries."""
    
    if not job_id:
        return CopilotResponse(response="Please provide a job ID")
    
    # Get rankings
    ranking_result = await db.execute(
        select(Ranking).where(Ranking.job_id == job_id).order_by(Ranking.final_score.desc())
    )
    rankings = ranking_result.scalars().all()
    
    # Find candidates with exactly 1 missing skill
    candidates_one_gap = []
    for ranking in rankings:
        missing = ranking.explanation.get("missing_skills", [])
        if len(missing) == 1:
            candidate = await db.get(Candidate, ranking.candidate_id)
            candidates_one_gap.append({
                "name": candidate.name,
                "score": ranking.final_score,
                "missing_skill": missing[0]
            })
    
    if not candidates_one_gap:
        response = "No candidates found with exactly one missing skill"
    else:
        response = f"**Candidates missing only one skill ({len(candidates_one_gap)} found)**:\n\n"
        for c in candidates_one_gap[:5]:
            response += f"- **{c['name']}** (Score: {c['score']:.1f}%) - Missing: {c['missing_skill']}\n"
    
    return CopilotResponse(response=response, candidates=candidates_one_gap)


async def handle_potential_query(query: str, job_id: str, db: AsyncSession) -> CopilotResponse:
    """Handle growth potential queries."""
    
    if not job_id:
        return CopilotResponse(response="Please provide a job ID")
    
    # Get candidates
    candidate_result = await db.execute(select(Candidate))
    candidates = candidate_result.scalars().all()
    
    # Score by leadership potential and activity
    potential_candidates = []
    for c in candidates:
        leadership_score = c.career_signal_score
        activity_score = c.behavioral_signal_score
        potential = (leadership_score * 0.6 + activity_score * 0.4) * 100
        
        potential_candidates.append({
            "name": c.name,
            "potential_score": potential,
            "career_signal": c.career_signal_score * 100,
            "activity_signal": c.behavioral_signal_score * 100,
        })
    
    top_potential = sorted(potential_candidates, key=lambda x: x["potential_score"], reverse=True)[:5]
    
    response = "**Candidates with Strong Growth Potential**:\n\n"
    for c in top_potential:
        response += f"- **{c['name']}** (Potential: {c['potential_score']:.1f}%) - Career: {c['career_signal']:.1f}%, Activity: {c['activity_signal']:.1f}%\n"
    
    return CopilotResponse(response=response, candidates=top_potential)


async def handle_general_query(query: str, job_id: str, db: AsyncSession) -> CopilotResponse:
    """Handle general queries."""
    
    return CopilotResponse(
        response="I can help you with: comparing candidates, finding candidates with skill gaps, or identifying high-potential candidates. Please rephrase your question.",
        reasoning="Query did not match specific patterns"
    )
