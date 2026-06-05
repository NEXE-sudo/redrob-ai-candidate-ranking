"""Service for computing candidate rankings."""

from typing import Dict, List, Any, Tuple
from app.core.config import settings
from app.engines.candidate_intelligence import CandidateIntelligenceService
from app.engines.embedding_retrieval import EmbeddingRetrievalService


class RankingService:
    """Service for computing multi-factor candidate rankings."""
    
    def __init__(self):
        """Initialize ranking service."""
        
        self.candidate_intel = CandidateIntelligenceService()
        self.embedding_service = EmbeddingRetrievalService()
        
        # Load weights from settings
        self.weights = {
            "technical": settings.WEIGHT_TECHNICAL,
            "experience": settings.WEIGHT_EXPERIENCE,
            "projects": settings.WEIGHT_PROJECTS,
            "behaviour": settings.WEIGHT_BEHAVIOUR,
            "semantic": settings.WEIGHT_SEMANTIC,
        }
    
    def compute_ranking_scores(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute all ranking scores for a candidate against a job.
        
        Returns dict with component scores and final score.
        """
        
        # Component scores
        technical_match = self._compute_technical_match(candidate, job_requirements)
        experience_match = self._compute_experience_match(candidate, job_requirements)
        project_relevance = self._compute_project_relevance(candidate, job_requirements)
        behavior_signal = self._compute_behavior_signal(candidate)
        semantic_similarity = self._compute_semantic_similarity(candidate, job_requirements)
        
        # Weighted final score (0-100)
        final_score = (
            technical_match * self.weights["technical"] +
            experience_match * self.weights["experience"] +
            project_relevance * self.weights["projects"] +
            behavior_signal * self.weights["behaviour"] +
            semantic_similarity * self.weights["semantic"]
        ) * 100
        
        return {
            "technical_match": technical_match * 100,
            "experience_match": experience_match * 100,
            "project_relevance": project_relevance * 100,
            "behavior_signal": behavior_signal * 100,
            "semantic_similarity": semantic_similarity * 100,
            "final_score": final_score,
        }
    
    def compute_explainability(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explainable insights for ranking.
        """
        
        # Extract matched and missing skills
        matched_skills, missing_skills = self._compute_skill_match(candidate, job_requirements)
        
        # Extract strengths and weaknesses
        strengths = self._identify_strengths(candidate, job_requirements, matched_skills)
        weaknesses = self._identify_weaknesses(candidate, job_requirements, missing_skills)
        
        # Key insights
        insights = self._generate_insights(
            candidate, job_requirements, scores, strengths, weaknesses
        )
        
        return {
            "overall_fit_percentage": scores["final_score"],
            "component_scores": {
                "technical_match": scores["technical_match"],
                "experience_match": scores["experience_match"],
                "project_relevance": scores["project_relevance"],
                "behavior_signal": scores["behavior_signal"],
                "semantic_similarity": scores["semantic_similarity"],
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "key_insights": insights,
        }
    
    def _compute_technical_match(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> float:
        """Compute technical skill match score (0-1)."""
        
        must_have = set(job_requirements.get("must_have", []))
        good_to_have = set(job_requirements.get("good_to_have", []))
        
        candidate_skills = set([
            s["name"] if isinstance(s, dict) else s
            for s in candidate.get("skills", [])
        ])
        
        # Normalize skill names for comparison
        candidate_skills_lower = set(s.lower() for s in candidate_skills)
        must_have_lower = set(s.lower() for s in must_have)
        good_to_have_lower = set(s.lower() for s in good_to_have)
        
        # Calculate matches
        must_have_matches = len(candidate_skills_lower & must_have_lower)
        good_to_have_matches = len(candidate_skills_lower & good_to_have_lower)
        
        total_required = len(must_have)
        total_nice_to_have = len(good_to_have)
        
        # Scoring
        score = 0.0
        
        # Must-have: weighted 70%
        if total_required > 0:
            must_have_ratio = must_have_matches / total_required
            score += must_have_ratio * 0.7
        else:
            score += 0.7
        
        # Good-to-have: weighted 30%
        if total_nice_to_have > 0:
            good_to_have_ratio = good_to_have_matches / total_nice_to_have
            score += good_to_have_ratio * 0.3
        else:
            score += 0.3
        
        return min(1.0, score)
    
    def _compute_experience_match(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> float:
        """Compute experience match score (0-1)."""
        
        score = 0.0
        
        # Total years of experience
        experience = candidate.get("experience", [])
        total_years = sum(
            self._get_experience_duration(exp) for exp in experience
        )
        
        # Seniority level required
        seniority = job_requirements.get("seniority", "mid")
        
        if seniority == "junior":
            # Junior role: any experience is good
            score += 0.5 if total_years > 0 else 0.2
        elif seniority == "mid":
            # Mid role: prefer 2-5 years
            if 2 <= total_years <= 5:
                score += 0.8
            elif total_years > 5:
                score += 0.7
            elif total_years >= 1:
                score += 0.5
            else:
                score += 0.2
        elif seniority == "senior":
            # Senior role: prefer 5+ years
            if total_years >= 5:
                score += 0.9
            elif total_years >= 3:
                score += 0.6
            else:
                score += 0.2
        
        # Relevant domain experience
        domains = self.candidate_intel.extract_domain_expertise(candidate)
        if domains:
            score += 0.2
        
        return min(1.0, score)
    
    def _compute_project_relevance(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> float:
        """Compute project relevance score (0-1)."""
        
        score = 0.0
        projects = candidate.get("projects", [])
        
        if not projects:
            return 0.2  # Base score for no projects
        
        # Number of projects
        score += min(0.4, len(projects) * 0.1)
        
        # Check technology overlap
        job_techs = set()
        job_techs.update(job_requirements.get("must_have", []))
        job_techs.update(job_requirements.get("good_to_have", []))
        
        for proj in projects:
            if isinstance(proj, dict):
                proj_techs = set(proj.get("technologies", []))
                overlap = len(proj_techs & job_techs)
                if overlap > 0:
                    score += min(0.3, overlap * 0.1)
        
        # Check for portfolio
        if any(isinstance(p, dict) and p.get("link") for p in projects):
            score += 0.2
        
        return min(1.0, score)
    
    def _compute_behavior_signal(self, candidate: Dict[str, Any]) -> float:
        """Compute behavioral signal score (0-1)."""
        
        return self.candidate_intel.compute_behavioral_signal_score(candidate)
    
    def _compute_semantic_similarity(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> float:
        """Compute semantic similarity between job and candidate (0-1)."""
        
        job_profile = self._build_job_profile(job_requirements)
        candidate_profile = self.embedding_service._build_profile_text(candidate)
        
        similarity = self.embedding_service.compute_semantic_similarity(
            job_profile, candidate_profile
        )
        
        return similarity
    
    def _compute_skill_match(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Identify matched and missing skills."""
        
        must_have = set(s.lower() for s in job_requirements.get("must_have", []))
        candidate_skills = set(
            (s["name"] if isinstance(s, dict) else s).lower()
            for s in candidate.get("skills", [])
        )
        
        matched = [s for s in candidate_skills if s in must_have]
        missing = [s for s in must_have if s not in candidate_skills]
        
        return matched, missing
    
    def _identify_strengths(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        matched_skills: List[str]
    ) -> List[str]:
        """Identify candidate strengths."""
        
        strengths = []
        
        # Strong matches
        if len(matched_skills) >= 3:
            strengths.append(f"Strong match on {len(matched_skills)} key skills")
        
        # Experience
        experience = candidate.get("experience", [])
        total_years = sum(self._get_experience_duration(exp) for exp in experience)
        if total_years >= 5:
            strengths.append(f"{total_years:.0f} years of relevant experience")
        
        # Projects
        projects = candidate.get("projects", [])
        if projects:
            strengths.append(f"Portfolio of {len(projects)} projects")
        
        # Behavioral signals
        activities = candidate.get("activity_signals", {})
        if isinstance(activities, dict):
            if activities.get("open_source_repos", 0) > 0:
                strengths.append("Active in open source")
            if activities.get("github_contributions", 0) > 100:
                strengths.append("Consistent code contributor")
        
        # Leadership
        domains = self.candidate_intel.extract_domain_expertise(candidate)
        if "Leadership" in domains:
            strengths.append("Demonstrated leadership experience")
        
        return strengths[:5]  # Top 5
    
    def _identify_weaknesses(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        missing_skills: List[str]
    ) -> List[str]:
        """Identify candidate weaknesses."""
        
        weaknesses = []
        
        # Missing skills
        if missing_skills:
            weaknesses.append(f"Missing {len(missing_skills)} key skills")
        
        # Limited experience
        experience = candidate.get("experience", [])
        total_years = sum(self._get_experience_duration(exp) for exp in experience)
        if total_years < 1:
            weaknesses.append("Limited professional experience")
        
        # No projects
        if not candidate.get("projects"):
            weaknesses.append("No portfolio projects")
        
        # Job hopping
        if len(experience) > 3:
            avg_tenure = total_years / len(experience) if experience else 0
            if avg_tenure < 1:
                weaknesses.append("Frequent job changes")
        
        return weaknesses[:3]  # Top 3
    
    def _generate_insights(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        scores: Dict[str, Any],
        strengths: List[str],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate key insights about the match."""
        
        insights = []
        
        final_score = scores["final_score"]
        
        if final_score >= 80:
            insights.append("Excellent candidate profile match")
        elif final_score >= 60:
            insights.append("Good overall fit with some gaps")
        elif final_score >= 40:
            insights.append("Moderate fit - could grow into role")
        else:
            insights.append("Significant skill gaps present")
        
        # Technical insights
        if scores["technical_match"] >= 80:
            insights.append("Strong technical foundation")
        elif scores["technical_match"] < 40:
            insights.append("Technical skills need development")
        
        # Experience insights
        if scores["experience_match"] >= 80:
            insights.append("Experience level well-aligned")
        
        # Potential insights
        if scores["behavior_signal"] >= 70:
            insights.append("Shows strong learning and engagement")
        
        return insights[:3]
    
    def _get_experience_duration(self, experience: Any) -> float:
        """Get duration of a work experience in years."""
        
        if not isinstance(experience, dict):
            return 0.0
        
        try:
            start = str(experience.get("start_date", ""))
            if experience.get("current"):
                from datetime import datetime
                end = str(datetime.now().year)
            else:
                end = str(experience.get("end_date", ""))
            
            start_year = int(start[:4]) if len(start) >= 4 else 0
            end_year = int(end[:4]) if len(end) >= 4 else 0
            
            return max(0, end_year - start_year)
        except:
            return 0.0
    
    def _build_job_profile(self, job_requirements: Dict[str, Any]) -> str:
        """Build text representation of job."""
        
        parts = [
            job_requirements.get("role", "Unknown Role"),
            " ".join(job_requirements.get("must_have", [])),
            " ".join(job_requirements.get("good_to_have", [])),
            " ".join(job_requirements.get("soft_skills", [])),
        ]
        
        return " ".join(parts)
