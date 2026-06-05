"""Service for LLM-based candidate evaluation."""

import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from app.core.config import settings


class LLMRecruiterService:
    """Service for LLM-powered recruiter agent."""
    
    def __init__(self):
        """Initialize LLM service."""
        
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        preliminary_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate candidate for a role using LLM.
        
        Only called for top candidates after initial ranking.
        """
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            candidate, job_requirements, preliminary_scores
        )
        
        try:
            # Call LLM
            response = self.model.generate_content(prompt)
            
            # Parse response
            evaluation = self._parse_evaluation_response(response.text)
            
            return evaluation
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return self._default_evaluation()
    
    def _build_evaluation_prompt(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        scores: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM evaluation."""
        
        candidate_profile = self._format_candidate_profile(candidate)
        job_profile = self._format_job_profile(job_requirements)
        
        prompt = f"""You are an expert recruiter evaluating a candidate for a specific role.

## JOB REQUIREMENTS
{job_profile}

## CANDIDATE PROFILE
{candidate_profile}

## PRELIMINARY SCORES
- Technical Match: {scores.get('technical_match', 0):.1f}%
- Experience Match: {scores.get('experience_match', 0):.1f}%
- Project Relevance: {scores.get('project_relevance', 0):.1f}%
- Behavior Signal: {scores.get('behavior_signal', 0):.1f}%
- Semantic Similarity: {scores.get('semantic_similarity', 0):.1f}%
- Overall Score: {scores.get('final_score', 0):.1f}%

## EVALUATION TASK
Conduct a detailed evaluation of this candidate for the role. Consider:

1. **Technical Fit**: Does the candidate have the required technical skills?
2. **Experience Relevance**: Is their experience directly applicable?
3. **Career Growth**: What is their growth trajectory? Any leadership potential?
4. **Project Relevance**: Do their past projects demonstrate capability?
5. **Behavioral Indicators**: What do their activity signals suggest?
6. **Gaps & Risks**: What are the main weaknesses or concerns?
7. **Overall Assessment**: Would you recommend this candidate?

## RESPONSE FORMAT
Return your evaluation as valid JSON (no markdown, just the raw JSON object):
{{
  "overall_score": <0-100 number>,
  "recommendation": "<Strong Hire|Good Fit|Consider|Not Fit>",
  "strengths": ["<strength1>", "<strength2>", ...],
  "weaknesses": ["<weakness1>", "<weakness2>", ...],
  "reasoning": "<1-2 sentence summary>",
  "growth_potential": "<assessment of growth potential>"
}}

Ensure the JSON is valid and can be parsed. Focus on substantive feedback."""

        return prompt
    
    def _format_candidate_profile(self, candidate: Dict[str, Any]) -> str:
        """Format candidate profile for prompt."""
        
        profile_parts = []
        
        # Name and contact
        profile_parts.append(f"**Name**: {candidate.get('name', 'N/A')}")
        if candidate.get('email'):
            profile_parts.append(f"**Email**: {candidate['email']}")
        
        # Skills
        skills = candidate.get('skills', [])
        if skills:
            skill_names = [
                s['name'] if isinstance(s, dict) else s for s in skills
            ]
            profile_parts.append(f"**Skills**: {', '.join(skill_names)}")
        
        # Experience
        experience = candidate.get('experience', [])
        if experience:
            profile_parts.append("**Work Experience**:")
            for exp in experience[:5]:  # Top 5 experiences
                if isinstance(exp, dict):
                    title = exp.get('title', 'N/A')
                    company = exp.get('company', 'N/A')
                    desc = exp.get('description', '')
                    exp_text = f"  - {title} at {company}"
                    if desc:
                        exp_text += f": {desc}"
                    profile_parts.append(exp_text)
        
        # Projects
        projects = candidate.get('projects', [])
        if projects:
            profile_parts.append("**Projects**:")
            for proj in projects[:3]:  # Top 3 projects
                if isinstance(proj, dict):
                    name = proj.get('name', 'N/A')
                    desc = proj.get('description', '')
                    techs = ", ".join(proj.get('technologies', []))
                    proj_text = f"  - {name}"
                    if desc:
                        proj_text += f": {desc}"
                    if techs:
                        proj_text += f" ({techs})"
                    profile_parts.append(proj_text)
        
        # Education
        education = candidate.get('education', [])
        if education:
            profile_parts.append("**Education**:")
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get('degree', 'N/A')
                    field = edu.get('field', '')
                    institution = edu.get('institution', 'N/A')
                    edu_text = f"  - {degree}"
                    if field:
                        edu_text += f" in {field}"
                    edu_text += f" from {institution}"
                    profile_parts.append(edu_text)
        
        # Activity signals
        activities = candidate.get('activity_signals', {})
        if activities:
            profile_parts.append("**Activity Signals**:")
            if isinstance(activities, dict):
                if activities.get('github_contributions', 0) > 0:
                    profile_parts.append(f"  - GitHub Contributions: {activities['github_contributions']}")
                if activities.get('open_source_repos', 0) > 0:
                    profile_parts.append(f"  - Open Source Repos: {activities['open_source_repos']}")
        
        return "\n".join(profile_parts)
    
    def _format_job_profile(self, job_requirements: Dict[str, Any]) -> str:
        """Format job profile for prompt."""
        
        profile_parts = []
        
        profile_parts.append(f"**Role**: {job_requirements.get('role', 'N/A')}")
        
        if job_requirements.get('seniority'):
            profile_parts.append(f"**Seniority Level**: {job_requirements['seniority']}")
        
        if job_requirements.get('must_have'):
            profile_parts.append(f"**Must Have**: {', '.join(job_requirements['must_have'])}")
        
        if job_requirements.get('good_to_have'):
            profile_parts.append(f"**Good To Have**: {', '.join(job_requirements['good_to_have'])}")
        
        if job_requirements.get('soft_skills'):
            profile_parts.append(f"**Soft Skills**: {', '.join(job_requirements['soft_skills'])}")
        
        return "\n".join(profile_parts)
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        
        try:
            # Try to extract JSON from response
            json_str = response_text
            
            # Look for JSON block
            if '{' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
            
            evaluation = json.loads(json_str)
            
            # Validate structure
            required_fields = ['overall_score', 'recommendation', 'strengths', 'weaknesses', 'reasoning']
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = None
            
            # Ensure score is 0-100
            if isinstance(evaluation.get('overall_score'), (int, float)):
                evaluation['overall_score'] = max(0, min(100, evaluation['overall_score']))
            
            return evaluation
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {response_text}")
            return self._default_evaluation()
    
    def _default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation structure."""
        
        return {
            "overall_score": 0,
            "recommendation": "Unable to evaluate",
            "strengths": [],
            "weaknesses": [],
            "reasoning": "Unable to process candidate evaluation",
            "growth_potential": "Unknown"
        }
