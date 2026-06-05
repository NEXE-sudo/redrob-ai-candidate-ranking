"""Service for understanding and parsing job requirements."""

import json
from typing import Dict, List, Any
from app.models.schemas import JobRequirementExtract
from app.core.config import settings


# Semantic expansion mappings for common AI/tech terms
SEMANTIC_EXPANSIONS = {
    "AI": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Generative AI", "LLM"],
    "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Deep Learning", "Reinforcement Learning"],
    "Python": ["Django", "FastAPI", "Flask", "Pandas", "NumPy", "Scikit-learn"],
    "JavaScript": ["React", "Node.js", "TypeScript", "Next.js", "Vue.js"],
    "Cloud": ["AWS", "GCP", "Azure"],
    "AWS": ["EC2", "S3", "Lambda", "RDS", "SageMaker"],
    "DevOps": ["Docker", "Kubernetes", "CI/CD", "Jenkins", "GitHub Actions"],
    "Data": ["Data Analysis", "Data Engineering", "Analytics", "Big Data"],
}

SENIORITY_KEYWORDS = {
    "junior": ["junior", "entry-level", "graduate", "fresher", "internship"],
    "mid": ["mid-level", "mid", "intermediate", "3-5 years"],
    "senior": ["senior", "lead", "staff", "principal", "10+ years", "5+ years"],
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "collaboration", "problem-solving",
    "critical thinking", "creativity", "attention to detail", "time management",
    "adaptability", "emotional intelligence", "mentoring"
}


class RequirementUnderstandingService:
    """Service for parsing and understanding job requirements."""
    
    def parse_requirements(self, job_description: str) -> JobRequirementExtract:
        """Parse job description and extract structured requirements."""
        
        # Normalize text
        description_lower = job_description.lower()
        
        # Extract seniority
        seniority = self._extract_seniority(description_lower)
        
        # Extract skills and requirements
        must_have, good_to_have = self._categorize_requirements(job_description)
        
        # Extract soft skills
        soft_skills = self._extract_soft_skills(description_lower)
        
        # Expand requirements semantically
        expanded = self._expand_requirements(must_have + good_to_have)
        
        # Detect ambiguities
        ambiguities = self._detect_ambiguities(job_description)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            must_have, good_to_have, soft_skills, ambiguities
        )
        
        # Extract role title
        role = self._extract_role_title(job_description)
        
        return JobRequirementExtract(
            role=role,
            seniority=seniority,
            must_have=must_have,
            good_to_have=good_to_have,
            soft_skills=soft_skills,
            expanded_requirements=expanded,
            ambiguity_detected=ambiguities,
            confidence=confidence,
        )
    
    def _extract_seniority(self, description: str) -> str:
        """Detect job seniority level."""
        
        for level, keywords in SENIORITY_KEYWORDS.items():
            if any(keyword in description for keyword in keywords):
                return level
        
        return "mid"
    
    def _categorize_requirements(self, description: str) -> tuple[List[str], List[str]]:
        """Categorize requirements into must-have and good-to-have."""
        
        must_have = []
        good_to_have = []
        
        lines = description.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect sections
            if "must have" in line_lower or "required" in line_lower:
                current_section = "must_have"
                continue
            elif "good to have" in line_lower or "nice to have" in line_lower or "preferred" in line_lower:
                current_section = "good_to_have"
                continue
            
            # Extract skills from current section
            if current_section and line.strip() and not line.startswith('#'):
                # Remove bullets, dashes, etc.
                skill = line.strip().lstrip('•-*· ').strip()
                if skill and len(skill) > 2:
                    if current_section == "must_have":
                        must_have.append(skill)
                    else:
                        good_to_have.append(skill)
        
        # If no sections detected, use simple heuristics
        if not must_have and not good_to_have:
            # Look for common tech terms
            common_techs = [
                "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust",
                "React", "Django", "FastAPI", "Node.js", "AWS", "Docker", "Kubernetes",
                "SQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
                "Machine Learning", "AI", "Deep Learning", "NLP",
                "REST API", "GraphQL", "Microservices", "CI/CD",
            ]
            
            for tech in common_techs:
                if tech.lower() in description.lower():
                    # First 5 are more likely must-have
                    if len(must_have) < 5:
                        must_have.append(tech)
                    else:
                        good_to_have.append(tech)
        
        return must_have, good_to_have
    
    def _extract_soft_skills(self, description: str) -> List[str]:
        """Extract soft skills mentioned in the description."""
        
        found_skills = []
        
        for skill in SOFT_SKILLS:
            if skill in description:
                found_skills.append(skill.title())
        
        return found_skills
    
    def _expand_requirements(self, requirements: List[str]) -> Dict[str, List[str]]:
        """Expand requirements semantically."""
        
        expanded = {}
        
        for req in requirements:
            if req in SEMANTIC_EXPANSIONS:
                expanded[req] = SEMANTIC_EXPANSIONS[req]
        
        return expanded
    
    def _detect_ambiguities(self, description: str) -> List[Dict[str, Any]]:
        """Detect vague or ambiguous requirements."""
        
        ambiguities = []
        vague_terms = [
            ("good", "Good is subjective. What specific qualities?"),
            ("experience", "How many years of experience?"),
            ("etc.", "Incomplete requirement list"),
            ("and more", "Additional unspecified requirements"),
        ]
        
        for term, concern in vague_terms:
            if term.lower() in description.lower():
                ambiguities.append({
                    "term": term,
                    "concern": concern,
                })
        
        return ambiguities
    
    def _calculate_confidence(
        self, 
        must_have: List[str], 
        good_to_have: List[str], 
        soft_skills: List[str],
        ambiguities: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in requirement extraction."""
        
        confidence = 0.5  # base confidence
        
        # More must-haves = higher confidence
        if must_have:
            confidence += min(0.2, len(must_have) * 0.05)
        
        # Well-categorized = higher confidence
        if good_to_have:
            confidence += 0.1
        
        # Soft skills included = higher confidence
        if soft_skills:
            confidence += 0.1
        
        # Ambiguities reduce confidence
        confidence -= min(0.2, len(ambiguities) * 0.1)
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_role_title(self, description: str) -> str:
        """Extract role title from description."""
        
        # Look for common role title patterns
        lines = description.split('\n')
        
        for line in lines[:5]:  # Check first few lines
            line = line.strip()
            if line and len(line) < 100 and not any(c in line for c in [':', '#', '*']):
                return line
        
        # Default
        return "Unknown Role"
