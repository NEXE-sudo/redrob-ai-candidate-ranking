"""Service for analyzing candidate profiles and computing intelligence signals."""

from typing import Dict, List, Any
from datetime import datetime


class CandidateIntelligenceService:
    """Service for analyzing candidates and computing intelligence signals."""
    
    def compute_technical_signal_score(self, candidate_data: Dict[str, Any]) -> float:
        """
        Compute technical signal score based on skills, experience, and depth.
        
        Score: 0-1
        """
        
        score = 0.0
        max_score = 1.0
        
        # Skills score
        skills = candidate_data.get("skills", [])
        if skills:
            # More skills = higher score (up to 0.3)
            score += min(0.3, len(skills) * 0.05)
            
            # Check for advanced proficiency
            advanced_count = sum(
                1 for s in skills 
                if isinstance(s, dict) and s.get("proficiency") in ["advanced", "expert"]
            )
            score += min(0.15, advanced_count * 0.08)
        
        # Experience depth
        experience = candidate_data.get("experience", [])
        if experience:
            # Years of experience
            total_years = self._calculate_total_years(experience)
            score += min(0.25, total_years * 0.02)
            
            # Technology diversity
            tech_count = 0
            for exp in experience:
                if isinstance(exp, dict):
                    tech_count += len(exp.get("technologies", []))
            score += min(0.15, tech_count * 0.02)
        
        # Certifications
        certs = candidate_data.get("certifications", [])
        if certs:
            score += min(0.15, len(certs) * 0.05)
        
        return min(max_score, score)
    
    def compute_career_signal_score(self, candidate_data: Dict[str, Any]) -> float:
        """
        Compute career signal score based on progression, growth, and trajectory.
        
        Score: 0-1
        """
        
        score = 0.0
        max_score = 1.0
        
        experience = candidate_data.get("experience", [])
        
        if not experience:
            return 0.0
        
        # Career progression analysis
        career_progression = self._analyze_career_progression(experience)
        
        # Score based on progression type
        if career_progression["type"] == "growth":
            score += 0.4
        elif career_progression["type"] == "lateral":
            score += 0.2
        elif career_progression["type"] == "decline":
            score -= 0.1
        
        # Role growth (IC -> Lead -> Manager)
        if career_progression["has_leadership"]:
            score += 0.3
        
        # Company transitions (startup -> FAANG or FAANG -> startup)
        if career_progression["has_prestige_transition"]:
            score += 0.2
        
        # Consistency (no job hopping)
        if career_progression["tenure_consistency"] > 0.7:
            score += 0.15
        else:
            score -= 0.1
        
        return max(0.0, min(max_score, score))
    
    def compute_behavioral_signal_score(self, candidate_data: Dict[str, Any]) -> float:
        """
        Compute behavioral signal score based on activity and engagement.
        
        Score: 0-1
        """
        
        score = 0.3  # base score for having data
        max_score = 1.0
        
        # Projects
        projects = candidate_data.get("projects", [])
        if projects:
            score += min(0.2, len(projects) * 0.05)
        
        # Open source activity
        activities = candidate_data.get("activity_signals", {})
        if isinstance(activities, dict):
            github_contrib = activities.get("github_contributions", 0)
            if github_contrib > 100:
                score += 0.2
            elif github_contrib > 10:
                score += 0.1
            
            open_source = activities.get("open_source_repos", 0)
            if open_source > 0:
                score += 0.15
            
            community = activities.get("community_posts", 0)
            if community > 0:
                score += 0.1
        
        # Profile updates
        portfolio = activities.get("portfolio_updates", 0) if isinstance(activities, dict) else 0
        if portfolio > 0:
            score += 0.05
        
        return min(max_score, score)
    
    def extract_domain_expertise(self, candidate_data: Dict[str, Any]) -> List[str]:
        """Extract identified domain expertise areas from candidate profile."""
        
        domains = set()
        
        # From skills
        skills = candidate_data.get("skills", [])
        for skill in skills:
            if isinstance(skill, dict):
                name = skill.get("name", "").lower()
            else:
                name = str(skill).lower()
            
            # Categorize skills into domains
            if any(term in name for term in ["machine learning", "ai", "deep learning", "nlp", "computer vision"]):
                domains.add("Machine Learning & AI")
            if any(term in name for term in ["python", "java", "c++", "javascript", "rust"]):
                domains.add("Backend Development")
            if any(term in name for term in ["react", "vue", "angular", "css", "html"]):
                domains.add("Frontend Development")
            if any(term in name for term in ["aws", "gcp", "azure", "docker", "kubernetes"]):
                domains.add("Cloud & DevOps")
            if any(term in name for term in ["sql", "nosql", "mongodb", "postgresql"]):
                domains.add("Data Engineering")
        
        # From experience
        experience = candidate_data.get("experience", [])
        for exp in experience:
            if isinstance(exp, dict):
                title = exp.get("title", "").lower()
                if any(term in title for term in ["data", "analytics"]):
                    domains.add("Data Engineering")
                if any(term in title for term in ["ml", "ai", "machine learning"]):
                    domains.add("Machine Learning & AI")
                if any(term in title for term in ["lead", "senior", "architect"]):
                    domains.add("Leadership")
        
        return list(domains)
    
    def _calculate_total_years(self, experience: List[Any]) -> float:
        """Calculate total years of experience."""
        
        total_years = 0.0
        
        for exp in experience:
            if isinstance(exp, dict):
                start = exp.get("start_date")
                end = exp.get("end_date")
                current = exp.get("current", False)
                
                try:
                    if start:
                        start_year = int(str(start)[:4]) if len(str(start)) >= 4 else 2020
                        
                        if current or not end:
                            end_year = datetime.now().year
                        else:
                            end_year = int(str(end)[:4]) if len(str(end)) >= 4 else datetime.now().year
                        
                        total_years += max(0, end_year - start_year)
                except:
                    pass
        
        return total_years
    
    def _analyze_career_progression(self, experience: List[Any]) -> Dict[str, Any]:
        """Analyze career progression patterns."""
        
        progression = {
            "type": "neutral",  # growth, lateral, decline
            "has_leadership": False,
            "has_prestige_transition": False,
            "tenure_consistency": 0.5,
        }
        
        if not experience:
            return progression
        
        # Sort by date (most recent first)
        sorted_exp = sorted(
            [e for e in experience if isinstance(e, dict)],
            key=lambda x: x.get("start_date", ""),
            reverse=True
        )
        
        if len(sorted_exp) < 2:
            progression["tenure_consistency"] = 1.0
            return progression
        
        # Check for leadership
        leadership_keywords = ["lead", "manager", "director", "vp", "head", "chief"]
        for exp in sorted_exp:
            title = exp.get("title", "").lower()
            if any(keyword in title for keyword in leadership_keywords):
                progression["has_leadership"] = True
                break
        
        # Check progression direction
        if sorted_exp[0].get("title", "") != sorted_exp[-1].get("title", ""):
            progression["type"] = "growth"
        
        # Calculate tenure consistency (no job hopping)
        years_per_role = []
        for exp in sorted_exp:
            try:
                start = int(str(exp.get("start_date", ""))[:4])
                if exp.get("current"):
                    end = datetime.now().year
                else:
                    end = int(str(exp.get("end_date", ""))[:4])
                years = end - start
                if years > 0:
                    years_per_role.append(years)
            except:
                pass
        
        if years_per_role:
            avg_tenure = sum(years_per_role) / len(years_per_role)
            # Consistency: prefer longer tenures (2+ years)
            consistency = min(1.0, avg_tenure / 3.0)
            progression["tenure_consistency"] = consistency
        
        return progression
