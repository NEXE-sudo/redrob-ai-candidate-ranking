"""Adaptive ranking engine that learns dataset structure."""

from typing import Dict, List, Any, Tuple
import math
from app.engines.dataset_analyzer import DatasetAnalyzer, FieldClassifier


class AdaptiveRankingEngine:
    """Dynamically adapts ranking to any dataset structure."""
    
    def __init__(self):
        """Initialize adaptive ranking engine."""
        self.analyzer = DatasetAnalyzer()
        self.classifier = FieldClassifier()
        self.dataset = []
        self.schema = None
        self.field_mapping = {}
        self.component_weights = {}
        self.scoring_methods = {}
    
    def initialize_with_dataset(
        self,
        dataset: List[Dict[str, Any]],
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Initialize ranking engine with a dataset.
        
        Args:
            dataset: List of records to analyze
            custom_weights: Optional custom weights for components
            
        Returns:
            Schema analysis and initialization details
        """
        self.dataset = dataset
        
        # Analyze dataset structure
        analysis = self.analyzer.analyze_dataset(dataset)
        self.schema = analysis
        self.field_mapping = analysis.get("field_mapping", {})
        
        # Set up scoring components and weights
        self._setup_components(analysis, custom_weights)
        
        return {
            "status": "initialized",
            "total_records": analysis["total_records"],
            "fields_detected": len(analysis["detected_fields"]),
            "components_created": len(self.component_weights),
            "schema_analysis": analysis
        }
    
    def compute_ranking(
        self,
        job_requirements: Dict[str, Any],
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute adaptive ranking for candidate against job.
        
        Args:
            job_requirements: Job description/requirements
            candidate: Candidate profile
            
        Returns:
            Ranking scores with components and explainability
        """
        if not self.schema:
            raise ValueError("Engine not initialized. Call initialize_with_dataset first.")
        
        # Compute component scores
        component_scores = {}
        component_details = {}
        
        for component_id, component_name in enumerate(self.schema["scoring_components_available"], 1):
            comp = component_name
            category = comp["category"]
            fields = comp["fields"]
            method = comp["scoring_method"]
            
            score, details = self._compute_component_score(
                category, fields, method, candidate, job_requirements
            )
            
            component_scores[category] = score
            component_details[category] = details
        
        # Compute weighted final score
        final_score = sum(
            component_scores.get(cat, 0.0) * weight * 100
            for cat, weight in self.component_weights.items()
        )
        final_score = min(100.0, max(0.0, final_score))
        
        # Generate explainability
        explanation = self._generate_explanation(
            candidate, job_requirements, component_scores, component_details
        )
        
        return {
            "candidate_id": candidate.get("id", candidate.get("name", "unknown")),
            "component_scores": component_scores,
            "component_details": component_details,
            "final_score": final_score,
            "explanation": explanation,
            "scoring_breakdown": {
                cat: {
                    "score": component_scores.get(cat, 0.0) * 100,
                    "weight": weight * 100,
                    "contribution": component_scores.get(cat, 0.0) * weight * 100
                }
                for cat, weight in self.component_weights.items()
            }
        }
    
    def rank_candidates(
        self,
        job_requirements: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple candidates for a job.
        
        Args:
            job_requirements: Job requirements
            candidates: List of candidates
            top_k: Return top K candidates (None for all)
            
        Returns:
            Sorted list of rankings
        """
        rankings = []
        for candidate in candidates:
            ranking = self.compute_ranking(job_requirements, candidate)
            rankings.append(ranking)
        
        # Sort by final score descending
        rankings = sorted(rankings, key=lambda x: x["final_score"], reverse=True)
        
        # Add rank numbers
        for i, ranking in enumerate(rankings, 1):
            ranking["rank"] = i
        
        # Return top K if specified
        if top_k:
            rankings = rankings[:top_k]
        
        return rankings
    
    def update_weights(self, new_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Update component weights and recompute rankings.
        
        Args:
            new_weights: New weights for each component
            
        Returns:
            Updated weights and validation
        """
        # Validate weights
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        # Update weights
        self.component_weights.update(new_weights)
        
        return {
            "status": "weights_updated",
            "updated_weights": self.component_weights,
            "total": sum(self.component_weights.values())
        }
    
    def _setup_components(
        self,
        analysis: Dict[str, Any],
        custom_weights: Dict[str, float] = None
    ) -> None:
        """Set up scoring components and weights."""
        
        components = analysis.get("scoring_components_available", [])
        
        # Initialize component weights
        total_weight = 0.0
        default_weights = {}
        
        for comp in components:
            category = comp["category"]
            weight = comp["weight"]
            default_weights[category] = weight
            total_weight += weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            self.component_weights = {
                cat: weight / total_weight
                for cat, weight in default_weights.items()
            }
        else:
            # No components - distribute equally
            n_components = len(components)
            if n_components > 0:
                self.component_weights = {
                    comp["category"]: 1.0 / n_components
                    for comp in components
                }
        
        # Apply custom weights if provided
        if custom_weights:
            for cat, weight in custom_weights.items():
                if cat in self.component_weights:
                    self.component_weights[cat] = weight
            
            # Re-normalize
            total = sum(self.component_weights.values())
            if total > 0:
                self.component_weights = {
                    cat: weight / total
                    for cat, weight in self.component_weights.items()
                }
        
        # Store scoring methods
        for comp in components:
            self.scoring_methods[comp["category"]] = comp["scoring_method"]
    
    def _compute_component_score(
        self,
        category: str,
        fields: List[str],
        method: str,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute score for a single component.
        
        Returns:
            (score: 0-1, details: explanation)
        """
        
        if category == "skills":
            return self._score_skills(fields, candidate, job_requirements, method)
        
        elif category == "experience":
            return self._score_experience(fields, candidate, job_requirements, method)
        
        elif category == "career_metadata":
            return self._score_career_metadata(fields, candidate, job_requirements, method)
        
        elif category == "activity_signals":
            return self._score_activity_signals(fields, candidate, method)
        
        elif category == "behavioral_signals":
            return self._score_behavioral_signals(fields, candidate, method)
        
        else:
            return 0.5, {"method": method, "fields": fields, "note": "unknown category"}
    
    def _score_skills(
        self,
        fields: List[str],
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        method: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score skills component."""
        
        score = 0.0
        details = {
            "method": method,
            "fields_used": fields,
            "extracted_skills": [],
            "matched": [],
            "missing": [],
            "unmatched": []
        }
        
        # Extract candidate skills from all relevant fields
        candidate_skills = set()
        for field in fields:
            if field in candidate:
                field_val = candidate[field]
                if isinstance(field_val, list):
                    candidate_skills.update(str(s).lower() for s in field_val)
                elif isinstance(field_val, str):
                    candidate_skills.update(s.lower() for s in field_val.split(","))
        
        details["extracted_skills"] = list(candidate_skills)
        
        # Get required skills from job
        required_skills = set()
        for key in ["must_have", "skills", "required_skills", "required_technologies"]:
            if key in job_requirements:
                val = job_requirements[key]
                if isinstance(val, list):
                    required_skills.update(str(s).lower() for s in val)
                elif isinstance(val, str):
                    required_skills.update(s.lower() for s in val.split(","))
        
        if required_skills:
            # Calculate matches
            matched = candidate_skills & required_skills
            missing = required_skills - candidate_skills
            unmatched = candidate_skills - required_skills
            
            details["matched"] = list(matched)
            details["missing"] = list(missing)
            details["unmatched"] = list(unmatched)
            
            # Score based on method
            if method == "proficiency_based":
                # If proficiency info available, weight by proficiency
                score = min(1.0, len(matched) / len(required_skills)) if required_skills else 0.0
            else:
                # Presence-based: percentage of required skills matched
                score = len(matched) / len(required_skills) if required_skills else 0.0
                
                # Bonus for having extra skills
                if unmatched:
                    score = min(1.0, score + 0.1)
        
        return score, details
    
    def _score_experience(
        self,
        fields: List[str],
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        method: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score experience component."""
        
        score = 0.0
        details = {
            "method": method,
            "fields_used": fields,
            "total_years": 0,
            "positions_count": 0,
            "seniority_alignment": "unknown"
        }
        
        # Extract years of experience
        total_years = 0
        for field in fields:
            if field in candidate:
                val = candidate[field]
                if isinstance(val, (int, float)):
                    total_years += val
                elif isinstance(val, str):
                    try:
                        total_years += float(val)
                    except:
                        pass
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "years" in item:
                            try:
                                total_years += float(item["years"])
                            except:
                                pass
        
        details["total_years"] = total_years
        
        # Get required seniority
        required_seniority = job_requirements.get("seniority", "mid")
        
        # Score based on method
        if method == "tenure_based":
            if required_seniority == "junior":
                score = min(1.0, total_years / 1.0) if total_years > 0 else 0.2
            elif required_seniority == "mid":
                if 2 <= total_years <= 5:
                    score = 0.9
                elif total_years >= 1:
                    score = 0.6
                elif total_years > 5:
                    score = 0.8
                else:
                    score = 0.2
            elif required_seniority == "senior":
                if total_years >= 5:
                    score = 0.95
                elif total_years >= 3:
                    score = 0.7
                else:
                    score = 0.3
            
            details["seniority_alignment"] = f"Matched {total_years:.1f} years to {required_seniority}"
        
        else:  # role_count_based
            # Count number of roles/positions
            position_count = 0
            for field in fields:
                if field in candidate:
                    val = candidate[field]
                    if isinstance(val, list):
                        position_count += len(val)
            
            details["positions_count"] = position_count
            score = min(1.0, position_count * 0.2)
        
        return score, details
    
    def _score_career_metadata(
        self,
        fields: List[str],
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        method: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score career metadata component."""
        
        score = 0.0
        details = {
            "method": method,
            "fields_used": fields,
            "has_education": False,
            "has_certifications": False,
            "education_count": 0,
            "certification_count": 0
        }
        
        for field in fields:
            if field in candidate:
                val = candidate[field]
                
                # Check for education
                if "education" in field.lower():
                    if val:
                        details["has_education"] = True
                        if isinstance(val, list):
                            details["education_count"] = len(val)
                        else:
                            details["education_count"] = 1
                        score += 0.3
                
                # Check for certifications
                elif "cert" in field.lower():
                    if val:
                        details["has_certifications"] = True
                        if isinstance(val, list):
                            details["certification_count"] = len(val)
                        else:
                            details["certification_count"] = 1
                        score += 0.3
                
                # Check for other metadata
                else:
                    if val:
                        score += 0.2
        
        return min(1.0, score), details
    
    def _score_activity_signals(
        self,
        fields: List[str],
        candidate: Dict[str, Any],
        method: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score activity signals component."""
        
        score = 0.0
        details = {
            "method": method,
            "fields_used": fields,
            "activity_data": {}
        }
        
        for field in fields:
            if field in candidate:
                val = candidate[field]
                details["activity_data"][field] = val
                
                if method == "contribution_based":
                    # Score based on contribution count
                    if isinstance(val, (int, float)):
                        score += min(0.5, val / 100.0)
                    elif isinstance(val, list) and val:
                        score += 0.3
                
                else:  # presence_based
                    if val:
                        score += 0.25
        
        return min(1.0, score), details
    
    def _score_behavioral_signals(
        self,
        fields: List[str],
        candidate: Dict[str, Any],
        method: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score behavioral signals component."""
        
        score = 0.0
        details = {
            "method": method,
            "fields_used": fields,
            "signals": {}
        }
        
        for field in fields:
            if field in candidate:
                val = candidate[field]
                details["signals"][field] = val
                
                if method == "score_based":
                    # Direct score usage
                    if isinstance(val, (int, float)):
                        score += min(0.5, val / 100.0)
                
                else:  # presence_based
                    if val:
                        score += 0.25
        
        return min(1.0, score), details
    
    def _generate_explanation(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        component_scores: Dict[str, float],
        component_details: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate explainable explanation for ranking."""
        
        # Identify strengths
        strengths = []
        weaknesses = []
        
        for category, score in component_scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {category.replace('_', ' ')} match ({score*100:.0f}%)")
            elif score <= 0.3:
                weaknesses.append(f"Weak {category.replace('_', ' ')} match ({score*100:.0f}%)")
        
        # Add field-specific insights
        for category, details in component_details.items():
            if category == "skills" and details.get("matched"):
                strengths.insert(0, f"Matches {len(details['matched'])} required skills")
            
            if category == "skills" and details.get("missing"):
                weaknesses.append(f"Missing {len(details['missing'])} required skills")
            
            if category == "experience":
                years = details.get("total_years", 0)
                if years > 0:
                    strengths.append(f"{years:.1f} years of relevant experience")
        
        return {
            "overall_assessment": "Good match" if len(strengths) >= len(weaknesses) else "Partial match",
            "top_strengths": strengths[:3],
            "main_gaps": weaknesses[:2],
            "component_analysis": {
                cat: {
                    "score": score * 100,
                    "assessment": component_details[cat]
                }
                for cat, score in component_scores.items()
            }
        }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get current schema information."""
        return {
            "schema": self.schema,
            "field_mapping": self.field_mapping,
            "component_weights": self.component_weights,
            "scoring_methods": self.scoring_methods
        }
