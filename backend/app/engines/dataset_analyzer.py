"""Engine for analyzing and classifying dataset fields."""

from typing import Dict, List, Any, Set, Tuple
import re
from collections import Counter


class FieldClassifier:
    """Classifies dataset fields into ranking component categories."""
    
    # Keywords for each category
    SKILLS_KEYWORDS = {
        "python", "javascript", "java", "c++", "sql", "html", "css", "react", "django",
        "fastapi", "node", "rust", "golang", "kotlin", "swift", "aws", "azure", "gcp",
        "kubernetes", "docker", "git", "machine learning", "ai", "nlp", "cv", "tensorflow",
        "pytorch", "scikit", "pandas", "numpy", "spark", "hadoop", "tableau", "power bi",
        "excel", "r programming", "scala", "ansible", "terraform", "jenkins", "circleci",
        "salesforce", "sap", "oracle", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
        "agile", "scrum", "jira", "confluence", "linux", "windows", "macos", "devops",
        "microservices", "api", "rest", "graphql", "websocket", "ssl", "oauth", "ldap",
        "junit", "pytest", "mocha", "jasmine", "selenium", "postman", "grpc", "protobuf"
    }
    
    EXPERIENCE_KEYWORDS = {
        "experience", "years", "duration", "tenure", "career", "employment", "job", "position",
        "role", "worked", "worked at", "employed", "company", "organization", "firm",
        "start date", "end date", "from", "to", "joined", "left", "current", "previous",
        "level", "seniority", "junior", "mid", "senior", "lead", "principal", "staff"
    }
    
    CAREER_METADATA_KEYWORDS = {
        "education", "degree", "university", "college", "certification", "certified", "diploma",
        "graduation", "major", "field of study", "school", "institute", "gpa", "score",
        "title", "position", "promotion", "progression", "growth", "leadership", "title history",
        "company size", "industry", "sector", "transition", "move", "change", "previous company"
    }
    
    ACTIVITY_SIGNALS_KEYWORDS = {
        "github", "contribution", "commit", "pull request", "fork", "star", "repository",
        "open source", "project", "portfolio", "blog", "publication", "paper", "patent",
        "conference", "speaking", "talk", "presentation", "workshop", "training", "course",
        "award", "recognition", "achievement", "medal", "trophy", "honor", "distinction"
    }
    
    BEHAVIORAL_SIGNALS_KEYWORDS = {
        "activity", "engagement", "community", "participation", "volunteer", "mentor",
        "collaboration", "teamwork", "communication", "leadership", "initiative", "innovation",
        "problem solving", "analytical", "creative", "adaptable", "reliable", "consistent",
        "contribution frequency", "last active", "participation rate", "engagement score",
        "social", "network", "connections", "followers", "following", "influence"
    }
    
    def classify_field(self, field_name: str, field_type: str = "string", sample_values: List[Any] = None) -> Dict[str, Any]:
        """
        Classify a single field into categories with confidence scores.
        
        Args:
            field_name: Name of the field
            field_type: Data type (string, number, date, boolean, array)
            sample_values: Sample values from the field
            
        Returns:
            Classification with primary category and confidence scores
        """
        field_lower = field_name.lower()
        
        # Initialize scores
        scores = {
            "skills": 0.0,
            "experience": 0.0,
            "career_metadata": 0.0,
            "activity_signals": 0.0,
            "behavioral_signals": 0.0,
            "demographic": 0.0,
            "unknown": 0.0
        }
        
        # Keyword-based classification
        scores["skills"] = self._score_keywords(field_lower, self.SKILLS_KEYWORDS)
        scores["experience"] = self._score_keywords(field_lower, self.EXPERIENCE_KEYWORDS)
        scores["career_metadata"] = self._score_keywords(field_lower, self.CAREER_METADATA_KEYWORDS)
        scores["activity_signals"] = self._score_keywords(field_lower, self.ACTIVITY_SIGNALS_KEYWORDS)
        scores["behavioral_signals"] = self._score_keywords(field_lower, self.BEHAVIORAL_SIGNALS_KEYWORDS)
        
        # Type-based heuristics
        if field_type in ["number", "integer", "float"]:
            if "score" in field_lower or "rating" in field_lower:
                scores["behavioral_signals"] += 0.3
            if "years" in field_lower or "duration" in field_lower:
                scores["experience"] += 0.3
            if "github" in field_lower or "contribution" in field_lower:
                scores["activity_signals"] += 0.3
        
        elif field_type == "date":
            if "start" in field_lower or "join" in field_lower:
                scores["experience"] += 0.4
            if "birth" in field_lower or "dob" in field_lower:
                scores["demographic"] += 0.5
            if "active" in field_lower:
                scores["activity_signals"] += 0.3
        
        elif field_type == "array":
            if "skill" in field_lower:
                scores["skills"] += 0.5
            elif "technology" in field_lower or "tech" in field_lower:
                scores["skills"] += 0.4
            elif "project" in field_lower:
                scores["activity_signals"] += 0.3
        
        elif field_type == "boolean":
            if "active" in field_lower or "current" in field_lower:
                scores["activity_signals"] += 0.3
            if "leadership" in field_lower or "lead" in field_lower:
                scores["behavioral_signals"] += 0.3
        
        # Analyze sample values
        if sample_values:
            scores = self._analyze_sample_values(field_lower, field_type, sample_values, scores)
        
        # Determine primary classification
        primary_category = max(scores, key=scores.get)
        if scores[primary_category] < 0.1:
            primary_category = "unknown"
        
        # Normalize scores to probabilities
        total = sum(max(0, s) for s in scores.values())
        if total > 0:
            scores = {k: max(0, v) / total for k, v in scores.items()}
        else:
            scores = {k: 0.0 for k in scores}
        
        return {
            "field_name": field_name,
            "field_type": field_type,
            "primary_category": primary_category,
            "category_scores": scores,
            "confidence": scores.get(primary_category, 0.0)
        }
    
    def _score_keywords(self, field_name: str, keywords: Set[str]) -> float:
        """Score field name against keyword set."""
        words = set(re.findall(r'\w+', field_name.lower()))
        matches = len(words & keywords)
        return min(1.0, matches * 0.5)
    
    def _analyze_sample_values(
        self,
        field_lower: str,
        field_type: str,
        sample_values: List[Any],
        scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze sample values to refine classification."""
        
        if not sample_values:
            return scores
        
        # Filter out None values
        values = [v for v in sample_values if v is not None]
        if not values:
            return scores
        
        # Check for URLs (portfolio, GitHub)
        if any("github" in str(v).lower() for v in values if isinstance(v, str)):
            scores["activity_signals"] += 0.3
        
        # Check for email patterns
        if any("@" in str(v) for v in values if isinstance(v, str)):
            scores["demographic"] += 0.2
        
        # Check for numeric values
        try:
            numeric_values = [float(v) for v in values if v and isinstance(v, (int, float, str))]
            if numeric_values:
                avg_val = sum(numeric_values) / len(numeric_values)
                if 0 <= avg_val <= 100:
                    scores["behavioral_signals"] += 0.2
                if 0 <= avg_val <= 60:
                    scores["experience"] += 0.2
        except (ValueError, TypeError):
            pass
        
        # Check for list/array patterns
        if isinstance(values[0], list):
            scores["skills"] += 0.2
            scores["activity_signals"] += 0.1
        
        return scores


class DatasetAnalyzer:
    """Analyzes dataset structure for adaptive ranking."""
    
    def __init__(self):
        """Initialize dataset analyzer."""
        self.classifier = FieldClassifier()
        self.detected_fields: List[Dict[str, Any]] = []
        self.field_mapping: Dict[str, str] = {}  # Field name -> category mapping
    
    def analyze_dataset(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze dataset structure and generate field classifications.
        
        Args:
            dataset: List of dictionaries representing records
            
        Returns:
            Analysis with detected fields, classifications, and suggestions
        """
        if not dataset:
            return {
                "total_records": 0,
                "detected_fields": [],
                "field_classifications": [],
                "missing_categories": list(self._get_all_categories()),
                "confidence": 0.0
            }
        
        # Detect all fields
        first_record = dataset[0]
        detected_fields = list(first_record.keys())
        
        # Classify each field
        classifications = []
        for field_name in detected_fields:
            field_type = self._infer_field_type(dataset, field_name)
            sample_values = [r.get(field_name) for r in dataset[:10]]
            
            classification = self.classifier.classify_field(
                field_name,
                field_type,
                sample_values
            )
            classifications.append(classification)
            self.field_mapping[field_name] = classification["primary_category"]
        
        # Determine coverage
        categories_found = set(c["primary_category"] for c in classifications if c["primary_category"] != "unknown")
        all_categories = self._get_all_categories()
        missing_categories = all_categories - categories_found
        coverage = len(categories_found) / len(all_categories)
        
        # Calculate overall confidence
        avg_confidence = sum(c["confidence"] for c in classifications) / len(classifications) if classifications else 0.0
        
        return {
            "total_records": len(dataset),
            "detected_fields": detected_fields,
            "field_classifications": classifications,
            "field_mapping": self.field_mapping,
            "categories_found": sorted(categories_found),
            "missing_categories": sorted(missing_categories),
            "coverage": coverage,
            "average_confidence": avg_confidence,
            "scoring_components_available": self._generate_scoring_components(classifications),
            "recommendations": self._generate_recommendations(classifications)
        }
    
    def _infer_field_type(self, dataset: List[Dict[str, Any]], field_name: str) -> str:
        """Infer data type of a field."""
        
        sample_values = [r.get(field_name) for r in dataset[:20] if field_name in r and r[field_name] is not None]
        
        if not sample_values:
            return "unknown"
        
        first_val = sample_values[0]
        
        if isinstance(first_val, bool):
            return "boolean"
        elif isinstance(first_val, int):
            return "integer"
        elif isinstance(first_val, float):
            return "float"
        elif isinstance(first_val, list):
            return "array"
        elif isinstance(first_val, dict):
            return "object"
        else:
            # Check if string contains date pattern
            str_val = str(first_val).lower()
            if any(pattern in str_val for pattern in ["202", "201", "19", "-", "/"]):
                return "date"
            if "." in str_val and len(str_val) < 20:
                try:
                    float(str_val)
                    return "float"
                except:
                    pass
            return "string"
    
    def _get_all_categories(self) -> Set[str]:
        """Get all available scoring categories."""
        return {
            "skills",
            "experience",
            "career_metadata",
            "activity_signals",
            "behavioral_signals"
        }
    
    def _generate_scoring_components(
        self,
        classifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate scoring components based on detected fields."""
        
        components = []
        
        # Group by category
        by_category = {}
        for clf in classifications:
            cat = clf["primary_category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(clf["field_name"])
        
        # Create components
        component_id = 1
        for category, fields in sorted(by_category.items()):
            if category != "unknown":
                components.append({
                    "id": component_id,
                    "name": category.replace("_", " ").title(),
                    "category": category,
                    "weight": self._default_weight(category),
                    "fields": fields,
                    "scoring_method": self._suggest_scoring_method(category, fields),
                    "description": self._get_component_description(category)
                })
                component_id += 1
        
        return components
    
    def _default_weight(self, category: str) -> float:
        """Get default weight for a category."""
        defaults = {
            "skills": 0.35,
            "experience": 0.20,
            "career_metadata": 0.15,
            "activity_signals": 0.20,
            "behavioral_signals": 0.10
        }
        return defaults.get(category, 0.1)
    
    def _suggest_scoring_method(self, category: str, fields: List[str]) -> str:
        """Suggest scoring method for a category."""
        
        field_str = " ".join(fields).lower()
        
        if category == "skills":
            if "proficiency" in field_str or "level" in field_str:
                return "proficiency_based"
            else:
                return "presence_based"
        
        elif category == "experience":
            if "years" in field_str or "duration" in field_str:
                return "tenure_based"
            else:
                return "role_count_based"
        
        elif category == "activity_signals":
            if "contribution" in field_str or "commit" in field_str:
                return "contribution_based"
            else:
                return "presence_based"
        
        elif category == "behavioral_signals":
            if "score" in field_str or "rating" in field_str:
                return "score_based"
            else:
                return "presence_based"
        
        else:
            return "presence_based"
    
    def _get_component_description(self, category: str) -> str:
        """Get description for component category."""
        descriptions = {
            "skills": "Technical skills match based on required and candidate skills",
            "experience": "Professional experience alignment with job requirements",
            "career_metadata": "Career background including education and certifications",
            "activity_signals": "Open source, portfolio, and community contributions",
            "behavioral_signals": "Engagement, consistency, and behavioral indicators"
        }
        return descriptions.get(category, "Component score")
    
    def _generate_recommendations(self, classifications: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving dataset coverage."""
        
        recommendations = []
        
        # Check for skills data
        if not any("skills" in c.get("field_name", "").lower() for c in classifications):
            recommendations.append(
                "No skills field detected. Consider adding a 'skills' column with a list of technologies/languages."
            )
        
        # Check for experience data
        if not any("experience" in c.get("field_name", "").lower() or "years" in c.get("field_name", "").lower() for c in classifications):
            recommendations.append(
                "No experience field detected. Consider adding 'years_experience' or 'experience' column."
            )
        
        # Check for activity data
        if not any("github" in c.get("field_name", "").lower() or "contribution" in c.get("field_name", "").lower() for c in classifications):
            recommendations.append(
                "No activity signals detected. Consider adding 'github_repos', 'contributions', or 'portfolio_link'."
            )
        
        # Check for behavioral data
        if not any("behavior" in c.get("field_name", "").lower() or "engagement" in c.get("field_name", "").lower() for c in classifications):
            recommendations.append(
                "No behavioral signals detected. Consider adding engagement metrics or activity scores."
            )
        
        return recommendations
