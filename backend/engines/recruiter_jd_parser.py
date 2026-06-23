"""
Recruiter-Centric JD Parser
Extracts required, preferred, and negative signals from job descriptions.
"""

from typing import List, Set, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class RequirementProfile:
    """Structured job requirement profile"""
    required_keywords: Set[str]
    preferred_keywords: Set[str]
    negative_signals: Set[str]
    target_experience_min: int  # years
    target_experience_max: int  # years
    location_preferences: List[str]
    relocation_required: bool
    hands_on_coding: bool
    leadership_required: bool


class RecruiterCentricJDParser:
    """
    Parse job descriptions using recruiter domain knowledge.
    Extract structured requirements instead of simple keyword matching.
    """
    
    # Required technical skills for ranking/retrieval roles
    REQUIRED_KEYWORDS = {
        'retrieval systems', 'ranking systems', 'embeddings',
        'vector databases', 'vector db', 'faiss', 'pinecone',
        'milvus', 'weaviate', 'qdrant', 'evaluation frameworks',
        'ndcg', 'mrr', 'map', 'offline evaluation', 'a/b testing',
        'ranking benchmarks'
    }
    
    # Nice-to-have experience
    PREFERRED_KEYWORDS = {
        'startup experience', 'product company', 'recommendation systems',
        'search systems', 'marketplace', 'learning-to-rank', 'ltr',
        'fine-tuning', 'lora', 'qlora', 'llm', 'rag',
        'production ml', 'shipped', 'deployed', 'recruiter workflows',
        'evaluation infrastructure', 'offline benchmarks', 'a/b test',
        'feedback loops'
    }
    
    # Red flags that penalize candidates
    NEGATIVE_SIGNALS = {
        'consulting-only', 'consulting career', 'consulting firms',
        'consulting firms only', 'only worked at consulting',
        'pure research', 'academic research', 'inactive', 'passive',
        'no activity', 'long notice period', '90 days', 'title chasing',
        'framework only', 'no production', 'no shipped', 'research-only',
        'hidden title', 'title chasers'
    }
    
    # Experience expectations
    EXPERIENCE_INDICATORS = {
        'junior': (0, 3),
        'senior': (5, 8),
        'staff': (8, 15),
        'principal': (10, 20),
        'entry-level': (0, 2),
        'mid-level': (3, 6),
        'lead': (5, 10)
    }
    
    def parse_jd(self, jd_text: str) -> RequirementProfile:
        """Parse JD into structured requirements"""
        text_lower = jd_text.lower()
        
        # Extract required keywords
        required = set()
        for keyword in self.REQUIRED_KEYWORDS:
            if keyword in text_lower:
                required.add(keyword)
        
        # Extract preferred keywords
        preferred = set()
        for keyword in self.PREFERRED_KEYWORDS:
            if keyword in text_lower:
                preferred.add(keyword)
        
        # Extract negative signals
        negatives = set()
        for signal in self.NEGATIVE_SIGNALS:
            if signal in text_lower:
                negatives.add(signal)
        
        # Extract experience range
        exp_min, exp_max = self._extract_experience_range(text_lower)
        
        # Extract location preferences
        locations = self._extract_locations(text_lower)
        
        # Detect relocation requirement
        relocation = self._detect_relocation_required(text_lower)
        
        # Detect hands-on coding requirement
        hands_on = self._detect_hands_on_coding(text_lower)
        
        # Detect leadership requirement
        leadership = self._detect_leadership(text_lower)
        
        return RequirementProfile(
            required_keywords=required,
            preferred_keywords=preferred,
            negative_signals=negatives,
            target_experience_min=exp_min,
            target_experience_max=exp_max,
            location_preferences=locations,
            relocation_required=relocation,
            hands_on_coding=hands_on,
            leadership_required=leadership
        )
    
    def _extract_experience_range(self, text: str) -> tuple:
        """Extract target experience range (min, max) in years"""
        # Look for patterns like "5-8 years", "5+ years", "8 years"
        patterns = [
            r'(\d+)\s*[-–]\s*(\d+)\s*years?',  # 5-8 years
            r'(\d+)\+\s*years?',  # 5+ years
            r'at least (\d+) years?',  # at least 5 years
            r'(\d+) years? of experience',  # 5 years of experience
        ]
        
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                matches.append(match.groups())
        
        if not matches:
            return 5, 8  # Default
        
        # Extract numbers
        numbers = []
        for match in matches:
            numbers.extend([int(x) for x in match if x])
        
        if numbers:
            return min(numbers), max(numbers) + 2
        
        return 5, 8
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location preferences"""
        locations = []
        common_locs = [
            'pune', 'noida', 'mumbai', 'hyderabad', 'delhi', 'bangalore',
            'bengaluru', 'india', 'remote', 'hybrid',
            'san francisco', 'sf', 'bay area',
            'new york', 'nyc',
            'seattle',
            'london', 'singapore', 'toronto'
        ]
        for loc in common_locs:
            if loc in text:
                locations.append(loc)
        return locations
    
    def _detect_relocation_required(self, text: str) -> bool:
        """Detect if relocation is required"""
        keywords = [
            'relocation required', 'must relocate', 'willing to relocate',
            'open to relocation', 'open to relocate', 'relocate if needed'
        ]
        return any(kw in text for kw in keywords)
    
    def _detect_hands_on_coding(self, text: str) -> bool:
        """Detect if hands-on coding is required"""
        keywords = [
            'hands-on', 'hands on', 'coding', 'engineering',
            'implementation', 'write code', 'development'
        ]
        return any(kw in text for kw in keywords)
    
    def _detect_leadership(self, text: str) -> bool:
        """Detect if leadership is required"""
        keywords = [
            'lead', 'manage', 'mentor', 'leadership', 'team',
            'principal', 'staff'
        ]
        return any(kw in text for kw in keywords)
