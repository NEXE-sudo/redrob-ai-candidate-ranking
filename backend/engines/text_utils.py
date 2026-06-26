"""
Shared Text Utilities Module
Centralized text processing functions used across multiple engines
"""

import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent keyword matching.
    Removes special characters, converts to lowercase, normalizes whitespace.
    
    Used in: feature_scorer.py, recruiter_jd_parser.py, candidate_profile_parser.py
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized lowercase text with special characters removed
    """
    text = str(text or '').lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """
    Normalize job titles for robust matching.
    Removes special characters and normalizes whitespace.
    
    Used in: feature_scorer.py, candidate_profile_parser.py
    
    Args:
        title: Raw job title
        
    Returns:
        Normalized title
    """
    title = str(title or '').lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_keywords_from_text(text: str, keyword_list: List[str], use_regex: bool = False) -> List[str]:
    """
    Extract keywords from text using substring or regex matching.
    
    Args:
        text: Text to search in
        keyword_list: List of keywords to find
        use_regex: If True, treat keywords as regex patterns
        
    Returns:
        List of found keywords in order of appearance
    """
    text_lower = text.lower()
    found = []
    
    for keyword in keyword_list:
        if use_regex:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                found.append(keyword)
        else:
            if keyword in text_lower:
                found.append(keyword)
    
    return list(dict.fromkeys(found))  # Remove duplicates while preserving order


def count_keyword_matches(text: str, keywords: set) -> int:
    """
    Count how many keywords appear in text.
    
    Args:
        text: Text to search in
        keywords: Set of keywords
        
    Returns:
        Count of keywords found in text
    """
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword in text_lower)


def extract_text_fields(candidate_raw: dict) -> str:
    """
    Extract and concatenate all text fields from a candidate record.
    
    Consolidates text extraction used in multiple places:
    - feature_scorer.py: _collect_candidate_text()
    - advanced_scorer.py: score_evaluation_framework()
    - optimized_ranking_engine.py: various scoring methods
    
    Args:
        candidate_raw: Candidate data dictionary
        
    Returns:
        Concatenated normalized text from all candidate fields
    """
    text_parts = []
    
    # Profile text
    profile = candidate_raw.get('profile', {})
    text_parts.append(profile.get('summary', '').lower())
    text_parts.append(profile.get('headline', '').lower())
    text_parts.append(profile.get('current_title', '').lower())
    text_parts.append(profile.get('location', '').lower())
    
    # Career history
    for role in candidate_raw.get('career_history', []):
        text_parts.append(role.get('title', '').lower())
        text_parts.append(role.get('description', '').lower())
        text_parts.append(role.get('company', '').lower())
    
    # Skills
    for skill in candidate_raw.get('skills', []):
        text_parts.append(skill.get('name', '').lower())
    
    # Certifications
    for cert in candidate_raw.get('certifications', []):
        text_parts.append(cert.get('name', '').lower())
    
    # Languages
    for lang in candidate_raw.get('languages', []):
        text_parts.append(lang.get('language', '').lower())
    
    all_text = ' '.join([t for t in text_parts if t])
    return all_text
