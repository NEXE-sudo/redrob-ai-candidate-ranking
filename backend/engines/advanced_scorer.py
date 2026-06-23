"""
Advanced Scoring Components
Evaluation Framework Score and Startup/Product Mindset Score
"""

from typing import Dict, Any, List
from .candidate_profile_parser import ParsedProfile, CandidateProfileParser


class AdvancedScorer:
    """Additional scoring dimensions for ranking quality"""
    
    # Evaluation framework keywords
    EVAL_KEYWORDS = {
        'ndcg', 'mrr', 'map', 'mean average precision', 'discounted cumulative gain',
        'mean reciprocal rank', 'offline evaluation', 'online evaluation', 'a/b test',
        'a/b testing', 'correlation', 'offline-online', 'evaluation framework',
        'ranking metric', 'relevance metric', 'test harness', 'benchmark'
    }
    
    # A/B testing keywords
    AB_TESTING_KEYWORDS = {
        'a/b test', 'ab test', 'split test', 'hypothesis test', 'control group',
        'experiment', 'treatment', 'statistical significance', 'p-value', 'power analysis',
        'online experiment'
    }
    
    # Startup/product company keywords
    STARTUP_KEYWORDS = {
        'startup', 'early stage', 'seed', 'series a', 'series b', 'founding',
        'founder', 'pre-seed', 'ycombinator', 'y combinator'
    }
    
    PRODUCT_OWNERSHIP_KEYWORDS = {
        'owned', 'shipped', 'launched', 'built', 'led', 'drove', 'product',
        'end-to-end', 'full stack', 'startup', 'mvp', 'product launch'
    }
    
    SMALL_COMPANY_SIZES = ['1-10', '11-50', '51-200']
    
    def __init__(self, parser: CandidateProfileParser = None):
        self.parser = parser or CandidateProfileParser()
    
    def score_evaluation_framework(
        self,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Score experience with evaluation frameworks
        
        Key signals:
        - Mentions of NDCG, MRR, MAP metrics
        - A/B testing experience
        - Offline-online correlation work
        - Ranking evaluation systems
        
        Returns:
            Score 0.0-1.0
        """
        
        score = 0.0
        
        # Extract all text
        all_text = (
            candidate_raw['profile'].get('summary', '').lower() + ' ' +
            candidate_raw['profile'].get('headline', '').lower()
        )
        
        for role in candidate_raw.get('career_history', []):
            all_text += ' ' + role.get('title', '').lower() + ' ' + role.get('description', '').lower()
        
        # Check for evaluation keywords
        eval_count = sum(1 for keyword in self.EVAL_KEYWORDS if keyword in all_text)
        if eval_count > 0:
            score += min(eval_count / 3, 0.4)  # Cap at 0.4 for basic mentions
        
        # Check for A/B testing
        ab_count = sum(1 for keyword in self.AB_TESTING_KEYWORDS if keyword in all_text)
        if ab_count > 0:
            score += 0.3
        
        # Check for NDCG/MRR/MAP specifically (strongest signal)
        if any(metric in all_text for metric in ['ndcg', 'mrr', 'map']):
            score += 0.3
        
        # Check for explicit ranking evaluation work
        if 'ranking' in all_text and 'evaluat' in all_text:
            score += 0.2
        
        # Check for offline-online correlation (very strong signal)
        if 'offline' in all_text and 'online' in all_text:
            score += 0.1
        
        return min(score, 1.0)
    
    def score_startup_product_mindset(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> float:
        """Score startup/product company experience and ownership signals
        
        Key signals:
        - Current/recent startup experience
        - Product ownership language
        - Small company exposure
        - MVP/launch experience
        - Multiple startups
        
        Returns:
            Score 0.0-1.0
        """
        
        score = 0.0
        
        # Check for startup keywords in profile
        profile_text = (
            candidate_raw['profile'].get('summary', '').lower() + ' ' +
            candidate_raw['profile'].get('headline', '').lower()
        )
        
        startup_mentions = sum(1 for keyword in self.STARTUP_KEYWORDS if keyword in profile_text)
        if startup_mentions > 0:
            score += 0.1
        
        # Analyze career history for startup/product signals
        startup_count = 0
        product_company_count = 0
        small_company_count = 0
        product_ownership_roles = 0
        
        for role in candidate_raw.get('career_history', []):
            title = role.get('title', '').lower()
            company = role.get('company', '').lower()
            description = role.get('description', '').lower()
            company_size = role.get('company_size', '')
            
            # Check for startup keywords
            if any(keyword in company or keyword in title 
                   for keyword in self.STARTUP_KEYWORDS):
                startup_count += 1
            
            # Check for small company size
            if company_size in self.SMALL_COMPANY_SIZES:
                small_company_count += 1
            
            # Check for product company signals
            if parsed_profile.company_type == 'product':
                product_company_count += 1
            
            # Check for product ownership language
            if any(keyword in description for keyword in self.PRODUCT_OWNERSHIP_KEYWORDS):
                product_ownership_roles += 1
                score += 0.15
        
        # Bonus for multiple startup stints
        if startup_count >= 2:
            score += 0.2
        elif startup_count == 1:
            score += 0.15
        
        # Bonus for small company experience (implies shipping mindset)
        if small_company_count >= 3:
            score += 0.25
        elif small_company_count >= 2:
            score += 0.15
        elif small_company_count >= 1:
            score += 0.05
        
        # Bonus for current startup role
        career_history = candidate_raw.get('career_history', [])
        if career_history:
            current_role = career_history[0]
            if current_role.get('is_current') and any(
                keyword in current_role.get('company', '').lower() 
                for keyword in self.STARTUP_KEYWORDS
            ):
                score += 0.2
        
        # Penalty for consulting background (reduces score)
        if parsed_profile.is_consulting_only:
            score = max(0, score - 0.3)
        
        # Bonus for product company without consulting
        if product_company_count >= 2 and not parsed_profile.is_consulting_only:
            score += 0.15
        
        return min(score, 1.0)
    
    def score_evaluation_experience(
        self,
        candidate_raw: Dict[str, Any]
    ) -> float:
        """Alias for score_evaluation_framework for consistency"""
        return self.score_evaluation_framework(candidate_raw)
    
    def score_product_mindset(
        self,
        candidate_raw: Dict[str, Any],
        parsed_profile: ParsedProfile
    ) -> float:
        """Alias for score_startup_product_mindset for consistency"""
        return self.score_startup_product_mindset(candidate_raw, parsed_profile)
