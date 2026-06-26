#!/usr/bin/env python3
"""
Test script for ranking engine
Tests components with a small sample before full run
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from engines.candidate_profile_parser import CandidateProfileParser
from engines.feature_scorer import FeatureScorer
from engines.embedding_retrieval import BM25Retriever
from engines.recruiter_jd_parser import RequirementProfile


def load_sample_candidates(num_samples=10):
    """Load first N candidates from JSONL for testing"""
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
    candidates_file = data_dir / "candidates.jsonl"
    
    candidates = []
    with open(candidates_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            candidates.append(json.loads(line))
    
    return candidates


def test_profile_parser():
    print("\n" + "="*60)
    print("TEST 1: Profile Parser")
    print("="*60)
    
    candidates = load_sample_candidates(5)
    parser = CandidateProfileParser()
    
    for candidate in candidates:
        cid = candidate['candidate_id']
        print(f"\nParsing {cid}...")
        
        try:
            parsed = parser.parse_candidate(candidate)
            print(f"  ✓ Years exp: {parsed.years_experience}")
            print(f"  ✓ Company type: {parsed.company_type}")
            print(f"  ✓ Consulting only: {parsed.is_consulting_only}")
            print(f"  ✓ Skills: {parsed.skill_counts}")
            print(f"  ✓ Certifications: {parsed.certifications_count}")
            print(f"  ✓ Languages: {parsed.languages_count}")
            print(f"  ✓ Timeline issues: {len(parsed.timeline_issues)}")
            
            # Red flags
            red_flags = parser.detect_red_flags(parsed, candidate)
            flags_set = [k for k, v in red_flags.items() if v]
            if flags_set:
                print(f"  ⚠  Red flags: {', '.join(flags_set)}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n✓ Profile Parser Tests Passed")
    return True


def test_feature_scorer():
    print("\n" + "="*60)
    print("TEST 2: Feature Scorer")
    print("="*60)
    
    candidates = load_sample_candidates(3)
    parser = CandidateProfileParser()
    scorer = FeatureScorer(parser)
    
    for candidate in candidates:
        cid = candidate['candidate_id']
        print(f"\nScoring {cid}...")
        
        try:
            parsed = parser.parse_candidate(candidate)
            components = scorer.score_candidate(
                candidate,
                parsed,
                semantic_similarity=0.5
            )
            
            print(f"  ✓ Technical Relevance: {components.technical_relevance:.3f}")
            print(f"  ✓ Production Experience: {components.production_experience:.3f}")
            print(f"  ✓ Profile Quality: {components.profile_quality:.3f}")
            print(f"  ✓ Behavioral Engagement: {components.behavioral_engagement:.3f}")
            print(f"  ✓ Experience Level Fit: {components.experience_level_fit:.3f}")
            print(f"  ✓ Final Score: {components.final_score:.3f}")
            
            # Apply disqualifiers
            final_with_disqualifiers = scorer.apply_disqualifying_factors(
                components.final_score,
                parsed,
                candidate
            )
            print(f"  ✓ After disqualifiers: {final_with_disqualifiers:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✓ Feature Scorer Tests Passed")
    return True


def test_feature_scorer_with_location_preferences():
    print("\n" + "="*60)
    print("TEST 2.1: Feature Scorer Location Preference")
    print("="*60)

    candidate = {
        'candidate_id': 'CAND_TEST',
        'profile': {
            'current_title': 'Senior AI Engineer',
            'current_company': 'Acme Labs',
            'headline': 'Senior AI Engineer',
            'summary': 'Built production machine learning systems.',
            'location': 'Pune',
            'country': 'India',
            'years_of_experience': 6
        },
        'career_history': [
            {
                'company': 'Acme Labs',
                'title': 'Senior AI Engineer',
                'start_date': '2021-01-01',
                'end_date': None,
                'duration_months': 42,
                'is_current': True,
                'industry': 'Technology',
                'company_size': '1001-5000',
                'description': 'Built production ML systems for customer-facing products.'
            }
        ],
        'skills': [
            {'name': 'Python', 'proficiency': 'expert', 'endorsements': 12, 'duration_months': 36}
        ],
        'education': [
            {'institution': 'Tech University', 'degree': 'B.E.', 'field_of_study': 'Computer Science', 'start_year': 2012, 'end_year': 2016, 'grade': '8.5 CGPA', 'tier': 'tier_2'}
        ],
        'certifications': [],
        'languages': [{'language': 'English', 'proficiency': 'professional'}],
        'redrob_signals': {
            'profile_completeness_score': 85,
            'interview_completion_rate': 0.95,
            'offer_acceptance_rate': 0.8,
            'avg_response_time_hours': 12,
            'recruiter_response_rate': 0.7,
            'open_to_work_flag': True,
            'github_activity_score': 75,
            'willing_to_relocate': True
        }
    }

    parser = CandidateProfileParser()
    scorer = FeatureScorer(parser)
    parsed = parser.parse_candidate(candidate)
    requirement_profile = RequirementProfile(
        required_keywords=set(),
        preferred_keywords=set(),
        negative_signals=set(),
        target_experience_min=5,
        target_experience_max=8,
        location_preferences=['pune'],
        relocation_required=False,
        hands_on_coding=False,
        leadership_required=False
    )

    try:
        components = scorer.score_candidate(
            candidate,
            parsed,
            semantic_similarity=0.5,
            requirement_profile=requirement_profile
        )
        print(f"  ✓ Profile Quality: {components.profile_quality_multiplier:.3f}")
        print(f"  ✓ Final Score: {components.final_score:.3f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    print("\n✓ Feature Scorer Location Preference Test Passed")
    return True


def test_bm25_retriever():
    print("\n" + "="*60)
    print("TEST 3: BM25 Retriever")
    print("="*60)
    
    candidates = load_sample_candidates(100)
    retriever = BM25Retriever()
    
    print(f"\nBuilding BM25 index for {len(candidates)} candidates...")
    try:
        retriever.build_index(candidates)
        print(f"  ✓ Index built successfully")
        
        # Test retrieval
        query = "Senior AI Engineer embeddings retrieval ranking"
        retrieved_ids, scores = retriever.retrieve(query, top_k=10)
        
        print(f"\nRetrieving top 10 for query: '{query}'")
        for cid, score in zip(retrieved_ids[:5], scores[:5]):
            print(f"  {cid}: {score:.3f}")
        
        print(f"  ✓ Retrieved {len(retrieved_ids)} candidates")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ BM25 Retriever Tests Passed")
    return True


def main():
    print("\n" + "="*60)
    print("RANKING ENGINE - COMPONENT TESTS")
    print("="*60)
    
    tests = [
        test_profile_parser,
        test_feature_scorer,
        test_bm25_retriever,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED - Ready for full ranking")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Fix issues before full ranking")
        return 1


if __name__ == '__main__':
    sys.exit(main())
