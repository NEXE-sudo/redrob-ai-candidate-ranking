"""Comprehensive unit tests for HoneypotDetector
Tests all honeypot detection methods including:
- Overlapping employment detection
- Career inconsistency detection
- Expert skills ratio validation
- And other honeypot risk signals

Run: python3 scripts/test_honeypot_detector.py
"""
import os
import sys
import json
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'backend'))

from engines.advanced_scoring_components import HoneypotDetector
from engines.candidate_profile_parser import CandidateProfileParser


def make_parsed(career_history):
    parser = CandidateProfileParser()
    candidate_raw = {
        'candidate_id': 'TEST',
        'profile': {},
        'career_history': career_history,
        'skills': [],
        'redrob_signals': {}
    }
    return parser.parse_candidate(candidate_raw)


# ============== OVERLAPPING EMPLOYMENT TESTS ==============

def case_no_overlap():
    # newest-first order
    career = [
        {'start_date': '2022-01-01', 'end_date': '2024-01-01', 'company': 'A Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Senior Engineer', 'duration_months':24, 'description':'Worked on production systems', 'is_current': False},
        {'start_date': '2019-01-01', 'end_date': '2021-12-31', 'company': 'B Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Engineer', 'duration_months':36, 'description':'Built services', 'is_current': False},
        {'start_date': '2016-01-01', 'end_date': '2018-12-31', 'company': 'C Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Engineer', 'duration_months':36, 'description':'Worked on infra', 'is_current': False},
        {'start_date': '2013-01-01', 'end_date': '2015-12-31', 'company': 'D Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Junior Engineer', 'duration_months':36, 'description':'Entry role', 'is_current': False},
    ]
    p = make_parsed(career)
    d = HoneypotDetector()
    assert d._has_overlapping_employment(p) is False
    print('✓ case_no_overlap: PASS')


def case_with_overlap():
    # newest-first order, introduce overlap between middle roles
    career = [
        {'start_date': '2022-01-01', 'end_date': '2024-01-01', 'company': 'A Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Senior Engineer', 'duration_months':24, 'description':'Worked on production systems', 'is_current': False},
        {'start_date': '2020-01-01', 'end_date': '2023-06-01', 'company': 'B Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Engineer', 'duration_months':42, 'description':'Built services', 'is_current': False},
        {'start_date': '2016-01-01', 'end_date': '2018-12-31', 'company': 'C Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Engineer', 'duration_months':36, 'description':'Worked on infra', 'is_current': False},
        {'start_date': '2013-01-01', 'end_date': '2015-12-31', 'company': 'D Co', 'industry': 'technology', 'company_size': '51-200', 'title': 'Junior Engineer', 'duration_months':36, 'description':'Entry role', 'is_current': False},
    ]
    p = make_parsed(career)
    d = HoneypotDetector()
    assert d._has_overlapping_employment(p) is True
    print('✓ case_with_overlap: PASS')


def case_two_roles():
    career = [
        {'start_date': '2021-01-01', 'end_date': '2022-06-01', 'company': 'X', 'industry': 'technology', 'company_size': '51-200', 'title':'Eng','duration_months':17,'description':'', 'is_current': False},
        {'start_date': '2018-01-01', 'end_date': '2020-12-31', 'company': 'Y', 'industry': 'technology', 'company_size': '51-200', 'title':'Eng','duration_months':36,'description':'', 'is_current': False},
    ]
    p = make_parsed(career)
    d = HoneypotDetector()
    # Per requirement, fewer than 2 valid entries → False
    assert d._has_overlapping_employment(p) is False
    print('✓ case_two_roles: PASS')


def case_with_current_role():
    career = [
        {'start_date': '2023-07-01', 'end_date': None, 'company': 'Current Co', 'industry': 'technology', 'company_size': '51-200', 'title':'Staff','duration_months':12,'description':'Leading projects','is_current': True},  # current job
        {'start_date': '2019-01-01', 'end_date': '2022-12-31', 'company': 'B Co', 'industry': 'technology', 'company_size': '51-200', 'title':'Engineer','duration_months':48,'description':'','is_current': False},
        {'start_date': '2016-01-01', 'end_date': '2018-12-31', 'company': 'C Co', 'industry': 'technology', 'company_size': '51-200', 'title':'Engineer','duration_months':36,'description':'','is_current': False},
        {'start_date': '2013-01-01', 'end_date': '2015-12-31', 'company': 'D Co', 'industry': 'technology', 'company_size': '51-200', 'title':'Junior','duration_months':36,'description':'','is_current': False},
    ]
    p = make_parsed(career)
    d = HoneypotDetector()
    assert d._has_overlapping_employment(p) is False
    print('✓ case_with_current_role: PASS')


# ============== CAREER INCONSISTENCY TESTS ==============

def test_boomerang_career():
    career = [
        {'title': 'Senior Data Scientist', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Search Engineer', 'company': 'B', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'Senior Data Scientist', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is False
    print('✓ test_boomerang_career: PASS')


def test_lateral_ml_ai_move():
    career = [
        {'title': 'AI Engineer', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Senior Data Scientist', 'company': 'B', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'ML Engineer', 'company': 'C', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is False
    print('✓ test_lateral_ml_ai_move: PASS')


def test_genuine_inconsistency():
    career = [
        {'title': 'Software Engineer', 'company': 'A', 'industry': 'technology', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Sales Manager', 'company': 'B', 'industry': 'sales', 'company_size': '201-500', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'Recruiter', 'company': 'C', 'industry': 'hr', 'company_size': '201-500', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is True
    print('✓ test_genuine_inconsistency: PASS')


# ============== EXPERT SKILLS TESTS ==============

def test_excessive_expert_skills_senior():
    skills = [
        {'name': f'Skill{i}', 'proficiency': 'expert', 'duration_months': 24}
        for i in range(9)
    ] + [
        {'name': 'Skill9', 'proficiency': 'advanced', 'duration_months': 24}
    ]
    candidate = {'skills': skills}
    d = HoneypotDetector()
    assert d._has_excessive_expert_skills(candidate, years_experience=16) is False
    print('✓ test_excessive_expert_skills_senior: PASS')


def test_excessive_expert_skills_junior():
    skills = [
        {'name': f'Skill{i}', 'proficiency': 'expert', 'duration_months': 12}
        for i in range(9)
    ] + [
        {'name': 'Skill9', 'proficiency': 'advanced', 'duration_months': 12}
    ]
    candidate = {'skills': skills}
    d = HoneypotDetector()
    assert d._has_excessive_expert_skills(candidate, years_experience=2) is True
    print('✓ test_excessive_expert_skills_junior: PASS')


def test_no_double_count_expert_signal():
    skills = [
        {'name': f'Skill{i}', 'proficiency': 'expert', 'duration_months': 24}
        for i in range(9)
    ] + [
        {'name': 'Skill9', 'proficiency': 'advanced', 'duration_months': 24}
    ]
    candidate = {'skills': skills}
    d = HoneypotDetector()

    suspicious = d._has_suspicious_skills(candidate)
    excessive = d._has_excessive_expert_skills(candidate, years_experience=2)
    assert suspicious is False
    assert excessive is True
    print('✓ test_no_double_count_expert_signal: PASS')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("HONEYPOT DETECTOR - COMPREHENSIVE TEST SUITE")
    print("="*60 + "\n")
    
    print("Overlapping Employment Tests:")
    case_no_overlap()
    case_with_overlap()
    case_two_roles()
    case_with_current_role()
    
    print("\nCareer Inconsistency Tests:")
    test_boomerang_career()
    test_lateral_ml_ai_move()
    test_genuine_inconsistency()
    
    print("\nExpert Skills Tests:")
    test_excessive_expert_skills_senior()
    test_excessive_expert_skills_junior()
    test_no_double_count_expert_signal()
    
    print("\n" + "="*60)
    print("✓ ALL HONEYPOT DETECTOR TESTS PASSED")
    print("="*60 + "\n")
