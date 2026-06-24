"""Unit tests for HoneypotDetector career inconsistency and expert skill rules.
Run: python3 scripts/test_honeypot_honeypot_detector.py
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


def test_boomerang_career():
    career = [
        {'title': 'Senior Data Scientist', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Search Engineer', 'company': 'B', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'Senior Data Scientist', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is False
    print('test_boomerang_career: PASS')


def test_lateral_ml_ai_move():
    career = [
        {'title': 'AI Engineer', 'company': 'A', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Senior Data Scientist', 'company': 'B', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'ML Engineer', 'company': 'C', 'industry': 'ai', 'company_size': '51-200', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is False
    print('test_lateral_ml_ai_move: PASS')


def test_genuine_inconsistency():
    career = [
        {'title': 'Software Engineer', 'company': 'A', 'industry': 'technology', 'company_size': '51-200', 'start_date': '2024-01-01', 'end_date': '2025-01-01', 'duration_months': 12, 'description': '', 'is_current': False},
        {'title': 'Sales Manager', 'company': 'B', 'industry': 'sales', 'company_size': '201-500', 'start_date': '2022-01-01', 'end_date': '2024-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
        {'title': 'Recruiter', 'company': 'C', 'industry': 'hr', 'company_size': '201-500', 'start_date': '2020-01-01', 'end_date': '2022-01-01', 'duration_months': 24, 'description': '', 'is_current': False},
    ]
    d = HoneypotDetector()
    parsed = make_parsed(career)
    assert d._has_career_inconsistency(parsed) is True
    print('test_genuine_inconsistency: PASS')


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
    print('test_excessive_expert_skills_senior: PASS')


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
    print('test_excessive_expert_skills_junior: PASS')


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
    print('test_no_double_count_expert_signal: PASS')


if __name__ == '__main__':
    test_boomerang_career()
    test_lateral_ml_ai_move()
    test_genuine_inconsistency()
    test_excessive_expert_skills_senior()
    test_excessive_expert_skills_junior()
    test_no_double_count_expert_signal()
    print('All honeypot detector tests passed')
