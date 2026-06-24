"""Quick unit-like test for HoneypotDetector._has_overlapping_employment
Run: python3 scripts/test_honeypot_detector.py
"""
import os
import sys
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
    print('case_no_overlap: PASS')


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
    print('case_with_overlap: PASS')


def case_two_roles():
    career = [
        {'start_date': '2021-01-01', 'end_date': '2022-06-01', 'company': 'X', 'industry': 'technology', 'company_size': '51-200', 'title':'Eng','duration_months':17,'description':'', 'is_current': False},
        {'start_date': '2018-01-01', 'end_date': '2020-12-31', 'company': 'Y', 'industry': 'technology', 'company_size': '51-200', 'title':'Eng','duration_months':36,'description':'', 'is_current': False},
    ]
    p = make_parsed(career)
    d = HoneypotDetector()
    # Per requirement, fewer than 2 valid entries → False
    assert d._has_overlapping_employment(p) is False
    print('case_two_roles: PASS')


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
    print('case_with_current_role: PASS')


if __name__ == '__main__':
    case_no_overlap()
    case_with_overlap()
    case_two_roles()
    case_with_current_role()
    print('All honeypot tests passed')
