# Redrob AI Candidate Ranking Pipeline

## Overview

A highly optimized, multi-stage Python ranking pipeline built for the Redrob AI Candidate Ranking challenge. This system is designed to evaluate 100,000+ candidates against a specific job description with sub-3-minute execution times on a standard CPU.

It employs a cascaded retrieval and reranking approach:
1. **BM25 Lexical Retrieval:** Fast initial filtering down to the top 5,000 candidates.
2. **FAISS Semantic Search:** Deep understanding using precomputed Sentence-Transformer embeddings (`all-mpnet-base-v2`), narrowing down to the top 1,000.
3. **Cross-Encoder Reranking:** Precision reranking using MS-MARCO MiniLM.
4. **Feature Scoring Hub:** A heuristic engine that integrates Redrob behavioral signals, parsed JD requirements, skill assessments, and career trajectories (accounting for consulting vs. product vs. startup backgrounds).
5. **Deterministic Tie-Breaking:** Strict compliance with challenge validator tie-break rules.

---

## Project Structure

```text
.
├── backend/
│   ├── rank.py                        # Main ranking pipeline entry point
│   ├── analyze_results.py             # Validation and analysis script
│   ├── requirements.txt               # Python dependencies
│   ├── models/                        # Local model cache
│   └── engines/
│       ├── candidate_profile_parser.py # Extracts structured candidate data & consulting signals
│       ├── feature_scorer.py           # Multi-factor scoring hub and behavioral modifiers
│       ├── optimized_ranking_engine.py # Core pipeline orchestrator
│       └── recruiter_jd_parser.py      # Parses job requirements and keywords
├── scripts/
│   └── test_stability_patch.py        # Unit tests covering scoring edge cases & tie-breakers
├── run_full_pipeline.sh               # E2E execution script
├── submission_metadata.yaml           # Hackathon submission metadata
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- 16GB+ RAM recommended (for fast FAISS/CrossEncoder processing)

### Installation

1. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

---

## Usage

### Run the Full Pipeline

The easiest way to execute the ranking system end-to-end is via the provided shell script:

```bash
bash run_full_pipeline.sh
```

By default, this looks for candidates in the challenge dataset path (`[PUB] India_runs_data_and_ai_challenge/.../candidates.jsonl`). 

You can also override the defaults:
```bash
bash run_full_pipeline.sh path/to/custom_candidates.jsonl path/to/output_dir
```

### Run Python Script Directly

To run the ranking engine manually with more granularity:

```bash
python backend/rank.py \
  --candidates ./candidates.jsonl \
  --out ranking_output/submission.csv
```

---

## Core Engines

### 1. Requirement Understanding Engine

**File**: `backend/engines/recruiter_jd_parser.py`

**What it does:**
- Parses job descriptions and metadata.
- Extracts must-have and good-to-have skills.
- Detects role seniority and required experience range.
- Identifies if hands-on coding or leadership is required.

### 2. Candidate Intelligence Engine

**File**: `backend/engines/candidate_profile_parser.py`

**What it does:**
- Analyzes candidate career progression and tenure.
- Extracts explicit skills and normalizes them for matching.
- Evaluates company history to flag "Consulting-Only" backgrounds vs. "Product Company" experience.
- Normalizes behavioral signals (e.g., notice period, GitHub activity).

### 3. Embedding & Retrieval Engine

**File**: `backend/engines/optimized_ranking_engine.py`

**What it does:**
- Uses BM25 for rapid top-K lexical filtering (Top 5,000).
- Uses Sentence Transformers (`all-mpnet-base-v2`) to generate 768-dim embeddings.
- Maintains and builds FAISS indices on the fly.
- Applies semantic similarity search to retrieve the Top 1,000 candidates.
- Performs Cross-Encoder reranking using `ms-marco-MiniLM-L-12-v2` for precise semantic alignment.

### 4. Ranking Engine (Feature Scoring Hub)

**File**: `backend/engines/feature_scorer.py`

**What it does:**
- Computes comprehensive, multi-component scoring systems.
- Calculates a weighted final score.
- Implements explicit reasoning capture and generates explainability data.
- Handles behavioral-signal modifiers and explicit penalties for gaming/keyword-stuffing.

**Scoring Strategy:**
```
Final Score Components:
    - title_relevance
    - skill_trust_score
    - assessment_score
    - semantic_similarity
    - technical_relevance
    - production_experience
    - experience_level_fit
    - education_score
    - evaluation_framework_score
    - product_mindset_score
    - product_fit_score
    - career_trajectory_score
```
*Note: The exact weights are scaled and modified multiplicatively based on behavioral adjustments (e.g. recruiter response rates, notice periods).*

---

## Explainability System

Every candidate ranking generates transparent reasoning, output primarily to `ranking_detailed.json` and summarized in `submission.csv`:

### 1. Score Breakdown
- Multi-component score logging.
- Weighted calculation transparency.

### 2. Skill Analysis
- Matched skills verification.
- Trust multiplier analysis (penalizes lazy keyword stuffing).

### 3. Career Assessment
- Experience alignment (seniority vs required range).
- Company pedigree adjustments (startups/product companies vs pure consulting).

---

## Outputs

After execution, the results will be placed in the `ranking_output/` directory:

1. **`submission.csv`**: The primary output required by the challenge validator. Contains `candidate_id`, `rank`, `score` (rounded to 4 decimals), and a generated `reasoning` string.
2. **`ranking_detailed.json`**: An in-depth dump of the top 100 candidates, including their raw component scores, unrounded `final_score`, and full extracted profiles. Useful for debugging and transparency.

---

## Testing & Validation

The project includes unit tests to ensure ranking logic robustness (e.g., preventing score explosions, testing edge cases like consulting penalties, handling duplicate response bonuses, and enforcing deterministic tie-breaks):

```bash
python scripts/test_stability_patch.py
```

To validate the `submission.csv` against the challenge specifications:

```bash
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" ranking_output/submission.csv
```
