# Ranking Engine Implementation Guide

## Overview

A production-style candidate ranking engine that ranks 100K candidates for a Senior AI Engineer role at Redrob AI. The system implements the comprehensive ranking strategy documented in `RANKING_STRATEGY.md`.

## Architecture

### Core Components

#### 1. **Candidate Profile Parser** (`candidate_profile_parser.py`)

Extracts structured information from raw candidate JSON profiles.

**Key Features:**

- Parse candidate profiles into structured `ParsedProfile` objects
- Classify company types (product, consulting, startup, other)
- Analyze career depth and timeline consistency
- Extract relevant skills by category (retrieval, embeddings, ranking, LLM fine-tuning, etc.)
- Detect red flags (keyword stuffing, unrealistic skills, suspicious patterns)

**Red Flag Detection:**

- Keyword stuffing: 50+ skills with <5 average endorsements
- Timeline issues: Overlapping roles, unexplained gaps, very short roles
- Consulting-only careers without product experience
- No GitHub activity or engagement signals
- Stale profiles (not active for 6+ months)

#### 2. **Feature Scorer** (`feature_scorer.py`)

Implements the 5-component scoring system from the ranking strategy.

**Scoring Components (Weighted):**

1. **Technical Relevance (35%)**: Keyword matching (tier-weighted), scale signals, recency
2. **Production Experience (25%)**: Career depth in ML/production roles, consistency checks
3. **Profile Quality (15%)**: Timeline consistency, skill realism, verification signals
4. **Behavioral Engagement (15%)**: Availability, recruiter response rate, GitHub activity
5. **Experience Level Fit (10%)**: Years of experience (target: 5-9 years)

**Disqualifying Factors:**

- Pure research background without production deployment (0.1x multiplier)
- Consulting-only career without product experience (0.1x multiplier)
- No production code in 18+ months (0.3x multiplier)
- Not available/not open to work (0.2x multiplier)

#### 3. **Embedding Retrieval** (`embedding_retrieval.py`)

FAISS-based semantic search with fallback BM25 keyword retrieval.

**EmbeddingRetriever:**

- Uses BAAI/bge-small-en-v1.5 sentence transformer model
- Builds FAISS IndexFlatIP (inner product for cosine similarity)
- Caches embeddings and indices for reuse
- Batch processing for efficiency on 100K+ candidates

**BM25Retriever:**

- Keyword-based retrieval using rank_bm25
- Lightweight and fast pre-filtering
- Can be used as initial filtering stage

#### 4. **Ranking Engine** (`ranking_engine.py`)

Orchestrates the complete ranking pipeline:

**Pipeline Stages:**

1. Load 100K candidates from JSONL
2. Build FAISS index for semantic retrieval
3. Retrieve top 3000 candidates using embeddings
4. Parse and score retrieved candidates
5. Apply disqualifying factors
6. Generate top 100 ranked candidates with explanations
7. Export results as CSV and detailed JSON

**Output:**

- `submission.csv`: Top 100 ranked candidates (candidate_id, rank, score, reasoning)
- `ranking_detailed.json`: Full scores and component breakdown

#### 5. **Main Execution Script** (`run_ranking.py`)

Entry point for the complete ranking pipeline.

**Execution Flow:**

```bash
python3 run_ranking.py
```

## Data Flow

```
Raw JSONL (100K candidates)
           ↓
    [Load Candidates]
           ↓
    [Build FAISS Index + BM25]
           ↓
    [Semantic Retrieval: Top 3000]
           ↓
    [Parse Profiles + Score]
           ↓
    [Apply Disqualifiers]
           ↓
    [Generate Explanations]
           ↓
    [Sort & Select Top 100]
           ↓
    CSV Output + JSON Details
```

## Key Implementation Decisions

### 1. Keyword Tier Weighting

Keywords are categorized by importance:

- **Tier 1 (40% weight)**: embeddings, retrieval, vector DB, FAISS, Pinecone, Milvus, ranking
- **Tier 2 (35% weight)**: LLM, fine-tuning, LoRA, learning-to-rank, evaluation metrics
- **Tier 3 (25% weight)**: Python, ML, AI, NLP, search

### 2. Semantic Retrieval + Feature Scoring

- Initial retrieval via embeddings to capture semantic meaning
- Then apply detailed feature scoring (not just similarity)
- This balances relevance with real-world signal

### 3. Profile Quality Signals

- Verification (email, phone, LinkedIn)
- Timeline consistency checks
- Skill realism assessment
- Prevents gaming via keyword stuffing

### 4. Behavioral Engagement Weight

- Open to work flag
- Recruiter response rate
- Interview completion rate
- GitHub activity
- These indicate actual availability and reliability

### 5. Disqualifying Factors

Rather than gradual scoring, certain patterns immediately downweight candidates:

- Pure research careers
- Consulting-only backgrounds
- Stale code (18+ months)
  This captures the JD's explicit dislikes

## Scoring Logic Details

### Technical Relevance Score

```python
# 1A: Keyword matching (normalized by tier)
keyword_score = (
    0.4 * (tier1_keywords / max_tier1) +
    0.35 * (tier2_keywords / max_tier2) +
    0.25 * (tier3_keywords / max_tier3)
)

# 1B: Scale signals
scale_score = (
    0.4 * (has_production_keywords) +
    0.4 * (has_scale_metrics) +
    0.2 * (company_size >= 500)
)

# 1C: Recency decay
technical = (keyword * 0.7 + scale * 0.3) * (1 - recency_penalty)
```

Recency penalty:

- Current/recent: 0.0
- 0-6 months old: 0.0
- 6-18 months: 0.1
- 18-36 months: 0.2
- 36+ months: 0.3

### Production Experience Score

```python
# Career depth: ML months and production months
deep_exp = (
    (ml_months >= 12) * 0.5 +
    (ml_months >= 24) * 0.3 +
    (production_months >= 12) * 0.2
)

# Consistency: Check for title-chaser pattern
consistency_penalty = (
    (title_jumps_for_seniority) * 0.1 +
    (non_technical_current_but_5yrs_exp) * 0.15
)

production = (deep_exp * 0.8 + 0.2) * (1 - consistency_penalty)
```

### Profile Quality Score

```python
quality = 1.0
quality -= (timeline_issues > 0) * 0.1 * count
quality -= (skill_count > 50) * 0.15  # Likely padding
quality -= (unrealistic_skills) * 0.2
quality -= (low_completeness) * 0.2
quality += (verification_count) * 0.05
```

### Behavioral Engagement Score

```python
score = 1.0
score *= (1.0 if notice_days <= 30 else 0.95 if notice_days <= 60 else...)
score *= (1.02 if response_rate > 0.5 else 1.0)
score *= (1.02 if saved_recruiters > 5 else 1.0)
score *= (1.05 if github_score > 30 else 1.0)
score *= (0.85 if profile_completeness < 60 else 1.0)
```

### Experience Level Fit Score

```python
if exp < 3: return 0.4
elif exp < 5: return 0.7
elif exp <= 9: return 1.0  # Perfect band
elif exp <= 12: return 0.95
elif exp <= 15: return 0.85
else: return 0.7  # Overqualified/stagnant
```

## Running the System

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:

- pandas >= 2.0
- numpy >= 1.24
- torch >= 2.1
- sentence-transformers >= 2.2
- faiss-cpu >= 1.7
- scikit-learn >= 1.3
- rank_bm25 >= 0.2
- tqdm

### Execution

**Option 1: Full ranking (100K candidates)**

```bash
python3 run_ranking.py
```

**Option 2: Component testing**

```bash
python3 test_ranking_components.py
```

### Output Files

Generated in `ranking_output/` directory:

- **submission.csv**: Required format for submission
  - `candidate_id`: CAND_XXXXXXX
  - `rank`: 1-100
  - `score`: Final score (0.0-1.0)
  - `reasoning`: Brief explanation

- **ranking_detailed.json**: Detailed breakdown per candidate
  ```json
  {
    "candidate_id": "CAND_0001234",
    "rank": 1,
    "score": 0.9234,
    "components": {
      "technical_relevance": 0.92,
      "production_experience": 0.96,
      "profile_quality": 0.88,
      "behavioral_engagement": 0.89,
      "experience_level_fit": 0.95,
      "semantic_similarity": 0.75
    },
    "reasoning": {
      "strengths": [...],
      "concerns": [...],
      "key_facts": [...],
      "disqualifiers": [...]
    }
  }
  ```

## Performance Considerations

### Memory Usage

- FAISS index for 100K candidates: ~50 MB (384-dim embeddings)
- Embeddings cache: ~150 MB
- Candidate profiles in memory: ~200 MB (at 5MB per 1K candidates)
- Total: ~500 MB

### Runtime

- Embedding generation: ~5-10 minutes (100K candidates, batch size 64)
- FAISS retrieval: <1 second (top 3000)
- Feature scoring: ~10-15 minutes (3000 candidates)
- Total: ~20-30 minutes for full run

### Optimization Notes

- FAISS uses CPU (can be accelerated with GPU if available)
- Caching allows re-ranking without re-embedding
- Batch processing maintains constant memory
- BM25 prefiltering optional but recommended for very large pools

## Explainability Output

Each ranked candidate includes:

- **Strengths**: 2-3 concrete facts from profile (never hallucinated)
- **Concerns**: Issues or risks identified
- **Key Facts**: Years of experience, education, recent role details
- **Disqualifiers**: Any blocking issues

Example:

```json
{
  "strengths": [
    "7.2 years production ML experience with strong embeddings/retrieval focus",
    "Shipped semantic search system to 10M+ users at Meta",
    "Advanced proficiency in Milvus and FAISS with 40+ endorsements"
  ],
  "concerns": [
    "Currently at TCS (consulting), but 4+ years at product company prior",
    "60-day notice period"
  ],
  "key_facts": [
    "Title: Senior ML Engineer",
    "Education: IIT (Tier 1)",
    "Most recent: Ranking systems at Meta (2+ years)"
  ]
}
```

## Validation

The system includes validation checks:

- ✓ Exactly 100 candidates ranked
- ✓ Scores monotonically decreasing
- ✓ All scores in [0.0, 1.0]
- ✓ All candidate IDs valid (CAND_XXXXXXX)
- ✓ No hallucinated profile data
- ✓ Reasoning matches actual profile fields

## Testing

Run component tests:

```bash
python3 test_ranking_components.py
```

Tests cover:

1. Profile parser (timeline consistency, company classification, red flags)
2. Feature scorer (all 5 components, disqualifiers)
3. BM25 retriever (keyword-based retrieval)
4. FAISS retriever (semantic similarity)

## Future Enhancements

1. **A/B Testing Framework**: Evaluate different weight combinations
2. **Online Learning**: Update weights based on actual hiring outcomes
3. **Explainability Improvements**: Generate more detailed rationale
4. **Multi-language Support**: Handle non-English profiles
5. **Real-time Indexing**: Incrementally add new candidates
6. **Feedback Loop**: Recruiters can rate ranking quality

## References

- JD Analysis: See RANKING_STRATEGY.md for detailed requirements analysis
- Scoring Framework: 5-component design from product strategy
- Technology: FAISS for scalable similarity search, Sentence Transformers for embeddings
- Explainability: Every score traceable to profile data
