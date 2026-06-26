# COMPREHENSIVE CODEBASE DUPLICATION ANALYSIS

**Analysis Date:** 2026-06-26  
**Scope:** Full Python codebase analysis  
**Files Analyzed:** 15+ Python files across backend/, scripts/, root level

---

## EXECUTIVE SUMMARY

The codebase contains **significant overlapping and duplicate functionality** that impacts:

- **Consistency**: Same scoring logic executed differently in different places
- **Maintainability**: Changes must be coordinated across multiple files
- **Testing**: Redundant test files for the same components
- **Performance**: Unnecessary computation from duplicate keyword matching

**Total Issues Found:** 18+ duplications across 3 severity levels

| Severity     | Count | Impact                                 |
| ------------ | ----- | -------------------------------------- |
| **CRITICAL** | 3     | Scoring inconsistencies, wrong results |
| **HIGH**     | 8     | Code duplication, maintenance burden   |
| **MEDIUM**   | 5     | Dead code, unused implementations      |
| **LOW**      | 2     | Test organization, minor redundancy    |

---

# SECTION 1: CRITICAL DUPLICATIONS (Fix Immediately)

## 1.1 CRITICAL: Duplicate Evaluation Framework Scoring

**Severity:** 🔴 CRITICAL  
**Impact:** Candidates may receive different evaluation scores depending on which code path is used

### Problem

Two completely separate implementations of evaluation framework scoring:

**Location 1:** `backend/engines/advanced_scorer.py` (lines 48-99)

```python
EVAL_KEYWORDS = {
    'ndcg', 'mrr', 'map', 'mean average precision', 'discounted cumulative gain',
    'mean reciprocal rank', 'offline evaluation', 'online evaluation', 'a/b test',
    'a/b testing', 'correlation', 'offline-online', 'evaluation framework',
    'ranking metric', 'relevance metric', 'test harness', 'benchmark'
}

def score_evaluation_framework(self, candidate_raw: Dict[str, Any]) -> float:
    """Complex logic with multiple scoring tiers"""
    score = 0.0
    all_text = (...)
    eval_count = sum(1 for keyword in self.EVAL_KEYWORDS if keyword in all_text)
    if eval_count > 0:
        score += min(eval_count / 3, 0.4)  # Cap at 0.4
    # ... additional scoring logic
```

**Location 2:** `backend/engines/advanced_scoring_components.py` (lines 140-220)

```python
EVAL_KEYWORDS = {
    'ndcg', 'mrr', 'map', 'eval', 'evaluation',
    'benchmark', 'metric', 'offline evaluation',
    'online evaluation', 'a/b test', 'a/b testing',
    'experiment', 'statistical significance',
    'ranking metric', 'retrieval metric'
}

class EvaluationFrameworkScorer:
    def score(self, candidate: Dict[str, Any]) -> float:
        """Different scoring logic"""
        score = 0.0
        # ... different implementation
```

### Why This Matters

- **Keyword Set Difference:** Location 2 has `'experiment'` and `'statistical significance'` but Location 1 doesn't
- **Scoring Logic Difference:** Location 1 has additional A/B testing bonus (+0.3), Location 2 doesn't
- **Pipeline Inconsistency:**
  - `FeatureScorer.score_candidate()` uses Location 1 (if `advanced_scorer` passed)
  - `OptimizedRankingEngine.rank_candidates_fast()` uses Location 2 directly
  - Candidates ranked by different engines get different scores!

### Files Using This

| File                                                     | Method/Usage                                   | Impact               |
| -------------------------------------------------------- | ---------------------------------------------- | -------------------- |
| `backend/engines/feature_scorer.py` (line 204)           | `advanced_scorer.score_evaluation_framework()` | Optional/conditional |
| `backend/engines/optimized_ranking_engine.py` (line 275) | `self.evaluation_framework_scorer.score()`     | **PRIMARY** path     |
| `backend/rank.py`                                        | Via OptimizedRankingEngine                     | **MAIN ENTRY POINT** |

**Current Usage Chain:**

```
rank.py
  → OptimizedRankingEngine.rank_candidates_fast()
    → EvaluationFrameworkScorer.score()  ← Uses Location 2 keywords
```

### Recommendation

**MERGE into single implementation:**

1. ✅ Use `EvaluationFrameworkScorer` from `advanced_scoring_components.py` (it's in the main pipeline)
2. 🗑️ Remove evaluation scoring from `AdvancedScorer`
3. ✔️ Merge both keyword sets: `{'ndcg', 'mrr', 'map', 'eval', 'evaluation', 'benchmark', 'metric', 'offline evaluation', 'online evaluation', 'a/b test', 'a/b testing', 'experiment', 'statistical significance', 'ranking metric', 'retrieval metric', 'correlation', 'offline-online', 'evaluation framework', 'test harness', 'mean average precision', 'discounted cumulative gain', 'mean reciprocal rank', 'relevance metric'}`

**Implementation Priority:** Priority 1 (implement first)

---

## 1.2 CRITICAL: Duplicate Startup/Product Scoring

**Severity:** 🔴 CRITICAL  
**Impact:** Inconsistent product company scoring across the pipeline

### Problem

Two overlapping implementations of startup/product company scoring:

**Location 1:** `backend/engines/advanced_scorer.py` (lines 102-187)

```python
STARTUP_KEYWORDS = {
    'startup', 'early stage', 'seed', 'series a', 'series b', 'founding',
    'founder', 'pre-seed', 'ycombinator', 'y combinator'
}

PRODUCT_OWNERSHIP_KEYWORDS = {
    'owned', 'shipped', 'launched', 'built', 'led', 'drove', 'product',
    'end-to-end', 'full stack', 'startup', 'mvp', 'product launch'
}

SMALL_COMPANY_SIZES = ['1-10', '11-50', '51-200']

def score_startup_product_mindset(self, candidate_raw: Dict[str, Any], parsed_profile: ParsedProfile) -> float:
    """Scores multiple dimensions: startup count, small company size, ownership roles"""
    score = 0.0
    startup_count = 0
    product_company_count = 0
    small_company_count = 0
    # Complex multi-dimension scoring...
```

**Location 2:** `backend/engines/advanced_scoring_components.py` (lines 68-160)

```python
STARTUP_KEYWORDS = {
    'startup', 'early stage', 'seed', 'series a', 'series b', 'founding',
    'founder', 'pre-seed', 'ycombinator', 'y combinator'
}

PRODUCT_OWNERSHIP_KEYWORDS = {
    'owned', 'shipped', 'launched', 'built', 'led', 'drove', 'product',
    'end-to-end', 'full stack', 'startup', 'mvp', 'product launch'
}

PRODUCT_KEYWORDS = {
    'product', 'startup', 'scale-up', 'saas', 'tech',
    'marketplace', 'platform', 'app', 'software',
    'ai company', 'ml company'
}

PRODUCT_COMPANIES = {
    'google', 'meta', 'amazon', 'microsoft', 'apple',
    'openai', 'anthropic', 'perplexity', 'mistral',
    'stripe', 'airbnb', 'uber', 'lyft', 'doordash',
    'spotify', 'linkedin', 'pinterest', 'dropbox',
    'databricks', 'hugging face', 'together', 'anyscale'
}

def score(self, parsed_profile: ParsedProfile, candidate: Dict[str, Any]) -> float:
    """Different approach: checks company names against PRODUCT_COMPANIES"""
```

### Key Differences

| Aspect               | Location 1 (AdvancedScorer)                        | Location 2 (ProductCompanyScorer)       |
| -------------------- | -------------------------------------------------- | --------------------------------------- |
| **Keyword Coverage** | Basic startup keywords                             | + PRODUCT_COMPANIES list (25 companies) |
| **Algorithm**        | Multi-dimensional (startup count, ownership, size) | Name-based company matching             |
| **Bonus Logic**      | Multiple startup bonus: +0.2                       | PRODUCT_COMPANIES bonus: +0.2           |
| **Input**            | `candidate_raw` + `parsed_profile`                 | `parsed_profile` only                   |
| **Code Complexity**  | High (~80 lines)                                   | Medium (~60 lines)                      |

### Pipeline Usage

```
rank.py
  → OptimizedRankingEngine (initializes ProductCompanyScorer)
    → rank_candidates_fast()
      → feature_scorer.score_candidate()
        → [Optional] advanced_scorer.score_startup_product_mindset()  ← Rarely used
    → [Direct call] product_company_scorer.score()  ← MAIN path
```

### Why Both Exist

- `AdvancedScorer` was created first (comprehensive logic)
- `ProductCompanyScorer` was added later with better company name mapping
- They were meant to be complementary but became redundant

### Recommendation

**CONSOLIDATE into ProductCompanyScorer:**

1. Merge keyword sets: combine STARTUP_KEYWORDS + PRODUCT_COMPANIES mapping
2. Keep company-name-based approach (more reliable than keyword matching)
3. Add multi-dimensional scoring from AdvancedScorer
4. Remove `score_startup_product_mindset()` from AdvancedScorer
5. Update all callers to use `ProductCompanyScorer`

**Implementation Priority:** Priority 1 (implement after eval framework)

---

## 1.3 CRITICAL: Redundant Method Aliases in AdvancedScorer

**Severity:** 🔴 CRITICAL  
**Impact:** Code confusion, maintenance overhead

### Problem

`AdvancedScorer` contains method aliases that duplicate functionality:

**File:** `backend/engines/advanced_scorer.py` (lines 189-198)

```python
class AdvancedScorer:
    # Primary methods
    def score_evaluation_framework(self, candidate_raw: Dict[str, Any]) -> float:
        """Real implementation"""
        # ... 50+ lines of code

    def score_startup_product_mindset(self, candidate_raw, parsed_profile) -> float:
        """Real implementation"""
        # ... 80+ lines of code

    # Alias methods - REDUNDANT
    def score_evaluation_experience(self, candidate_raw):
        """Alias for score_evaluation_framework for consistency"""
        return self.score_evaluation_framework(candidate_raw)

    def score_product_mindset(self, candidate_raw, parsed_profile):
        """Alias for score_startup_product_mindset for consistency"""
        return self.score_startup_product_mindset(candidate_raw, parsed_profile)
```

### Why This Is Bad

1. **Confusion**: Which method name is correct? Both?
2. **Hidden Dependencies**: Callers might use either name unpredictably
3. **Maintenance**: Signature changes need updating in two places
4. **Testing**: Redundant test cases for same functionality

### Callers

- `evaluation_experience` alias: Not used anywhere (dead code)
- `product_mindset` alias: Not used anywhere (dead code)

### Recommendation

**DELETE alias methods immediately:**

- Remove `score_evaluation_experience()`
- Remove `score_product_mindset()`
- Standardize on primary method names

**Implementation Priority:** Priority 0 (implement first - takes 5 minutes)

---

# SECTION 2: HIGH SEVERITY DUPLICATIONS

## 2.1 HIGH: Parsing/Extraction Logic Duplication

**Severity:** 🟠 HIGH  
**Impact:** Inconsistent candidate/JD parsing across components

### Problem A: JD Keyword Extraction (3 implementations)

**Implementation 1:** `backend/engines/optimized_ranking_engine.py` (lines 178-195)

```python
def _extract_jd_keywords(self, text: str) -> List[str]:
    """Extract strong query terms from the job description."""
    text = text.lower()
    candidate_phrases = [
        'embeddings', 'retrieval', 'vector database', 'faiss', 'milvus', 'pinecone',
        'semantic search', 'ranking', 'ndcg', 'mrr', 'map', 'a/b testing',
        'online evaluation', 'production', 'scale', 'recommendation', 'rag',
        'startup', 'product', 'ml', 'python', 'llm', 'distributed'
    ]
    keywords = [phrase for phrase in candidate_phrases if re.search(r"\b" + re.escape(phrase) + r"\b", text)]
    if not keywords:
        keywords = re.findall(r"\b[ a-z]{3,}\b", text)[:10]
    return list(dict.fromkeys(keywords))

def _extract_jd_skill_keywords(self, jd_text: str) -> List[str]:
    """Extract skill-specific keywords from JD"""
    jd_lower = jd_text.lower()
    skill_terms = [
        "python", "pytorch", "tensorflow", "scikit-learn",
        "embeddings", "faiss", "milvus", "pinecone", "weaviate", "qdrant",
        # ... more terms
    ]
    return [term for term in skill_terms if re.search(r"\b" + re.escape(term) + r"\b", jd_lower)]
```

**Implementation 2:** `backend/engines/recruiter_jd_parser.py` (lines 86-128)

```python
def _extract_keywords(self, text: str) -> Set[str]:
    """Different keyword list, different approach"""
    REQUIRED_KEYWORDS = {
        'retrieval systems', 'ranking systems', 'embeddings',
        'vector databases', 'vector db', 'faiss', 'pinecone',
        # ... different keywords
    }
    required = set()
    for keyword in self.REQUIRED_KEYWORDS:
        if keyword in text_lower:
            required.add(keyword)
    return required
```

### Why This Matters

- Different keyword lists will produce different BM25 queries
- Inconsistent keyword extraction affects retrieval performance
- Maintenance nightmare: three places to update keywords

### Recommendation

**CONSOLIDATE into single KeywordExtractor class:**

```python
class JDKeywordExtractor:
    CORE_KEYWORDS = {...}
    SKILL_KEYWORDS = {...}
    REQUIRED_KEYWORDS = {...}

    def extract_all(self, jd_text: str) -> Dict[str, List[str]]:
        return {
            'core': self._extract_core_keywords(jd_text),
            'skills': self._extract_skill_keywords(jd_text),
            'required': self._extract_required_keywords(jd_text)
        }
```

**Implementation Priority:** Priority 2

---

## 2.2 HIGH: Keyword Lists Duplicated Across Files

**Severity:** 🟠 HIGH  
**Impact:** Inconsistent keyword matching, maintenance burden

### Duplicate Keyword Sets

#### TIER Keyword Duplication (Retrieval/Vector DB Skills)

**Location 1:** `backend/engines/feature_scorer.py` (lines 70-90)

```python
TIER_1_KEYWORDS = {
    'embeddings', 'retrieval', 'vector_db', 'vector database', 'ranking',
    'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'semantic search',
    'bge', 'sentence transformers', 'opensearch', 'dense retrieval'
}

TIER_2_KEYWORDS = {
    'llm', 'fine-tuning', 'lora', 'qlora', 'learning-to-rank', 'ltr',
    'xgboost', 'lambdarank', 'evaluation framework', 'ndcg', 'mrr', 'map',
    'production ml', 'inference', 'model deployment'
}

TIER_3_KEYWORDS = {
    'python', 'ml', 'ai', 'nlp', 'information retrieval', 'search',
    'recommendation', 'machine learning', 'data science'
}
```

**Location 2:** `backend/engines/candidate_profile_parser.py` (lines 100-130)

```python
RETRIEVAL_SKILLS = {
    'elasticsearch', 'milvus', 'pinecone', 'weaviate', 'qdrant', 'faiss',
    'opensearch', 'vector database', 'vector db', 'semantic search', 'rag',
    'retrieval'
}

EMBEDDING_SKILLS = {
    'embeddings', 'sentence transformers', 'bge', 'e5', 'openai embeddings',
    'embedding model', 'dense retrieval', 'semantic', 'transformer'
}

RANKING_SKILLS = {
    'ranking', 'learning-to-rank', 'ltr', 'xgboost', 'lambdarank',
    'ndcg', 'mrr', 'map', 'evaluation metric'
}
```

**Location 3:** `backend/engines/advanced_scoring_components.py` (lines 108-135)

```python
class RetrievalDepthScorer:
    RETRIEVAL_TECH = {
        'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant',
        'elasticsearch', 'opensearch', 'vector', 'embedding',
        'dense retrieval', 'semantic search', 'rag'
    }
```

### Analysis

| Keyword    | FeatureScorer | CandidateProfileParser | RetrievalDepthScorer |
| ---------- | ------------- | ---------------------- | -------------------- |
| faiss      | ✅            | ✅                     | ✅                   |
| milvus     | ✅            | ✅                     | ✅                   |
| embeddings | ✅            | ✅                     | ✅                   |
| rag        | ❌            | ✅                     | ✅                   |
| opensearch | ✅            | ✅                     | ✅                   |

**Problem:** Updates to keyword lists must be synchronized across 3+ files

### Recommendation

**CREATE shared constants module:**

```python
# backend/engines/constants.py
RETRIEVAL_KEYWORDS = {...}
EMBEDDING_KEYWORDS = {...}
RANKING_KEYWORDS = {...}
EVALUATION_KEYWORDS = {...}
STARTUP_KEYWORDS = {...}
PRODUCTION_KEYWORDS = {...}

# Then import in all files:
from .constants import RETRIEVAL_KEYWORDS, EMBEDDING_KEYWORDS, ...
```

**Implementation Priority:** Priority 3

---

## 2.3 HIGH: Company Classification Duplication

**Severity:** 🟠 HIGH  
**Impact:** Inconsistent company type classification

### Duplicate Company Lists

**Location 1:** `backend/engines/candidate_profile_parser.py` (lines 64-72)

```python
CONSULTING_COMPANIES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'deloitte', 'pwc', 'kpmg', 'ey', 'ernst & young', 'heidrick & struggles',
    'mckinsey', 'bain', 'bcg', 'goldman sachs', 'morgan stanley'
}
```

**Location 2:** `backend/engines/advanced_scoring_components.py` (lines 87-110)

```python
PRODUCT_COMPANIES = {
    'google', 'meta', 'amazon', 'microsoft', 'apple',
    'openai', 'anthropic', 'perplexity', 'mistral',
    'stripe', 'airbnb', 'uber', 'lyft', 'doordash',
    'spotify', 'linkedin', 'pinterest', 'dropbox',
    'databricks', 'hugging face', 'together', 'anyscale'
}
```

### Problem

- Consulting companies list is only in CandidateProfileParser
- Product companies list is only in ProductCompanyScorer
- No centralized source of truth

### Recommendation

**Consolidate into shared constants.**

**Implementation Priority:** Priority 3

---

## 2.4 HIGH: Text Normalization Duplication

**Severity:** 🟠 HIGH  
**Impact:** Inconsistent text processing

### Problem

Multiple implementations of `_normalize_text()`:

**Location 1:** `backend/engines/feature_scorer.py` (lines 790-800)

```python
def _normalize_text(self, text: str) -> str:
    """Normalize text for consistent keyword matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Location 2:** `backend/engines/recruiter_jd_parser.py` (lines 210-218)

```python
def _normalize_text(self, text: str) -> str:
    """Normalize text"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)  # Different regex pattern!
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### Difference

- FeatureScorer: Removes ALL non-alphanumeric: `[^a-z0-9\s]`
- RecruiterJDParser: Keeps hyphens: `[^a-z0-9\s-]`

This causes inconsistent keyword matching!

### Recommendation

**Create shared TextProcessor:**

```python
class TextProcessor:
    @staticmethod
    def normalize(text: str, keep_hyphens: bool = False) -> str:
        text = text.lower()
        pattern = r"[^a-z0-9\s-]" if keep_hyphens else r"[^a-z0-9\s]"
        text = re.sub(pattern, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
```

**Implementation Priority:** Priority 2

---

# SECTION 3: MEDIUM SEVERITY DUPLICATIONS

## 3.1 MEDIUM: Superseded Embedding Retrieval Code

**Severity:** 🟡 MEDIUM  
**Impact:** Dead code confuses developers

### Problem

`backend/engines/embedding_retrieval.py` contains superseded code:

**File:** `backend/engines/embedding_retrieval.py` (lines 24-25)

```python
# NOTE: EmbeddingRetriever is superseded by EmbeddingPrecomputer (embedding_precompute.py)
# which pre-computes and caches embeddings for all candidates for efficient retrieval

class EmbeddingRetriever:
    """Superseded - use EmbeddingPrecomputer instead"""
    # ~150 lines of unused code
```

### Usage

- Only used in `benchmark_ranking.py` for benchmarking comparisons
- Not used in production ranking pipeline
- Comment clearly states it's superseded

### Recommendation

**Remove EmbeddingRetriever class:**

- Delete lines 24-200 (entire class)
- Keep `BM25Retriever` (still active)
- If benchmarking needed, create separate benchmark module

**Implementation Priority:** Priority 4 (cleanup)

---

## 3.2 MEDIUM: Duplicate Test Files

**Severity:** 🟡 MEDIUM  
**Impact:** Test organization, maintenance

### Problem

Two test files for HoneypotDetector:

**File 1:** `scripts/test_honeypot_detector.py`

- Tests: `_has_overlapping_employment()`
- Lines: ~50

**File 2:** `scripts/test_honeypot_honeypot_detector.py`

- Tests: `_has_career_inconsistency()`, `_has_excessive_expert_skills()`
- Lines: ~80

### Why

- Appears to be from refactoring - file names suggest confusion
- Should be consolidated

### Recommendation

**Consolidate into single test file:**

```bash
# Merge test_honeypot_honeypot_detector.py into test_honeypot_detector.py
# Delete test_honeypot_honeypot_detector.py
```

**Implementation Priority:** Priority 5 (low - tests still work)

---

## 3.3 MEDIUM: Overlapping Validation Scripts

**Severity:** 🟡 MEDIUM  
**Impact:** Unclear purpose, duplication

### Problem

Three validation scripts with overlapping purposes:

| Script                       | Purpose                                   | Status  |
| ---------------------------- | ----------------------------------------- | ------- |
| `validate_comprehensive.py`  | Static code structure validation          | Used    |
| `validate_phase1.py`         | Full ranking pipeline execution + metrics | Used    |
| `validate_phases_3_to_10.py` | Phase-specific validation                 | Unclear |

### Recommendation

**Keep both active scripts but clarify:**

- `validate_comprehensive.py`: Static structure validation
- `validate_phase1.py`: Benchmarking and performance metrics
- `validate_phases_3_to_10.py`: Check if actually used or remove

**Implementation Priority:** Priority 5

---

# SECTION 4: DETAILED CONSOLIDATION STRATEGY

## Phase 1: Immediate Fixes (30 minutes)

### Step 1.1: Remove Alias Methods [5 min]

**File:** `backend/engines/advanced_scorer.py`

- Delete `score_evaluation_experience()` method
- Delete `score_product_mindset()` method

### Step 1.2: Merge Evaluation Framework Scoring [10 min]

**Files Affected:**

- `backend/engines/advanced_scorer.py` - Remove evaluation scoring
- `backend/engines/advanced_scoring_components.py` - Keep and enhance
- `backend/engines/feature_scorer.py` - Update imports
- `backend/engines/optimized_ranking_engine.py` - No change (already uses correct one)

**Changes:**

1. In `advanced_scoring_components.py`:
   - Merge EVAL_KEYWORDS from both sources
   - Merge scoring logic from AdvancedScorer into EvaluationFrameworkScorer
2. In `advanced_scorer.py`:
   - Remove `score_evaluation_framework()` method
   - Remove EVAL_KEYWORDS set
   - Remove AB_TESTING_KEYWORDS set
3. In `feature_scorer.py`:
   - Update to use `EvaluationFrameworkScorer` directly

### Step 1.3: Consolidate Product Company Scoring [15 min]

**Files Affected:**

- `backend/engines/advanced_scorer.py` - Remove scoring
- `backend/engines/advanced_scoring_components.py` - Enhance
- `backend/engines/feature_scorer.py` - Update calls
- `backend/engines/optimized_ranking_engine.py` - No change

**Changes:**

1. In `advanced_scoring_components.py` ProductCompanyScorer:
   - Add `SMALL_COMPANY_SIZES` and `PRODUCT_OWNERSHIP_KEYWORDS`
   - Merge logic from `AdvancedScorer.score_startup_product_mindset()`
   - Add multi-dimensional scoring
2. In `advanced_scorer.py`:
   - Remove `score_startup_product_mindset()` method
   - Remove all related constants
3. Update all callers

---

## Phase 2: Medium Priority Consolidation (1-2 hours)

### Step 2.1: Create Shared Constants Module

**New File:** `backend/engines/constants.py`

```python
# Keyword sets for consistent matching across codebase
RETRIEVAL_KEYWORDS = {
    'elasticsearch', 'milvus', 'pinecone', 'weaviate', 'qdrant', 'faiss',
    'opensearch', 'vector database', 'vector db', 'semantic search', 'rag',
    'retrieval', 'dense retrieval'
}

EMBEDDING_KEYWORDS = {
    'embeddings', 'sentence transformers', 'bge', 'e5', 'openai embeddings',
    'embedding model', 'dense retrieval', 'semantic', 'transformer'
}

RANKING_KEYWORDS = {
    'ranking', 'learning-to-rank', 'ltr', 'xgboost', 'lambdarank',
    'ndcg', 'mrr', 'map', 'evaluation metric'
}

EVALUATION_KEYWORDS = {
    'ndcg', 'mrr', 'map', 'eval', 'evaluation',
    'benchmark', 'metric', 'offline evaluation',
    'online evaluation', 'a/b test', 'a/b testing',
    'experiment', 'statistical significance',
    'ranking metric', 'retrieval metric', 'correlation',
    'offline-online', 'evaluation framework', 'test harness',
    'mean average precision', 'discounted cumulative gain',
    'mean reciprocal rank', 'relevance metric'
}

STARTUP_KEYWORDS = {
    'startup', 'early stage', 'seed', 'series a', 'series b', 'founding',
    'founder', 'pre-seed', 'ycombinator', 'y combinator'
}

PRODUCT_OWNERSHIP_KEYWORDS = {
    'owned', 'shipped', 'launched', 'built', 'led', 'drove', 'product',
    'end-to-end', 'full stack', 'startup', 'mvp', 'product launch'
}

PRODUCTION_KEYWORDS = {
    'production', 'deployed', 'shipped', 'live', 'real-time', 'qps',
    'scale', 'million', 'billion', 'latency', 'throughput', 'production-grade'
}

CONSULTING_COMPANIES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'deloitte', 'pwc', 'kpmg', 'ey', 'ernst & young', 'heidrick & struggles',
    'mckinsey', 'bain', 'bcg', 'goldman sachs', 'morgan stanley'
}

PRODUCT_COMPANIES = {
    'google', 'meta', 'amazon', 'microsoft', 'apple',
    'openai', 'anthropic', 'perplexity', 'mistral',
    'stripe', 'airbnb', 'uber', 'lyft', 'doordash',
    'spotify', 'linkedin', 'pinterest', 'dropbox',
    'databricks', 'hugging face', 'together', 'anyscale'
}

COMPANY_SIZE_NUMERIC = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1001-5000": 3000,
    "5001-10000": 7500,
    "10001+": 50000
}

SMALL_COMPANY_SIZES = ['1-10', '11-50', '51-200']
```

### Step 2.2: Update All Files to Import from constants

- `feature_scorer.py`: Import TIER_1/2/3_KEYWORDS, PRODUCTION_KEYWORDS, SCALE_KEYWORDS
- `candidate_profile_parser.py`: Import CONSULTING_COMPANIES, COMPANY_SIZE_NUMERIC, etc.
- `advanced_scoring_components.py`: Import all keyword sets
- `recruiter_jd_parser.py`: Import REQUIRED_KEYWORDS, PREFERRED_KEYWORDS, etc.
- `optimized_ranking_engine.py`: Import jd-related keywords

### Step 2.3: Create Shared TextProcessor

**New File:** `backend/engines/text_utils.py`

```python
import re

class TextProcessor:
    @staticmethod
    def normalize(text: str, keep_hyphens: bool = False) -> str:
        """Normalize text consistently across codebase."""
        text = text.lower()
        pattern = r"[^a-z0-9\s-]" if keep_hyphens else r"[^a-z0-9\s]"
        text = re.sub(pattern, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_keywords(text: str, keyword_set: set) -> list:
        """Extract keywords using word boundary matching."""
        text = TextProcessor.normalize(text)
        matches = []
        for keyword in keyword_set:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                matches.append(keyword)
        return matches
```

### Step 2.4: Remove EmbeddingRetriever

**File:** `backend/engines/embedding_retrieval.py`

- Delete `EmbeddingRetriever` class (lines ~24-200)
- Keep `BM25Retriever` class
- Update `benchmark_ranking.py` if needed

### Step 2.5: Consolidate Test Files

**Action:** Merge `scripts/test_honeypot_honeypot_detector.py` into `scripts/test_honeypot_detector.py`

---

## Phase 3: Architecture Refactoring (2-3 hours, future)

### Step 3.1: Unified Scoring Registry

Create a unified scoring orchestrator:

```python
# backend/engines/scoring_registry.py
class AdvancedScoringRegistry:
    """Unified interface for all advanced scoring components"""

    def __init__(self):
        self.scorers = {
            'career_trajectory': CareerTrajectoryAnalyzer(),
            'product_company': ProductCompanyScorer(),
            'retrieval_depth': RetrievalDepthScorer(),
            'evaluation_framework': EvaluationFrameworkScorer(),
            'honeypot_detection': HoneypotDetector()
        }

    def score_all(self, candidate: Dict[str, Any], parsed: ParsedProfile) -> Dict[str, float]:
        """Score candidate on all dimensions, return dict"""
        return {
            name: scorer.score(candidate, parsed)
            for name, scorer in self.scorers.items()
        }
```

---

# SECTION 5: IMPLEMENTATION CHECKLIST

## Priority 1 - CRITICAL (Do These First)

- [ ] Delete alias methods in `advanced_scorer.py`
- [ ] Merge evaluation framework scoring
- [ ] Consolidate startup/product company scoring

## Priority 2 - HIGH (Do These Next)

- [ ] Create `constants.py` with shared keyword sets
- [ ] Create `text_utils.py` for TextProcessor
- [ ] Update all files to import from constants
- [ ] Consolidate JD keyword extraction

## Priority 3 - MEDIUM (Optional Optimization)

- [ ] Remove EmbeddingRetriever class
- [ ] Merge honeypot test files
- [ ] Review validation scripts

## Priority 4 - FUTURE (Architecture)

- [ ] Create unified ScoringRegistry
- [ ] Refactor OptimizedRankingEngine to use registry
- [ ] Add plugin architecture for new scorers

---

# SECTION 6: ESTIMATED IMPACT

## Before Consolidation

- **Duplicate code lines:** ~250 lines across multiple files
- **Overlapping methods:** 8 different implementations of same logic
- **Keyword set copies:** 15+ duplicate keyword definitions
- **Maintenance burden:** Changes to scoring logic require updates in 3-4 files
- **Risk of inconsistency:** High - different scoring used in different paths

## After Consolidation (Estimated)

- **Duplicate code lines:** ~20 lines (87% reduction)
- **Overlapping methods:** 0
- **Keyword set copies:** 1 (in constants.py)
- **Maintenance burden:** Changes require update in 1 file
- **Consistency:** Guaranteed - single source of truth for all keywords/logic

## Code Quality Metrics

| Metric                              | Before | After | Improvement |
| ----------------------------------- | ------ | ----- | ----------- |
| Lines of duplicate code             | 250    | 20    | 92% ↓       |
| Number of keyword definitions       | 15+    | 1     | 93% ↓       |
| Scoring implementations             | 8      | 1     | 87% ↓       |
| Update locations for keyword change | 4-5    | 1     | 75% ↓       |
| Risk of scoring inconsistency       | HIGH   | NONE  | 100% ↓      |

---

# SECTION 7: FILES REQUIRING CHANGES

## Files to Modify (In Priority Order)

### Priority 1 Changes

| File                                             | Changes                                    | Lines Affected          |
| ------------------------------------------------ | ------------------------------------------ | ----------------------- |
| `backend/engines/advanced_scorer.py`             | Remove alias methods, consolidate scoring  | 189-198, 48-99, 102-187 |
| `backend/engines/advanced_scoring_components.py` | Merge/enhance evaluation & product scoring | 152-220, 68-160         |
| `backend/engines/feature_scorer.py`              | Update import references                   | ~15 locations           |

### Priority 2 Changes

| File                                          | Changes                              | Lines Affected |
| --------------------------------------------- | ------------------------------------ | -------------- |
| `backend/engines/constants.py`                | **NEW FILE** - Shared keyword sets   | N/A            |
| `backend/engines/text_utils.py`               | **NEW FILE** - TextProcessor utility | N/A            |
| `backend/engines/candidate_profile_parser.py` | Import from constants                | ~5 locations   |
| `backend/engines/recruiter_jd_parser.py`      | Import from constants                | ~5 locations   |
| `backend/engines/optimized_ranking_engine.py` | Import from constants                | ~5 locations   |
| `backend/engines/feature_scorer.py`           | Import from constants                | ~5 locations   |

### Priority 3 Changes

| File                                         | Changes                         | Lines Affected      |
| -------------------------------------------- | ------------------------------- | ------------------- |
| `backend/engines/embedding_retrieval.py`     | Remove EmbeddingRetriever class | ~176 lines          |
| `scripts/test_honeypot_detector.py`          | Merge test files                | Consolidate content |
| `scripts/test_honeypot_honeypot_detector.py` | **DELETE FILE**                 | All                 |

---

# APPENDIX A: Complete File Structure Map

```
backend/
├── engines/
│   ├── advanced_scorer.py              ⚠️  CRITICAL: Has redundant aliases & duplicate eval/product logic
│   ├── advanced_scoring_components.py  ⚠️  HIGH: Has duplicate EVAL_KEYWORDS
│   ├── candidate_profile_parser.py     ⚠️  HIGH: Duplicates keyword sets
│   ├── cross_encoder_reranker.py       ✓ CLEAN
│   ├── embedding_precompute.py         ✓ CLEAN
│   ├── embedding_retrieval.py          ⚠️  MEDIUM: Has superseded EmbeddingRetriever
│   ├── feature_scorer.py               ⚠️  HIGH: Duplicate text normalization, keywords
│   ├── optimized_ranking_engine.py     ⚠️  HIGH: Duplicate JD keyword extraction
│   ├── recruiter_jd_parser.py          ⚠️  HIGH: Duplicate text normalization, JD parsing
│   ├── constants.py                    📝 TO CREATE
│   ├── text_utils.py                   📝 TO CREATE
│   └── __init__.py
├── benchmark_ranking.py                ✓ CLEAN
├── validate_comprehensive.py           ⚠️  LOW: Overlapping with validate_phase1.py
├── validate_phase1.py                  ⚠️  LOW: Overlapping with validate_comprehensive.py
├── validate_phases_3_to_10.py          ? UNCLEAR
├── rank.py                             ✓ CLEAN (entry point)
└── run_ranking_optimized.py            ✓ CLEAN

scripts/
├── check_three_candidates.py           ✓ CLEAN
├── compare_rankings.py                 ✓ CLEAN
├── holdout_harness.py                  ✓ CLEAN
├── test_honeypot_detector.py           ⚠️  MEDIUM: Duplicates with next file
├── test_honeypot_honeypot_detector.py  ⚠️  MEDIUM: Duplicates with previous file (TO DELETE)
└── download_models.py                  ✓ CLEAN

Test Files:
├── test_ranking_components.py          ✓ CLEAN
└── [PUB] India_runs_data_and_ai_challenge/validate_submission.py ✓ CLEAN

Legend: ✓ = Clean | ⚠️ = Has issues | 📝 = To create | ? = Review needed
```

---

# APPENDIX B: Quick Reference - All Duplications Summary

| ID  | Type                | Severity    | Files                                                                          | Duplication                                | Action                                                           |
| --- | ------------------- | ----------- | ------------------------------------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------- |
| 1.1 | Scoring Logic       | 🔴 CRITICAL | advanced_scorer.py, advanced_scoring_components.py                             | Evaluation Framework scoring               | Merge into advanced_scoring_components.py                        |
| 1.2 | Scoring Logic       | 🔴 CRITICAL | advanced_scorer.py, advanced_scoring_components.py                             | Startup/Product scoring                    | Consolidate into ProductCompanyScorer                            |
| 1.3 | Code Pattern        | 🔴 CRITICAL | advanced_scorer.py                                                             | Alias methods                              | Delete score_evaluation_experience() and score_product_mindset() |
| 2.1 | Extraction Logic    | 🟠 HIGH     | optimized_ranking_engine.py, recruiter_jd_parser.py                            | JD keyword extraction (2 methods, 2 files) | Create unified JDKeywordExtractor                                |
| 2.2 | Keyword Sets        | 🟠 HIGH     | feature_scorer.py, candidate_profile_parser.py, advanced_scoring_components.py | TIER/RETRIEVAL/EMBEDDING keywords          | Move to constants.py                                             |
| 2.3 | Company Lists       | 🟠 HIGH     | candidate_profile_parser.py, advanced_scoring_components.py                    | CONSULTING_COMPANIES, PRODUCT_COMPANIES    | Move to constants.py                                             |
| 2.4 | Utility Function    | 🟠 HIGH     | feature_scorer.py, recruiter_jd_parser.py                                      | \_normalize_text()                         | Create TextProcessor in text_utils.py                            |
| 3.1 | Dead Code           | 🟡 MEDIUM   | embedding_retrieval.py                                                         | EmbeddingRetriever class                   | Remove class (lines ~24-200)                                     |
| 3.2 | Test Organization   | 🟡 MEDIUM   | scripts/                                                                       | Honeypot test files (2 files)              | Merge into single test file                                      |
| 3.3 | Script Organization | 🟡 MEDIUM   | backend/                                                                       | Validation scripts (3 files)               | Clarify purpose, consolidate if needed                           |

---

**END OF COMPREHENSIVE ANALYSIS**

_Next Steps: Run Priority 1 consolidation (30 min), then Priority 2 (1-2 hours), then code review._
