# Candidate Ranking Strategy for Senior AI Engineer Role

## Redrob Hackathon Challenge

**Date**: June 5, 2026  
**Role**: Senior AI Engineer — Founding Team at Redrob AI  
**Objective**: Rank 100K candidates for top 100 matches with explainable reasoning

---

## Executive Summary

This document outlines a data-driven ranking strategy that aligns candidate profiles with the specific requirements of the Senior AI Engineer role at Redrob AI. The JD explicitly warns against keyword-matching and emphasizes **real production experience with retrieval/ranking systems**, **strong Python**, **product engineering mindset**, and **recency** over resume credentials.

### Key Strategic Insight from JD

> "The 'right answer' is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."

This means our ranking system must look beyond surface-level keyword matches to assess **actual production experience** reflected in career history, timeline consistency, realistic skill combinations, and behavioral engagement signals.

---

## 1. Role Requirements Analysis

### Must-Have (Disqualifiers if Missing)

1. **Production Experience with Embeddings-Based Retrieval** (e.g., Sentence Transformers, BGE, Milvus)
2. **Production Experience with Vector Databases/Hybrid Search** (Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, FAISS)
3. **Strong Python** - code quality matters
4. **Ranking System Evaluation Experience** - NDCG, MRR, MAP, offline-to-online correlation
5. **Recent Production Code** - within last 18 months
6. **Product Engineering Mindset** - not pure research background

### Strong Signals (Major Positives)

- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- HR-tech / recruiting tech background
- Distributed systems or large-scale inference optimization
- Open-source contributions in AI/ML

### Explicit Disqualifiers

- Pure research background (no production deployment)
- AI experience <12 months using only LangChain/OpenAI
- Haven't written production code in 18+ months
- Career entirely at consulting firms (TCS, Infosys, Wipro, etc.) with no product company experience
- Primary expertise in computer vision/speech/robotics without NLP/IR experience
- Closed-source-only systems for 5+ years without external validation

### Nice-to-Have Factors

- 5-9 years experience (range, not hard requirement)
- Located in/willing to relocate to Pune/Noida
- Short notice period (< 30 days preferred)

---

## 2. Data Source: Candidate Schema

### Profile Section (Profile Quality Signals)

| Field                  | Purpose                            | Scoring Impact                                         |
| ---------------------- | ---------------------------------- | ------------------------------------------------------ |
| `years_of_experience`  | Baseline seniority filter          | Prefer 5-8 years for this role                         |
| `current_title`        | Role alignment                     | "Engineer", "ML", "AI", "Data" titles signal relevance |
| `current_company`      | Company type (product vs services) | **CRITICAL**: Flag pure consulting companies           |
| `current_company_size` | Scale exposure                     | Product orgs 201+ better than tiny startups            |
| `current_industry`     | Domain alignment                   | Tech/SaaS > Finance > Manufacturing > Other            |
| `summary`              | Self-assessed expertise            | Red flag: keyword stuffing, vague claims               |
| `headline`             | Focus clarity                      | Should mention ML/AI/Engineering                       |

### Career History Section (Most Predictive)

| Field                | Purpose            | Scoring Impact                                                                            |
| -------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| `title` progression  | Career trajectory  | Shows growth from IC → IC or IC → IC (not mgmt)                                           |
| `description`        | **MOST IMPORTANT** | Parse for: embeddings, retrieval, ranking, vector DBs, ranking, evaluation metrics, scale |
| `duration_months`    | Role depth         | >12 months shows hands-on time; <6 months risky                                           |
| `is_current`         | Recent work        | Recent role > 2+ years old                                                                |
| `company_size`       | Scale experience   | 201+ employees signals production systems                                                 |
| Timeline consistency | Data quality       | Check for gaps, overlaps, unrealistic transitions                                         |

### Skills Section (Supplementary Signal)

| Field             | Purpose               | Scoring Impact                                        |
| ----------------- | --------------------- | ----------------------------------------------------- |
| `name`            | Skill relevance       | **Map to role requirements**                          |
| `proficiency`     | Depth claimed         | "expert" / "advanced" more meaningful than "beginner" |
| `endorsements`    | Social proof of skill | 30+ endorsements meaningful; 0 suspicious             |
| `duration_months` | Depth of practice     | >24 months suggests real usage; <6 months exploratory |

**Critical Skills Map for This Role**:

- **Retrieval/Search**: Elasticsearch, Milvus, Pinecone, Weaviate, FAISS, Qdrant, OpenSearch
- **Embeddings**: Sentence Transformers, BGE, E5, "embeddings", "semantic search", "vector"
- **Ranking**: XGBoost, LambdaRank, "ranking", "learning-to-rank", "LTR", "relevance"
- **LLM Fine-tuning**: LoRA, QLoRA, PEFT, "fine-tuning LLMs", "prompt"
- **Evaluation**: NDCG, MRR, MAP, "A/B testing", "offline evaluation", "online metrics"
- **Python**: Python, NumPy, Pandas, Scikit-learn (not just "Python 101")
- **Production ML**: MLflow, BentoML, "production", "deployment", "inference"
- **Infra**: Kubernetes, Docker, "distributed systems", "large-scale", "Apache Spark"

### Education Section

| Field            | Purpose              | Scoring Impact                              |
| ---------------- | -------------------- | ------------------------------------------- |
| `degree`         | Qualification level  | B.E./B.Tech > B.Sc; M.Tech good             |
| `field_of_study` | Alignment            | CS/ML/Math > Hardware/Mechanical > Non-tech |
| `tier`           | Institution prestige | Tier 1-2 > Tier 3-4; optional signal        |

### Certifications Section

- Ignore generic certifications
- **Positive**: ML/AI certifications from reputable sources
- **Negative**: Certification volume with inconsistent background (red flag for credential padding)

### Redrob Signals (Engagement & Availability)

| Signal                                                   | Purpose             | Scoring Impact                                          |
| -------------------------------------------------------- | ------------------- | ------------------------------------------------------- |
| `open_to_work_flag`                                      | Availability        | **Must be true** (else strong discount)                 |
| `profile_completeness_score`                             | Profile quality     | >70% indicates serious candidate                        |
| `github_activity_score`                                  | External validation | >30 is very positive; -1 (no GitHub) is neutral         |
| `recruiter_response_rate`                                | Engagement          | >0.4 good; 0.0-0.1 red flag                             |
| `avg_response_time_hours`                                | Responsiveness      | <24 hrs excellent; >168 hrs poor                        |
| `interview_completion_rate`                              | Reliability         | >0.7 good; <0.4 concerning                              |
| `notice_period_days`                                     | Availability        | <30 preferred; >90 penalty                              |
| `willing_to_relocate`                                    | Location fit        | True for role (Pune/Noida)                              |
| `expected_salary_range_inr_lpa`                          | Alignment check     | Should be reasonable for Sr. role (typically 20-50 LPA) |
| `verified_email`, `verified_phone`, `linkedin_connected` | Authenticity        | Multiple verifications = higher trust                   |
| `saved_by_recruiters_30d`                                | Market signal       | >5 means hot candidate                                  |
| `search_appearance_30d`                                  | Discoverability     | >100 means good search match                            |
| `skill_assessment_scores`                                | **VERY IMPORTANT**  | Passing scores in key skills validate claims            |

---

## 3. Scoring Components (Modular Framework)

Each component produces a score [0.0, 1.0] that combines into final ranking.

### Component 1: Technical Relevance Score (Weight: 35%)

**Goal**: Quantify direct alignment with role requirements

#### 1A. Required Skills Detection (70% of technical score)

Scan career descriptions + skills section for **critical keywords**:

**Tier-1 Keywords (2.0x weight)**: embeddings, retrieval, vector_db, ranking, FAISS, Pinecone, Milvus, Weaviate, semantic_search, BGE, sentence_transformers

**Tier-2 Keywords (1.5x weight)**: LLM, fine-tuning, LoRA, learning_to_rank, XGBoost, evaluation_framework, NDCG, MRR, MAP, production_ML

**Tier-3 Keywords (1.0x weight)**: Python, ML, AI, NLP, information_retrieval, search, recommendation

Score calculation:

```
keyword_score = (
    0.4 * (tier1_keywords_found / max(tier1_keywords_in_jd, 1)) +
    0.35 * (tier2_keywords_found / max(tier2_keywords_in_jd, 1)) +
    0.25 * (tier3_keywords_found / max(tier3_keywords_in_jd, 1))
)
```

Cap at 1.0.

#### 1B. Production Scale Signal (30% of technical score)

From career history descriptions, detect:

- **Scale metrics**: "100K+", "10M+", "1B+", "real-time", "QPS", "millions"
- **Production evidence**: "deployed to users", "live", "production", "customer-facing"
- **Company size**: Large FAANG/AI orgs (Google, Meta, etc.) vs startups vs consulting

Scoring:

```
scale_score = 0.0
if "deployed to" or "live" or "production" in description:
    scale_score += 0.4
if any(metric in description for metric in ["100K", "1M", "real-time", "QPS"]):
    scale_score += 0.4
if company_size >= "501-1000" or "product company":
    scale_score += 0.2
```

#### 1C. Recency Bonus (Decay Factor)

```
recency_penalty = 0.0
if is_current_role and duration_months >= 3:
    recency_penalty = 0.0 (full score)
else if most_recent_role_ended_months_ago <= 6:
    recency_penalty = 0.0
else if ended_6_to_18_months_ago:
    recency_penalty = 0.1
else if ended_18_to_36_months_ago:
    recency_penalty = 0.2
else:
    recency_penalty = 0.3

technical_relevance = (keyword_score * 0.7 + scale_score * 0.3) * (1 - recency_penalty)
```

### Component 2: Production Experience Score (Weight: 25%)

**Goal**: Validate hands-on production shipping track record

#### 2A. Career Depth Analysis

```
deep_production_experience = 0.0
for each role in career_history:
    if "Engineer" or "ML" or "AI" in title and duration_months >= 12:
        deep_production_experience += 1.0
    else if duration_months >= 24:
        deep_production_experience += 0.5

production_experience_score = min(deep_production_experience / 2, 1.0)
```

#### 2B. Role Consistency (Role Should Match Profile)

```
mismatch_penalty = 0.0
if current_title mismatch with career_history titles:
    mismatch_penalty += 0.15
if summary claims AI/ML but history shows pure operations:
    mismatch_penalty += 0.15
if multiple unrelated job hops (every <1 year):
    mismatch_penalty += 0.2

production_experience = (production_experience_score * 0.8 + 0.2) * (1 - mismatch_penalty)
```

### Component 3: Profile Quality Score (Weight: 15%)

**Goal**: Detect suspicious profiles, timeline issues, unrealistic skill combos

#### 3A. Profile Consistency Checks

```
quality_score = 1.0

# Timeline analysis
if overlapping roles or gaps > 12 months:
    quality_score -= 0.2

# Skill realism
skill_count = len(skills)
if skill_count > 50:
    quality_score -= 0.15  # Likely padding
if contains_unrealistic_combo(skills):  # e.g., [Photoshop, FAISS, Kubernetes]
    quality_score -= 0.2

# Profile completeness
if profile_completeness_score < 50:
    quality_score -= 0.2
if summary is empty or too generic:
    quality_score -= 0.1

# Verification signals
if verified_email and verified_phone:
    quality_score += 0.1
```

#### 3B. Behavioral Fit (Disqualifying Checks)

```
behavioral_quality = 1.0

if open_to_work_flag == false:
    behavioral_quality = 0.5  # Strong discount, not full disqualifier

if recruiter_response_rate < 0.1:
    behavioral_quality -= 0.2  # Unresponsive

if interview_completion_rate < 0.4:
    behavioral_quality -= 0.15  # Unreliable
```

### Component 4: Behavioral Engagement Score (Weight: 15%)

**Goal**: Assess availability and market activity

```
behavioral_score = 1.0

# Availability
if notice_period_days <= 30:
    behavioral_score *= 1.0
else if notice_period_days <= 60:
    behavioral_score *= 0.95
else:
    behavioral_score *= 0.80 - (notice_period_days - 60) / 100

# Engagement signals
if github_activity_score > 30:
    behavioral_score *= 1.05  # Cap at 1.0
if recruiter_response_rate > 0.5:
    behavioral_score *= 1.02
if saved_by_recruiters_30d > 5:
    behavioral_score *= 1.02  # Hot candidate

# Market activity
if search_appearance_30d > 100:
    behavioral_score *= 1.02
if profile_completeness_score < 60:
    behavioral_score *= 0.85
```

### Component 5: Experience Level Fit (Weight: 10%)

**Goal**: Match to required seniority (5-9 years)

```
experience_fit = 1.0
if years_of_experience < 3:
    experience_fit = 0.4  # Too junior
else if years_of_experience < 5:
    experience_fit = 0.7
else if years_of_experience <= 9:
    experience_fit = 1.0  # Perfect band
else if years_of_experience <= 12:
    experience_fit = 0.95  # Slightly over but OK
else if years_of_experience <= 15:
    experience_fit = 0.85
else:
    experience_fit = 0.7  # Risk of overqualification / stagnation
```

---

## 4. Disqualifying Factors

Candidates scoring <0.15 on ANY of these are moved to bottom (score \*= 0.1):

1. **pure_research_only**: Career entirely in academia or research labs with no production deployment
2. **all_consulting_background**: Every role at TCS/Infosys/Wipro/Accenture/Cognizant (exceptions for mixed backgrounds)
3. **no_production_code_18mo**: Most recent role is non-technical or >18 months old without explanation
4. **misleading_profile**: Massive skill list (>100 skills) with zero endorsements / assessment scores
5. **honeypot_keyword_only**: Profile mentions all buzzwords but role descriptions are vague/contradictory
6. **no_engagement**: open_to_work_flag=false AND not active in last 6 months

---

## 5. Semantic Retrieval Strategy

### Stage 1: Initial Retrieval (BM25 + Keyword)

1. Index all 100K candidates using BM25 on text fields
2. For each candidate: concatenate headline, summary, current_title, skills, career descriptions
3. Retrieve top 2000 candidates using keyword match for JD terms

### Stage 2: Embedding-Based Ranking

1. **Query Embedding**: Use Sentence Transformers (e.g., BGE-small-en-v1.5) to embed the JD
2. **Candidate Embedding**: Embed concatenated profile text for each candidate
3. **FAISS Indexing**: Build FAISS index for 100K candidate embeddings
4. **Top-K Retrieval**: Retrieve top 3000 candidates by cosine similarity

### Stage 3: Feature Scoring

Apply all scoring components to top 3000 candidates

### Stage 4: Final Ranking

- Combine semantic similarity (5% weight) + component scores (95% weight)
- Apply disqualifying factors
- Sort by final score descending
- Select top 100 candidates

---

## 6. Explainability Template

For each ranked candidate, generate structured reasoning:

```json
{
  "candidate_id": "CAND_XXXXXXX",
  "rank": 1,
  "score": 0.94,
  "components": {
    "technical_relevance": 0.92,
    "production_experience": 0.96,
    "profile_quality": 0.88,
    "behavioral_engagement": 0.89,
    "experience_level_fit": 0.95
  },
  "reasoning": {
    "strengths": [
      "7.2 years production ML experience with strong embeddings/retrieval focus",
      "Shipped semantic search system to 10M+ users at Meta, handles embedding drift & index refresh",
      "Expert-level Python with XGBoost-based ranking deployed in production",
      "Advanced proficiency in Milvus and FAISS with 40+ endorsements",
      "GitHub activity score 62 shows ongoing open-source contributions",
      "Interview completion rate 89% shows reliability"
    ],
    "concerns": [
      "Currently at TCS (consulting background), but has 4+ years at product company prior",
      "60-day notice period (acceptable, can be bought out)",
      "Limited LLM fine-tuning experience compared to preferred profile"
    ],
    "key_facts": [
      "Title: Senior ML Engineer",
      "Experience: 7.2 years",
      "Most recent role: Ranking systems at Meta (2+ years)",
      "Education: IIT (Tier 1)",
      "Verification: Email, Phone verified"
    ]
  }
}
```

**Explainability Rules**:

- ✅ NEVER hallucinate - cite specific fields (title, company, duration, description)
- ✅ MUST mention at least one technical strength from job description
- ✅ MUST mention production scale evidence
- ✅ SHOULD mention behavioral signals if relevant (activity, availability)
- ✅ SHOULD mention timeline or role consistency
- ✅ SHOULD note concerns even for highly-ranked candidates (balanced view)

---

## 7. Red Flags (Quality Signals to Monitor)

### High-Risk Profiles (Apply 0.3-0.5x multiplier):

- **Keyword Stuffing Honeypot**: 50+ skills listed, <20 endorsements total, vague role descriptions
- **Unrealistic Skill Combos**: [Photoshop, Kubernetes, Advanced NLP, Accounting] without explanation
- **Consulting-Only Career**: Every role at TCS/Infosys/Wipro with no product experience
- **Stale Profile**: Not logged in 6+ months, 0% recruiter response, but profile very complete (likely fake)
- **Timeline Chaos**: Overlapping roles, 1-month stints, unexplained 2-year gaps
- **Title-Chaser Pattern**: VP → Director → VP → VP (constantly switching titles for progression)

### Green Flags (Apply 1.05-1.1x multiplier):

- **GitHub Activity**: Commit history shows recent ML/AI contributions
- **External Validation**: Published paper, talk, blog post on relevant topics
- **Depth Signal**: 3+ years at same company in relevant role (not job-hopping)
- **Specificity**: Mentions specific systems built ("We processed 50M embeddings/day using FAISS")
- **Active Engagement**: High response rate (>70%), recent profile updates, interview completion >80%

---

## 8. Implementation Priority

1. **Phase 1** (Weeks 1-2): Build requirement parser + candidate profile analyzer
2. **Phase 2** (Weeks 2-3): Build scoring pipeline + quality checks
3. **Phase 3** (Weeks 3-4): Integrate embeddings + FAISS retrieval
4. **Phase 4** (Week 4): Fine-tune weights + explainability layer
5. **Phase 5** (Week 5): Generate final CSV + validation

---

## 9. Expected Outcome

- **Top 1-5**: Candidates with 5-8 years AI/ML + production retrieval/ranking experience + recent activity + product company background
- **Top 6-25**: Strong AI/ML + some retrieval focus OR strong production experience but less direct match
- **Top 26-100**: Mixed signals - good production depth but older experience, OR newer candidates with strong frameworks/education, OR consulting backgrounds with product stints

### Success Criteria

✅ Top 10 candidates are not pure keyword matches  
✅ Each ranked candidate has clear evidence from profile (not hallucinated)  
✅ Consulting-only backgrounds are deprioritized unless mixed with product experience  
✅ Recent activity + behavioral engagement distinguish between otherwise similar candidates  
✅ Profile quality flags catch artificial/misleading profiles

---

## Appendix: Key Distinctions from Standard Recruiting

| Aspect                    | Standard Approach       | This Strategy                                   |
| ------------------------- | ----------------------- | ----------------------------------------------- |
| **Keyword Matching**      | Count matching keywords | Understand context + weight tier                |
| **Title Matching**        | "Engineer" → high score | Parse career trajectory + company type          |
| **Skill List**            | More skills = better    | Realistic combos + endorsements matter          |
| **Recent Activity**       | Optional signal         | Critical for availability assessment            |
| **Consulting Background** | Neutral or positive     | Explicit red flag unless mixed                  |
| **Education**             | Tier 1 must-have        | Helpful but not decisive with strong experience |
| **Notice Period**         | Rarely considered       | Major factor for hiring speed                   |
| **Research vs Shipping**  | Both equally valued     | Shipping + product mindset heavily weighted     |
