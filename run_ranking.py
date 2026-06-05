#!/usr/bin/env python3
"""
Main ranking script - Entry point for the ranking engine
Processes 100K candidate dataset and produces top 100 ranked candidates
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.engines.ranking_engine import RankingEngine


# Job description text for the Senior AI Engineer role
JD_TEXT = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid)

We need someone who is simultaneously comfortable with:
- Deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning
- Scrappy product-engineering attitude: willing to ship a working ranker in a week

The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates.

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar)
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python. Yes really, we care about code quality.
- Hands-on experience designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation

Things we'd like you to have but won't reject you for:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

Things we explicitly do NOT want:
- Title-chasers optimizing for seniority progression
- People who have only worked at consulting firms without product company experience
- Pure research background without production deployment
- People whose primary expertise is computer vision/speech/robotics without significant NLP/IR exposure
- Haven't written production code in the last 18 months

5-9 years experience (range, not requirement)
"""


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("REDROB AI CANDIDATE RANKING ENGINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    DATA_DIR = Path("/home/NEXE/projects/Redrob hackathon/[PUB] India_runs_data_and_ai_challenge")
    DATA_DIR = DATA_DIR / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
    CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
    OUTPUT_DIR = Path("/home/NEXE/projects/Redrob hackathon/ranking_output")
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Candidates file: {CANDIDATES_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Verify files exist
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: Candidates file not found: {CANDIDATES_FILE}")
        return 1
    
    try:
        # Initialize ranking engine
        print("\nInitializing ranking engine...")
        engine = RankingEngine(
            embedding_model='BAAI/bge-small-en-v1.5',
            use_bm25_prefilter=True,
            cache_dir='./ranking_cache'
        )
        
        # Load candidates
        print(f"\nLoading candidates from {CANDIDATES_FILE}...")
        engine.load_candidates(str(CANDIDATES_FILE))
        
        # Prepare job description
        engine.prepare_jd_text(JD_TEXT)
        
        # Parse all profiles (optional but speeds up scoring)
        print("\nPre-parsing all profiles...")
        engine.parse_all_profiles()
        
        # Execute ranking pipeline
        print("\nExecuting ranking pipeline...")
        ranked_candidates, csv_df = engine.rank_candidates(
            top_k=100,
            retrieval_top_k=3000,
            include_explainability=True
        )
        
        # Save results
        print(f"\nSaving results to {OUTPUT_DIR}...")
        engine.save_results(ranked_candidates, csv_df, str(OUTPUT_DIR))
        
        # Print top candidates summary
        print("\n" + "="*70)
        print("TOP 10 RANKED CANDIDATES")
        print("="*70)
        
        for ranked in ranked_candidates[:10]:
            print(f"\nRank #{ranked.rank}: {ranked.candidate_id} (Score: {ranked.final_score:.4f})")
            print(f"  Technical Relevance: {ranked.components.technical_relevance:.2%}")
            print(f"  Production Experience: {ranked.components.production_experience:.2%}")
            print(f"  Profile Quality: {ranked.components.profile_quality:.2%}")
            print(f"  Behavioral Engagement: {ranked.components.behavioral_engagement:.2%}")
            
            if ranked.reasoning.get('strengths'):
                print(f"  Strengths: {ranked.reasoning['strengths'][0]}")
            if ranked.reasoning.get('concerns'):
                print(f"  Concerns: {ranked.reasoning['concerns'][0]}")
        
        # Validation
        print("\n" + "="*70)
        print("VALIDATION CHECKS")
        print("="*70)
        
        # Check monotonic decreasing
        scores = [c.final_score for c in ranked_candidates]
        is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        print(f"✓ Scores monotonically decreasing: {is_monotonic}")
        print(f"✓ Total candidates ranked: {len(ranked_candidates)}")
        print(f"✓ Score range: {scores[-1]:.4f} to {scores[0]:.4f}")
        
        # Check CSV format
        print(f"✓ CSV has {len(csv_df)} rows and {len(csv_df.columns)} columns")
        print(f"✓ CSV columns: {list(csv_df.columns)}")
        
        print("\n" + "="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("RANKING COMPLETE ✓")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
