import os
import sys
import json
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.optimized_ranking_engine import OptimizedRankingEngine
from engines.embedding_precompute import EmbeddingPrecomputer

class TestAdversarialRanking(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {
                "candidate_id": "Cand_A_Spammer",
                "profile": {
                    "summary": "embeddings retrieval FAISS vector database ranking pinecone milvus semantic search weaviate NDCG MRR MAP",
                    "headline": "ranking retrieval",
                    "current_title": "Software Engineer",
                    "years_of_experience": 5
                },
                "career_history": [
                    {
                        "title": "Software Engineer",
                        "company": "SomeCompany",
                        "description": "Used python."
                    }
                ],
                "skills": [
                    {"name": "python", "proficiency": "beginner", "duration_months": 0}
                ]
            },
            {
                "candidate_id": "Cand_B_Relevant",
                "profile": {
                    "summary": "Designed and deployed a dense retrieval pipeline using sentence-transformers and FAISS to improve semantic search latency and ranking quality.",
                    "headline": "Senior Ranking Engineer",
                    "current_title": "Senior Ranking Engineer",
                    "years_of_experience": 6,
                    "company_type": "product"
                },
                "career_history": [
                    {
                        "title": "Senior Ranking Engineer",
                        "company": "Tech Corp",
                        "description": "Led the retrieval ranking team. Shipped a real-time vector search index. Improved NDCG by 15% through offline evaluation."
                    }
                ],
                "skills": [
                    {"name": "python", "proficiency": "expert", "duration_months": 72},
                    {"name": "faiss", "proficiency": "advanced", "duration_months": 36},
                    {"name": "ranking", "proficiency": "expert", "duration_months": 48}
                ]
            },
            {
                "candidate_id": "Cand_C_Title_WeakTech",
                "profile": {
                    "summary": "I am a Staff ML Engineer. I have worked on basic regression models.",
                    "headline": "Staff ML Engineer",
                    "current_title": "Staff ML Engineer",
                    "years_of_experience": 8
                },
                "career_history": [
                    {
                        "title": "Staff ML Engineer",
                        "company": "Bank",
                        "description": "Data analysis."
                    }
                ],
                "skills": [
                    {"name": "sql", "proficiency": "expert", "duration_months": 96}
                ]
            },
            {
                "candidate_id": "Cand_D_ModTitle_StrongTech",
                "profile": {
                    "summary": "Software engineer building large scale retrieval systems, vector databases, and semantic search.",
                    "headline": "Software Engineer",
                    "current_title": "Software Engineer",
                    "years_of_experience": 5
                },
                "career_history": [
                    {
                        "title": "Software Engineer",
                        "company": "AI Startup",
                        "description": "Built semantic search using embeddings and faiss in production. Scaled to millions of queries."
                    }
                ],
                "skills": [
                    {"name": "python", "proficiency": "advanced", "duration_months": 60},
                    {"name": "embeddings", "proficiency": "expert", "duration_months": 48}
                ]
            },
            {
                "candidate_id": "Cand_E_Verbose",
                "profile": {
                    "summary": "I am a very very very very very very very good engineer. " * 20 + "I know python.",
                    "headline": "Engineer",
                    "current_title": "Engineer",
                    "years_of_experience": 3
                },
                "career_history": [
                    {
                        "title": "Engineer",
                        "company": "Comp",
                        "description": "Very very very very very very very long description about nothing." * 20
                    }
                ],
                "skills": [{"name": "python", "proficiency": "beginner", "duration_months": 12}]
            },
            {
                "candidate_id": "Cand_F_Concise",
                "profile": {
                    "summary": "Python engineer.",
                    "headline": "Engineer",
                    "current_title": "Engineer",
                    "years_of_experience": 3
                },
                "career_history": [
                    {
                        "title": "Engineer",
                        "company": "Comp",
                        "description": "Developed features."
                    }
                ],
                "skills": [{"name": "python", "proficiency": "beginner", "duration_months": 12}]
            }
        ]
        
        self.test_dir = Path(__file__).parent / "tmp_test_data"
        self.test_dir.mkdir(exist_ok=True)
        
        self.jsonl_path = self.test_dir / "adv_candidates.jsonl"
        with open(self.jsonl_path, 'w') as f:
            for c in self.candidates:
                f.write(json.dumps(c) + "\n")
                
        # Generate embeddings
        precomputer = EmbeddingPrecomputer(cache_dir=str(self.test_dir))
        precomputer.precompute_embeddings(str(self.jsonl_path))
            
        self.engine = OptimizedRankingEngine(
            embeddings_cache_dir=str(self.test_dir),
            enable_cross_encoder=False  # Disable for faster test
        )
        self.engine.load_candidates(str(self.jsonl_path))
        
        jd_text = "We need an engineer with experience in retrieval, ranking, FAISS, embeddings, and semantic search. Production experience is a plus."
        self.engine.prepare_jd_text(jd_text)

    def test_adversarial_ranking(self):
        results, scored_candidates = self.engine.rank_candidates_fast(
            top_k=6, bm25_top_k=6, faiss_top_k=6, cross_encoder_top_k=0
        )
        
        ranks = {res["candidate_id"]: res["rank"] for res in results}
        scores = {res["candidate_id"]: res["final_score"] for res in results}
        
        print("\nAdversarial Scores:")
        for r in results:
            print(f"{r['candidate_id']}: {r['final_score']:.4f}")
            
        # Verify B outranks A
        self.assertLess(ranks["Cand_B_Relevant"], ranks["Cand_A_Spammer"], "B should outrank Spammer A")
        
        # Verify D outranks C
        self.assertLess(ranks["Cand_D_ModTitle_StrongTech"], ranks["Cand_C_Title_WeakTech"], "D should outrank weak C")
        
        # Verify F is roughly competitive with E (verbosity shouldn't inflate score massively)
        diff = abs(scores["Cand_E_Verbose"] - scores["Cand_F_Concise"])
        self.assertLess(diff, 0.1, "Verbosity should not artificially inflate score significantly")

if __name__ == '__main__':
    unittest.main()
