import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🤖", layout="wide")

st.title("Redrob Candidate Ranker")
st.write("This Space hosts a lightweight sandbox for the candidate ranking pipeline.")

if st.button("Run ranking demo"):
    with st.spinner("Running the ranking pipeline..."):
        os.system(f"{sys.executable} backend/rank.py --candidates {ROOT / 'PUB India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'} --out {ROOT / 'ranking_output/submission.csv'}")
    st.success("Ranking completed. Check the output folder for the generated submission.")
