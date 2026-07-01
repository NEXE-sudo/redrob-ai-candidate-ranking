import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🤖", layout="wide")

st.title("Redrob Candidate Ranker")
st.write("This Space hosts a lightweight sandbox for the candidate ranking pipeline.")

sample_path = ROOT / "sample_submission.csv"
output_path = ROOT / "ranking_output" / "submission.csv"

if output_path.exists():
    display_path = output_path
else:
    display_path = sample_path

st.subheader("Preview")
df = pd.read_csv(display_path)
st.dataframe(df.head(10), use_container_width=True)

st.caption(f"Showing rows from {display_path.name}.")

if not output_path.exists():
    st.info("The full ranking pipeline is available in the repository, while this Space serves a lightweight preview until the full local assets are present.")
