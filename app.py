import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
OUTPUT_DIR = ROOT / "ranking_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🤖", layout="wide")

st.title("Redrob Candidate Ranker")
st.write(
    "Upload a small candidates JSONL file (up to 100 candidates) to run the ranking pipeline end to end and download the ranked CSV."
)

uploaded_file = st.file_uploader("Upload candidates JSONL", type=["jsonl"])

if uploaded_file is None:
    st.info("Upload a .jsonl candidates file to begin.")
else:
    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = [line for line in raw_text.splitlines() if line.strip()]
    candidate_count = len(lines)

    st.caption(f"Detected {candidate_count} candidate record(s) in the uploaded file.")

    if candidate_count > 100:
        st.error("The sandbox supports up to 100 candidates. Please upload a smaller file.")
    else:
        if st.button("Run Ranking"):
            with st.spinner("Running the ranking pipeline..."):
                temp_root = ROOT / "tmp"
                temp_root.mkdir(parents=True, exist_ok=True)
                temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_", dir=str(temp_root)))
                input_path = temp_dir / f"{uuid.uuid4().hex}.jsonl"
                input_path.write_text(raw_text, encoding="utf-8")
                output_path = OUTPUT_DIR / "submission.csv"

                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(BACKEND_DIR / "rank.py"),
                            "--candidates",
                            str(input_path),
                            "--out",
                            str(output_path),
                        ],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    st.error("The ranking run timed out. Please try a smaller file.")
                    st.stop()
                except Exception as exc:
                    st.error(f"The ranking run failed: {exc}")
                    st.stop()

                if result.returncode != 0:
                    st.error("Ranking failed. See the terminal output below for details.")
                    st.code(result.stderr or result.stdout)
                elif not output_path.exists():
                    st.error("The ranking completed but no CSV was produced.")
                else:
                    try:
                        df = pd.read_csv(output_path)
                    except Exception as exc:
                        st.error(f"The generated CSV could not be read: {exc}")
                    else:
                        st.success("Ranking completed successfully.")
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download ranked CSV",
                            data=output_path.read_bytes(),
                            file_name="submission.csv",
                            mime="text/csv",
                        )

                        if result.stdout:
                            st.caption("Ranking command output:")
                            st.code(result.stdout[-4000:])
