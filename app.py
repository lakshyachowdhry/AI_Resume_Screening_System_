from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from evaluator import evaluate_model
from parser import extract_text_from_pdf
from preprocess import preprocess_text
from similarity import (
    compute_bert_similarity,
    compute_job_resume_similarity,
    compute_resume_to_resume_similarity,
)
from genai_helper import (
    configure_gemini,
    generate_candidate_analysis,
    generate_interview_questions,
    generate_email_draft,
)
from skills import analyze_skill_match, skill_sets_to_strings
from utils import (
    MODEL_DIR,
    compute_final_score,
    dataframe_to_csv_download,
    log_event,
    safe_load_joblib,
    setup_logging,
)

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def load_model_and_vectorizer():
    return safe_load_joblib(MODEL_PATH), safe_load_joblib(VECTORIZER_PATH)


def summarize_resume(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())
    return text[:max_chars] + "..." if len(text) > max_chars else text


def main():
    setup_logging()
    st.set_page_config(page_title="AI Resume Screening System", layout="wide")

    st.title("AI Resume Screening System")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Job Configuration")

        job_role = st.selectbox(
            "Job Role",
            ["Custom", "Data Scientist", "AI/ML Engineer", "Data Analyst"],
        )

        if job_role == "Data Scientist":
            default_jd = """
Looking for a Data Scientist with:
- Python, Machine Learning, SQL
- Experience with scikit-learn, pandas, numpy
- Data visualization skills (matplotlib/seaborn)
"""
        elif job_role == "AI/ML Engineer":
            default_jd = """
Looking for an AI/ML Engineer with:
- Deep Learning (TensorFlow/PyTorch)
- Model deployment experience
- Strong Python and system design
"""
        elif job_role == "Data Analyst":
            default_jd = """
Looking for a Data Analyst with:
- SQL, Excel, Tableau/Power BI
- Strong analytical and visualization skills
"""
        else:
            default_jd = ""

        # ✅ SESSION FIX
        if "job_description" not in st.session_state:
            st.session_state.job_description = default_jd

        if job_role != "Custom":
            st.session_state.job_description = default_jd

        job_description = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            key="job_description",
            height=200,
        )

        st.markdown("---")

        # -------- GEMINI --------
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            configure_gemini(api_key)

        st.markdown("---")

        if st.button("Run Evaluation"):
            try:
                st.write(evaluate_model())
            except Exception as e:
                st.error(e)

    # ---------------- MAIN ----------------
    files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

    if st.button("Analyze Candidates"):

        if not job_description.strip():
            st.error("Enter job description")
            return

        if not files:
            st.error("Upload resumes")
            return

        names, texts, summaries = [], [], []

        for f in files:
            txt = extract_text_from_pdf(f)
            if txt:
                names.append(f.name)
                texts.append(txt)
                summaries.append(summarize_resume(txt))

        # -------- SIMILARITY --------
        tfidf_scores = compute_job_resume_similarity(job_description, texts)
        bert_scores, _ = compute_bert_similarity(job_description, texts)

        # -------- SKILLS --------
        skills = [analyze_skill_match(texts[i], job_description) for i in range(len(texts))]

        results = []

        for i in range(len(texts)):
            combined = (tfidf_scores[i] + bert_scores[i]) / 2 * 100

            results.append({
                "Rank": i + 1,  # will fix after sorting
                "Candidate": names[i],
                "TF-IDF %": round(tfidf_scores[i] * 100, 2),
                "BERT %": round(bert_scores[i] * 100, 2),
                "Skills %": skills[i].match_score,
                "Final %": round(combined, 2),
                "Summary": summaries[i],
                "Skills Found": ", ".join(skill_sets_to_strings(skills[i].detected_skills)),
                "Missing Skills": ", ".join(skill_sets_to_strings(skills[i].missing_skills)),
                "Raw": texts[i],
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Final %", ascending=False).reset_index(drop=True)

        # ✅ FIX RANK AFTER SORT
        df["Rank"] = df.index + 1

        # -------- TOP CANDIDATE --------
        st.subheader("Top Candidate")
        st.success(f"{df.iloc[0]['Candidate']} — {df.iloc[0]['Final %']}%")

        # -------- TABLE --------
        st.subheader("Ranking Table")
        st.dataframe(df[["Rank", "Candidate", "TF-IDF %", "BERT %", "Skills %", "Final %"]])

        # -------- GRAPH --------
        st.subheader("Score Distribution")
        st.bar_chart(df.set_index("Candidate")["Final %"])

        # -------- DETAILS --------
        st.subheader("Detailed Analysis")

        for _, row in df.iterrows():
            with st.expander(f"#{row['Rank']} {row['Candidate']} — {row['Final %']}%"):

                st.write("Summary:", row["Summary"])
                st.write("Skills Found:", row["Skills Found"])
                st.write("Missing Skills:", row["Missing Skills"])

                st.write(f"TF-IDF: {row['TF-IDF %']}%")
                st.write(f"BERT: {row['BERT %']}%")
                st.write(f"Skills Score: {row['Skills %']}%")

                # -------- GEN AI --------
                if api_key:
                    if st.button("Generate Analysis", key=f"a{row['Candidate']}"):
                        st.write(generate_candidate_analysis(row["Raw"], job_description))

                    if st.button("Interview Questions", key=f"q{row['Candidate']}"):
                        st.write(generate_interview_questions(row["Raw"], job_description))

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Acceptance Email", key=f"acc{row['Candidate']}"):
                            st.write(generate_email_draft(row["Candidate"], row["Raw"], job_description, "accept"))

                    with col2:
                        if st.button("Rejection Email", key=f"rej{row['Candidate']}"):
                            st.write(generate_email_draft(row["Candidate"], row["Raw"], job_description, "reject"))

        # -------- CSV --------
        st.subheader("Download Results")
        st.download_button(
            "Download CSV",
            dataframe_to_csv_download(df),
            "results.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()