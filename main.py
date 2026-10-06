import io
import json
import os
import joblib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from openai import OpenAI

MODEL = "xgboost_model.pkl"
FEATURES = [
    "years_experience", "skills_match_score", "education_level",
    "project_count", "resume_length", "github_activity"
]

def get_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return os.getenv("DEEPSEEK_API_KEY")

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

def extract_text(file):
    data = file.getvalue()
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    return data.decode("utf-8", errors="ignore")

def parse_resume(resume, job):
    key = get_key()
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY is missing. Add it in Streamlit "
            "Settings → Secrets or as an environment variable."
        )

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    prompt = f"""
Analyze this resume against the job description.

JOB DESCRIPTION:
{job}

RESUME:
{resume[:50000]}

Return JSON only:
{{
 "candidate_name": "",
 "years_experience": 0,
 "education_level": "Bachelors",
 "project_count": 0,
 "github_activity": 0,
 "matched_skills": [],
 "missing_skills": [],
 "skills_match_score": 0,
 "reasoning": ""
}}

Rules:
- skills_match_score: 0-100.
- Never invent information.
- years_experience = professional experience.
- project_count = explicitly mentioned projects.
- github_activity = explicitly stated activity; otherwise 0.
- education_level = High School, Bachelors, Masters, PhD, or Other.
"""

    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(r.choices[0].message.content)

def main():
    st.set_page_config(
        page_title="AI Resume Screening",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 AI-Based Resume Screening")
    st.caption("DeepSeek Resume Parser + HCL XGBoost Screening Model")

    job = st.text_area("Job Description", height=220)
    file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if not job or not file:
        st.info("Enter the job description and upload a resume.")
        return

    try:
        resume = extract_text(file)

        with st.spinner("AI is analyzing the resume..."):
            parsed = parse_resume(resume, job)

            education = parsed.get("education_level", "Other")
            if education not in {"High School", "Bachelors", "Masters", "PhD", "Other"}:
                education = "Other"

            X = pd.DataFrame([{
                "years_experience": max(0, float(parsed.get("years_experience", 0))),
                "skills_match_score": min(100, max(0, float(parsed.get("skills_match_score", 0)))),
                "education_level": education,
                "project_count": max(0, float(parsed.get("project_count", 0))),
                "resume_length": len(resume.split()),
                "github_activity": max(0, float(parsed.get("github_activity", 0))),
            }])[FEATURES]

            model = load_model()
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])

        if pred:
            st.success(f"✅ SHORTLISTED — {prob:.1%} probability")
        else:
            st.error(f"❌ NOT SHORTLISTED — {(1-prob):.1%} probability")

        a, b, c = st.columns(3)
        a.metric("Skill Match", f"{X.iloc[0]['skills_match_score']:.0f}%")
        b.metric("Experience", f"{X.iloc[0]['years_experience']:.0f} yrs")
        c.metric("Projects", f"{X.iloc[0]['project_count']:.0f}")

        st.subheader("Candidate")
        st.write(f"**Name:** {parsed.get('candidate_name', 'Not found')}")
        st.write(f"**Education:** {education}")

        st.subheader("Skills")
        st.write("**Matched:** " + ", ".join(parsed.get("matched_skills", [])))
        st.write("**Missing:** " + ", ".join(parsed.get("missing_skills", [])))

        st.subheader("AI Reasoning")
        st.write(parsed.get("reasoning", ""))

        with st.expander("Features sent to XGBoost"):
            st.json(X.iloc[0].to_dict())

    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
