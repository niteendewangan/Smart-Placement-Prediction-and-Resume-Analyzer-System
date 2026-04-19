import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


#Streamlit App
import streamlit as st
import pandas as pd
from src.placement_model.predict import predict_placement
from src.resume_analyzer.parser import extract_text_from_pdf
from src.resume_analyzer.skill_extractor import extract_skills, get_resume_score
from src.recommender.job_recommender import recommend_jobs

st.set_page_config(page_title="Smart Placement System", layout="wide")

st.title("🎓 Smart Placement Intelligence System")

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📄 Resume Analyzer", "🎯 Job Recommender"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Placement Prediction")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        internships = st.number_input("Internships", 0, 10, 1)

    with col2:
        projects = st.number_input("Projects", 0, 10, 2)
        skills_count = st.number_input("Skills Count", 0, 20, 5)

    if st.button("Predict"):
        features = [cgpa, internships, projects, skills_count]
        result = predict_placement(features)

        st.metric("Placement Probability", f"{result}%")

        if result < 50:
            st.error("⚠️ Improve your profile!")
        else:
            st.success("✅ You are placement ready!")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Resume Analyzer")

    file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if file:
        text = extract_text_from_pdf(file)
        skills = extract_skills(text)
        score = get_resume_score(skills)

        st.write("### Skills Detected")
        st.write(skills)

        st.progress(score / 100)

        st.write(f"### Resume Score: {score}/100")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Job Recommendation")

    user_skills = st.text_input("Enter your skills (comma separated)")

    if st.button("Recommend Jobs"):
        skills_list = [s.strip().lower() for s in user_skills.split(",")]
        jobs = recommend_jobs(skills_list)

        for job in jobs:
            st.write(f"👉 {job}")
            
            
            