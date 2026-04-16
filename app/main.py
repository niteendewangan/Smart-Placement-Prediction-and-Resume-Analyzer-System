# Streamlit app

import streamlit as st
from src.placement_model.predict import predict_placement
from src.resume_analyzer.parser import extract_text_from_pdf
from src.resume_analyzer.skill_extractor import extract_skills, get_resume_score

st.title("🎓 Smart Placement Predictor + Resume Analyzer")

# -------------------------
# Placement Prediction
# -------------------------
st.header("📊 Placement Prediction")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
internships = st.number_input("Internships", 0, 10, 1)
projects = st.number_input("Projects", 0, 10, 2)
skills_count = st.number_input("Number of Skills", 0, 20, 5)

if st.button("Predict Placement"):
    features = [cgpa, internships, projects, skills_count]
    result = predict_placement(features)
    st.success(f"Placement Probability: {result}%")

# -------------------------
# Resume Analyzer
# -------------------------
st.header("📄 Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    skills = extract_skills(text)
    score = get_resume_score(skills)

    st.subheader("✅ Detected Skills")
    st.write(skills)

    st.subheader("📈 Resume Score")
    st.write(f"{score}/100")

    if score < 50:
        st.warning("Add more relevant skills and projects!")
    else:
        st.success("Good resume!")