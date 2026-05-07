from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.placement_model.predict import get_model_metadata, predict_placement
from src.recommender.job_recommender import JOB_DB, recommend_jobs
from src.resume_analyzer.parser import extract_text_from_pdf
from src.resume_analyzer.skill_extractor import extract_skills, get_resume_score, missing_skills
from src.utils.preprocessing import PROCESSED_DATA_PATH, prepare_placement_dataset, summarize_placement_dataset

st.set_page_config(
    page_title="Placement Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
        }
        .hero {
            background: linear-gradient(135deg, #0f2740 0%, #135c8d 100%);
            padding: 1.8rem 2rem;
            border-radius: 24px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 20px 50px rgba(15, 39, 64, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
        }
        .hero p {
            margin: 0.8rem 0 0 0;
            color: rgba(255,255,255,0.88);
            line-height: 1.6;
        }
        .note-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.8rem;
        }
        .pill-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.4rem;
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0f4c81;
            font-size: 0.83rem;
            font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return prepare_placement_dataset()


@st.cache_data(show_spinner=False)
def load_training_metadata() -> dict[str, object]:
    return get_model_metadata()


def render_pills(items: list[str]) -> None:
    if not items:
        st.caption("No items available.")
        return

    markup = "".join(f"<span class='pill'>{item.title()}</span>" for item in items)
    st.markdown(f"<div class='pill-wrap'>{markup}</div>", unsafe_allow_html=True)


def readiness_message(score: float) -> tuple[str, str]:
    if score >= 75:
        return "High readiness", "Your profile is performing strongly against the training data benchmark."
    if score >= 50:
        return "Moderate readiness", "You are in a workable range, but stronger screening signals would improve confidence."
    return "Needs improvement", "The model sees this profile as high risk. Focus on academics and aptitude performance first."


def parse_training_time(raw_value: str | None) -> str:
    if not raw_value:
        return "Unavailable"

    try:
        timestamp = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        return timestamp.strftime("%d %b %Y, %I:%M %p UTC")
    except ValueError:
        return raw_value


def build_job_match_table(skills: list[str]) -> pd.DataFrame:
    rows = []
    skill_set = set(skills)

    for role, required_skills in JOB_DB.items():
        required_set = set(required_skills)
        matched = sorted(skill_set & required_set)
        missing = sorted(required_set - skill_set)
        match_score = round((len(matched) / len(required_set)) * 100) if required_set else 0

        rows.append(
            {
                "Role": role,
                "Match Score (%)": match_score,
                "Matched Skills": ", ".join(matched) if matched else "None",
                "Missing Skills": ", ".join(missing) if missing else "None",
            }
        )

    return pd.DataFrame(rows).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)


apply_theme()

processed_df = load_processed_data()
dataset_summary = summarize_placement_dataset(processed_df)
training_metadata = load_training_metadata()
test_metrics = training_metadata.get("test_metrics", {})
leaderboard = pd.DataFrame(training_metadata.get("leaderboard", []))
cohort_profile = pd.DataFrame()

if not processed_df.empty:
    cohort_profile = (
        processed_df.groupby("placed")[["cgpa", "placement_exam_marks"]]
        .mean()
        .rename(index={0: "Not Placed", 1: "Placed"})
        .T
    )

st.markdown(
    f"""
    <div class="hero">
        <h1>Placement Intelligence Dashboard</h1>
        <p>
            Cleaned data from <strong>data/raw/placement.csv</strong>, trained a fresh placement model,
            and surfaced both the dataset and model performance directly inside Streamlit.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline Status")
    st.caption("Source: `data/raw/placement.csv`")
    st.caption(f"Processed file: `{PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)}`")
    st.caption(f"Rows after cleaning: {dataset_summary['sample_size']}")
    st.caption(f"Placement rate: {dataset_summary['placement_rate']:.2f}%")
    st.caption(f"Last training run: {parse_training_time(training_metadata.get('trained_at_utc'))}")

    st.divider()
    st.subheader("Model")
    st.caption(f"Selected model: {training_metadata.get('best_model', 'Unavailable')}")
    if test_metrics:
        st.caption(f"Accuracy: {test_metrics.get('accuracy', 0) * 100:.2f}%")
        st.caption(f"ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")

overview_tab, predict_tab, resume_tab, jobs_tab = st.tabs(
    [
        "Overview",
        "Predict Placement",
        "Resume Analyzer",
        "Job Recommender",
    ]
)

with overview_tab:
    st.subheader("Cleaned Dataset and Model Snapshot")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Cleaned Records", f"{dataset_summary['sample_size']}")
    metric_cols[1].metric("Placement Rate", f"{dataset_summary['placement_rate']:.2f}%")
    metric_cols[2].metric("Average CGPA", f"{dataset_summary['average_cgpa']:.2f}")
    metric_cols[3].metric("Average Exam Score", f"{dataset_summary['average_exam_marks']:.2f}")

    training_cols = st.columns(4)
    training_cols[0].metric("Best Model", training_metadata.get("best_model", "Unavailable"))
    training_cols[1].metric(
        "Test Accuracy",
        f"{test_metrics.get('accuracy', 0) * 100:.2f}%" if test_metrics else "Unavailable",
    )
    training_cols[2].metric(
        "Test F1 Score",
        f"{test_metrics.get('f1', 0) * 100:.2f}%" if test_metrics else "Unavailable",
    )
    training_cols[3].metric(
        "ROC AUC",
        f"{test_metrics.get('roc_auc', 0):.4f}" if test_metrics else "Unavailable",
    )

    if test_metrics and test_metrics.get("roc_auc", 0) < 0.55:
        st.warning(
            "The pipeline is working, but this dataset has weak predictive signal with the current two input columns. "
            "Use the model as a demo/benchmark, not as a high-confidence production predictor."
        )

    chart_col, notes_col = st.columns([1.2, 0.8], gap="large")

    with chart_col:
        st.markdown("#### Placed vs Not Placed Feature Averages")
        if cohort_profile.empty:
            st.info("Cohort chart unavailable because the processed dataset is empty.")
        else:
            st.bar_chart(cohort_profile)

    with notes_col:
        st.markdown("#### Training Notes")
        st.markdown(
            """
            <div class="note-card">
                The preprocessing step standardizes columns, converts the values to numeric,
                drops duplicates and missing rows, and keeps only valid CGPA, exam-score, and target ranges.
            </div>
            <div class="note-card">
                The training pipeline compares Logistic Regression and Random Forest, then saves the stronger model using ROC AUC as the selection metric.
            </div>
            <div class="note-card">
                The same saved artifact is used for the live prediction tab, so the dashboard and backend stay consistent.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Model Leaderboard")
    if leaderboard.empty:
        st.warning("Training metadata was not found. Run the training script to populate leaderboard metrics.")
    else:
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)

    st.markdown("#### Processed Dataset Preview")
    st.dataframe(processed_df.head(20), hide_index=True, use_container_width=True)

with predict_tab:
    st.subheader("Live Placement Prediction")
    st.caption("This form uses the same two features that were trained from the cleaned raw CSV.")

    with st.form("placement_prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

        with form_col2:
            placement_exam_marks = st.slider(
                "Placement Exam Marks",
                min_value=0,
                max_value=100,
                value=40,
                step=1,
            )

        submitted = st.form_submit_button("Predict Placement Probability", use_container_width=True)

    if submitted:
        probability = predict_placement([cgpa, placement_exam_marks])
        band, message = readiness_message(probability)

        result_cols = st.columns(4)
        result_cols[0].metric("Placement Probability", f"{probability:.2f}%")
        result_cols[1].metric("Readiness Band", band)
        result_cols[2].metric("CGPA vs Avg", f"{cgpa:.2f}", f"{cgpa - dataset_summary['average_cgpa']:+.2f}")
        result_cols[3].metric(
            "Exam Score vs Avg",
            f"{placement_exam_marks}",
            f"{placement_exam_marks - dataset_summary['average_exam_marks']:+.2f}",
        )

        st.progress(min(probability / 100, 1.0))

        if probability >= 75:
            st.success(message)
        elif probability >= 50:
            st.warning(message)
        else:
            st.error(message)

        comparison_df = pd.DataFrame(
            {
                "Your Input": [cgpa, placement_exam_marks],
                "Training Average": [
                    dataset_summary["average_cgpa"],
                    dataset_summary["average_exam_marks"],
                ],
            },
            index=["CGPA", "Placement Exam Marks"],
        )
        st.bar_chart(comparison_df)

with resume_tab:
    st.subheader("Resume Analyzer")
    st.caption("Upload a PDF resume to extract skills, estimate profile strength, and connect that skill set to likely roles.")

    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_resume is not None:
        resume_text = extract_text_from_pdf(uploaded_resume)
        detected_skills = sorted(extract_skills(resume_text))
        resume_score = get_resume_score(detected_skills)
        role_table = build_job_match_table(detected_skills)

        score_cols = st.columns(3)
        score_cols[0].metric("Detected Skills", len(detected_skills))
        score_cols[1].metric("Resume Score", f"{resume_score}/100")
        score_cols[2].metric("Top Role Match", role_table.iloc[0]["Role"] if not role_table.empty else "Unavailable")

        st.progress(min(resume_score / 100, 1.0))
        st.markdown("#### Skills Detected")
        render_pills(detected_skills)

        st.markdown("#### Missing Skills From Current Skill Library")
        render_pills(sorted(missing_skills(detected_skills))[:8])

        st.markdown("#### Role Alignment")
        st.dataframe(role_table, hide_index=True, use_container_width=True)

        recommended_roles = recommend_jobs(detected_skills)
        st.markdown("#### Recommended Roles")
        if recommended_roles == ["No strong match found"]:
            st.info("No strong role match found yet. Add more role-specific tools and keywords to the resume.")
        else:
            render_pills(recommended_roles)

        with st.expander("Preview Extracted Resume Text"):
            st.write(resume_text[:1500] if resume_text else "No text could be extracted from the uploaded file.")

with jobs_tab:
    st.subheader("Manual Job Recommendation")
    st.caption("Enter skills manually to see role matches, missing capabilities, and a quick shortlist from the recommender.")

    with st.form("job_recommender_form"):
        user_skills_text = st.text_area(
            "Enter your skills (comma separated)",
            placeholder="python, sql, power bi, machine learning",
            height=120,
        )
        jobs_submitted = st.form_submit_button("Generate Role Matches", use_container_width=True)

    if jobs_submitted:
        user_skills = [skill.strip().lower() for skill in user_skills_text.split(",") if skill.strip()]

        if not user_skills:
            st.warning("Enter at least one skill to generate recommendations.")
        else:
            role_table = build_job_match_table(user_skills)
            recommendations = recommend_jobs(user_skills)

            match_cols = st.columns(3)
            match_cols[0].metric("Skills Submitted", len(user_skills))
            match_cols[1].metric("Best Match", role_table.iloc[0]["Role"] if not role_table.empty else "Unavailable")
            match_cols[2].metric(
                "Best Match Score",
                f"{role_table.iloc[0]['Match Score (%)']}%" if not role_table.empty else "0%",
            )

            st.markdown("#### Your Skill Stack")
            render_pills(user_skills)

            st.markdown("#### Role Match Table")
            st.dataframe(role_table, hide_index=True, use_container_width=True)

            st.markdown("#### Recommended Jobs")
            if recommendations == ["No strong match found"]:
                st.info("No strong job match found yet. The match table above shows which skills to add next.")
            else:
                render_pills(recommendations)
