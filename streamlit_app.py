import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------
# Custom CSS
# -------------------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.metric-box {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------
# Load Model
# -------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "placement_prediction_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------------------
# Header
# -------------------------------------
st.title("🎓 AI Student Placement Prediction Dashboard")
st.caption("Predict placement opportunities using academic performance, skills, and experience.")

# -------------------------------------
# Sidebar Inputs
# -------------------------------------
with st.sidebar:
    st.header("📋 Student Profile")

    age = st.number_input("Age", min_value=18, max_value=30, value=21)
    gender = st.selectbox("Gender", ["Male", "Female"])
    degree = st.selectbox("Degree", ["B.Tech", "B.E", "B.Sc", "M.Tech", "M.Sc"])
    branch = st.selectbox("Branch", ["CSE", "AI", "ECE", "Mechanical", "Civil"])
    
    cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)
    aptitude = st.slider("Aptitude Score", 0, 100, 70)
    internships = st.number_input("Internships", 0, 10, 1)
    projects = st.number_input("Projects", 0, 20, 2)
    certifications = st.number_input("Certifications", 0, 20, 1)
    
    coding = st.slider("Coding Skills", 0, 100, 70)
    communication = st.slider("Communication Skills", 0, 100, 70)
    soft_skills = st.slider("Soft Skills", 0, 100, 70)
    
    backlogs = st.number_input("Backlogs", 0, 20, 0)

    predict_btn = st.button("🚀 Predict Placement", use_container_width=True)

# -------------------------------------
# Feature Engineering
# -------------------------------------
career_readiness = (
    0.30 * cgpa +
    0.20 * coding +
    0.15 * communication +
    0.15 * aptitude +
    0.10 * internships +
    0.10 * certifications
)

academic_strength = cgpa + aptitude
technical_strength = coding + projects + certifications

if internships == 0:
    internship_category = "None"
elif internships <= 2:
    internship_category = "Moderate"
else:
    internship_category = "High"

# -------------------------------------
# Dashboard Metrics
# -------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Career Readiness", f"{career_readiness:.1f}")
c2.metric("Academic Strength", f"{academic_strength:.1f}")
c3.metric("Technical Strength", f"{technical_strength:.1f}")

# -------------------------------------
# Skill Visualization - Horizontal Bar Chart
# -------------------------------------
viz_col, info_col = st.columns([1.2, 1])

with viz_col:
    st.subheader("📊 Skill Profile")

    skills = {
        "Coding": coding,
        "Communication": communication,
        "Aptitude": aptitude,
        "Soft Skills": soft_skills,
        "CGPA (Scaled)": cgpa * 10
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=list(skills.keys()),
        x=list(skills.values()),
        orientation='h',
        marker=dict(
            color=['#00ff88', '#00ccff', '#ff8800', '#ff44aa', '#aa44ff'],
            line=dict(color='white', width=1)
        ),
        text=[f"{v:.1f}" for v in skills.values()],
        textposition='outside',
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            title="Score (out of 100)",
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=14)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with info_col:
    st.subheader("📈 Detailed Breakdown")
    for skill, value in skills.items():
        st.write(f"**{skill}**")
        st.progress(value / 100)
        st.caption(f"{value:.1f}/100")
        st.markdown("---")

# -------------------------------------
# Prediction
# -------------------------------------
if predict_btn:
    student_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Degree": [degree],
        "Branch": [branch],
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Coding_Skills": [coding],
        "Communication_Skills": [communication],
        "Aptitude_Test_Score": [aptitude],
        "Soft_Skills_Rating": [soft_skills],
        "Certifications": [certifications],
        "Backlogs": [backlogs],
        "Career_Readiness_Score": [career_readiness],
        "Academic_Strength": [academic_strength],
        "Technical_Strength": [technical_strength],
        "Internship_Category": [internship_category]
    })

    try:
        prediction = model.predict(student_data)[0]
        probability = None
        try:
            probability = model.predict_proba(student_data)[0][1] * 100
        except:
            pass

        st.divider()

        if probability is not None:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Placement Probability (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

        if prediction == 1:
            st.success("🎉 Student is likely to be **PLACED**")
        else:
            st.error("⚠️ Student is likely to be **NOT PLACED**")

        # Strengths & Improvements
        strengths = []
        improvements = []

        if coding >= 80:
            strengths.append("💻 Strong programming skills")
        if cgpa >= 8.0:
            strengths.append("🎓 Excellent academic performance")
        if communication >= 75:
            strengths.append("🗣️ Strong communication skills")
        if internships > 0:
            strengths.append("🏢 Practical industry exposure")
        if projects >= 3:
            strengths.append("🚀 Good project portfolio")
        if certifications >= 2:
            strengths.append("📜 Strong certification profile")

        if coding < 85:
            improvements.append(f"Improve coding skills from {coding} to 85+")
        if communication < 80:
            improvements.append("Strengthen communication and interview skills")
        if aptitude < 80:
            improvements.append("Practice aptitude and logical reasoning regularly")
        if internships < 2:
            improvements.append("Gain at least 2 internships before placements")
        if projects < 4:
            improvements.append("Build more real-world projects")
        if certifications < 3:
            improvements.append("Earn additional certifications")
        if backlogs > 0:
            improvements.append(f"Clear {backlogs} backlog(s)")

        if not improvements:
            improvements.extend([
                "Participate in national-level hackathons",
                "Build a personal portfolio website",
                "Contribute to open-source projects",
                "Prepare for product-based company interviews"
            ])

        colA, colB = st.columns(2)

        with colA:
            st.subheader("💪 Strengths")
            for item in strengths:
                st.success(item)
            if not strengths:
                st.info("No major strengths identified.")

        with colB:
            st.subheader("🎯 Recommendations")
            for item in improvements:
                st.warning(item)

        # Download Report
        report = f"""Placement Prediction Report
================================

Prediction: {"Placed" if prediction == 1 else "Not Placed"}
Probability: {probability:.1f}% 

Student Details:
- CGPA: {cgpa}
- Coding Skills: {coding}
- Communication: {communication}
- Aptitude: {aptitude}
- Internships: {internships}
- Projects: {projects}
- Career Readiness Score: {career_readiness:.2f}
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="placement_report.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")