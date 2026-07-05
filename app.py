"""
app.py
----------------------------------------------------
AI Student Performance Analytics
End-to-End Machine Learning Web Application (Streamlit)

Entry point. Handles page routing, styling, the prediction
form, and wiring together utils.py / dashboard.py / report.py.
----------------------------------------------------
"""

import joblib
import pandas as pd
import streamlit as st

import dashboard as dash
import report as rpt
import utils

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.init_session_state()

# ==========================================================
# Global CSS — Dark Glassmorphism, Blue/Purple Gradient Theme
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 20%, #1E1B4B 0%, #0F172A 45%, #0B1120 100%);
    color: #E2E8F0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0B1120 100%);
    border-right: 1px solid rgba(148,163,184,0.1);
}

h1, h2, h3, h4 {
    color: #F1F5F9;
    font-weight: 700;
}

.gradient-text {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 18px;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.25);
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #93C5FD;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 16px;
}

.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    transition: transform 0.2s ease;
}

.pipeline-step:hover {
    transform: scale(1.04);
    border-color: rgba(139,92,246,0.5);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px;
    padding: 16px;
}

.stButton>button {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    color: white;
    border: none;
    border-radius: 12px;
    width: 100%;
    height: 52px;
    font-size: 17px;
    font-weight: 700;
    transition: opacity 0.2s ease, transform 0.2s ease;
}

.stButton>button:hover {
    opacity: 0.9;
    transform: translateY(-2px);
}

.footer-text {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    padding-top: 24px;
    border-top: 1px solid rgba(148,163,184,0.1);
    margin-top: 32px;
}

.rec-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #8B5CF6;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14.5px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Model Loading (defensive)
# ==========================================================

pipeline, model_error = utils.safe_load_pipeline(joblib)

# ==========================================================
# Sidebar Navigation
# ==========================================================

st.sidebar.markdown("## 🎓 Student Analytics")
st.sidebar.caption("AI-Powered Performance Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎯 Prediction", "📊 Dashboard", "📄 PDF Report", "📜 Prediction History", "ℹ️ About"],
    label_visibility="collapsed",
)

st.sidebar.divider()
if model_error:
    st.sidebar.error("Model not loaded", icon="⚠️")
else:
    st.sidebar.success("Model ready", icon="✅")
st.sidebar.caption(f"Session predictions: {len(utils.get_history_df())}")


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.markdown('<span class="hero-badge">⚡ Machine Learning · Random Forest</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">AI Student Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown(
        "#### An end-to-end machine learning system that predicts student academic "
        "performance, quantifies risk, and generates personalized improvement plans."
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", "91%")
    c2.metric("Dataset Size", "395 students")
    c3.metric("Features", "32")
    c4.metric("Algorithm", "Random Forest")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("✨ Project Features")
        st.markdown("""
- ✅ Student final-grade prediction (0-20 scale)
- ✅ Automatic percentage conversion
- ✅ 5-tier performance categorization
- ✅ Academic risk scoring
- ✅ Dynamic AI-generated recommendations
- ✅ Interactive analytics dashboard
- ✅ Downloadable professional PDF reports
- ✅ Session-based prediction history with CSV export
""")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🛠 Technology Stack")
        st.markdown("""
- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Plotly
- **Reporting:** ReportLab
- **Model Persistence:** Joblib
""")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("🔄 How Prediction Works")

    steps = [
        ("1️⃣", "Input", "Student fills academic, study, and lifestyle details"),
        ("2️⃣", "Preprocessing", "Pipeline encodes categorical features automatically"),
        ("3️⃣", "Inference", "Random Forest regressor predicts final grade"),
        ("4️⃣", "Insight", "Category, risk, and recommendations are generated"),
    ]
    cols = st.columns(4)
    for col, (emoji, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="pipeline-step"><div style="font-size:28px">{emoji}</div>'
                f'<b>{title}</b><br><span style="color:#94A3B8;font-size:13px">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="footer-text">Built with Streamlit · Scikit-Learn · Plotly &nbsp;|&nbsp; '
        '© 2026 AI Student Performance Analytics</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🎯 Prediction":

    st.markdown('<h1 class="gradient-text">🎯 Student Performance Prediction</h1>', unsafe_allow_html=True)
    st.markdown("Fill in the student's details below to predict final academic performance.")

    if model_error:
        st.error(model_error, icon="🚫")

    st.divider()

    with st.form("prediction_form"):

        st.subheader("📘 Section 1 — Academic Information")
        col1, col2 = st.columns(2)
        with col1:
            school = st.selectbox("School", utils.SCHOOL_OPTIONS)
            sex = st.selectbox("Gender", utils.SEX_OPTIONS)
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", utils.ADDRESS_OPTIONS)
            famsize = st.selectbox("Family Size", utils.FAMSIZE_OPTIONS)
            Pstatus = st.selectbox("Parents Living Together", utils.PSTATUS_OPTIONS)
        with col2:
            Medu = st.slider("Mother's Education", 0, 4, 2)
            Fedu = st.slider("Father's Education", 0, 4, 2)
            Mjob = st.selectbox("Mother's Job", utils.JOB_OPTIONS)
            Fjob = st.selectbox("Father's Job", utils.JOB_OPTIONS)
            reason = st.selectbox("Reason for Choosing School", utils.REASON_OPTIONS)
            guardian = st.selectbox("Guardian", utils.GUARDIAN_OPTIONS)

        st.divider()
        st.subheader("📚 Section 2 — Study Information")
        col1, col2 = st.columns(2)
        with col1:
            traveltime = st.slider("Travel Time", 1, 4, 2)
            studytime = st.slider("Weekly Study Time", 1, 4, 2)
            failures = st.slider("Past Class Failures", 0, 4, 0)
        with col2:
            schoolsup = st.selectbox("School Support", utils.YES_NO_OPTIONS)
            famsup = st.selectbox("Family Support", utils.YES_NO_OPTIONS)
            paid = st.selectbox("Extra Paid Classes", utils.YES_NO_OPTIONS)

        st.divider()
        st.subheader("❤️ Section 3 — Lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            activities = st.selectbox("Extra Curricular Activities", utils.YES_NO_OPTIONS)
            nursery = st.selectbox("Attended Nursery School", utils.YES_NO_OPTIONS)
            higher = st.selectbox("Wants Higher Education", utils.YES_NO_OPTIONS)
            internet = st.selectbox("Internet Access at Home", utils.YES_NO_OPTIONS)
            romantic = st.selectbox("In a Romantic Relationship", utils.YES_NO_OPTIONS)
        with col2:
            famrel = st.slider("Family Relationship", 1, 5, 3)
            freetime = st.slider("Free Time After School", 1, 5, 3)
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 2)
            health = st.slider("Current Health Status", 1, 5, 3)
            absences = st.slider("Total Absences", 0, 93, 5)

        st.divider()
        st.subheader("📈 Section 4 — Previous Grades")
        col1, col2 = st.columns(2)
        with col1:
            G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
        with col2:
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

        st.divider()
        submitted = st.form_submit_button("🚀 Predict Student Performance", use_container_width=True)

    if submitted:

        if pipeline is None:
            st.error(
                "Cannot generate a prediction because the model pipeline failed to load. "
                "Please check that the .pkl file is present.",
                icon="🚫",
            )
        else:
            student_inputs = {
                "school": school, "sex": sex, "age": age, "address": address,
                "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
                "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
                "traveltime": traveltime, "studytime": studytime, "failures": failures,
                "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
                "activities": activities, "nursery": nursery, "higher": higher,
                "internet": internet, "romantic": romantic, "famrel": famrel,
                "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
                "health": health, "absences": absences, "G1": G1, "G2": G2,
            }

            try:
                student_df = pd.DataFrame({k: [v] for k, v in student_inputs.items()})
                student_df = student_df[utils.ALL_FEATURE_COLUMNS]  # enforce column order

                raw_prediction = pipeline.predict(student_df)[0]
                prediction = float(max(0.0, min(20.0, raw_prediction)))
                percentage = (prediction / 20) * 100

                meta = utils.get_performance_category(prediction)
                recommendations = utils.generate_recommendations(student_inputs, prediction)

                utils.add_to_history(student_inputs, prediction, percentage, meta)

            except KeyError as exc:
                st.error(f"Input schema mismatch — missing feature: {exc}", icon="🚫")
            except ValueError as exc:
                st.error(f"Invalid input for the model pipeline: {exc}", icon="🚫")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error during prediction: {exc}", icon="🚫")
            else:
                st.divider()
                st.header("📊 Prediction Results")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Predicted Grade", utils.format_grade(prediction))
                r2.metric("Percentage", utils.format_percentage(percentage))
                r3.metric("Performance", f"{meta['emoji']} {meta['category']}")
                r4.metric("Risk Level", meta["risk"])

                if meta["category"] == "Excellent":
                    st.balloons()
                    st.success(meta["message"], icon="🌟")
                elif meta["category"] == "Critical":
                    st.warning(meta["message"], icon="❌")
                else:
                    st.info(meta["message"], icon="ℹ️")

                st.divider()
                st.subheader("🧠 AI Recommendations")
                for rec in recommendations:
                    st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

                st.caption("Results have been saved to Prediction History and are ready for PDF export.")


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

elif page == "📊 Dashboard":

    st.markdown('<h1 class="gradient-text">📊 Performance Dashboard</h1>', unsafe_allow_html=True)

    latest = st.session_state.get("latest_prediction")

    if latest is None:
        st.info(
            "No predictions yet this session. Go to the 🎯 Prediction page to run one — "
            "the dashboard will populate automatically.",
            icon="ℹ️",
        )
    else:
        inputs = latest["inputs"]
        prediction = latest["prediction"]
        meta = latest["meta"]
        history_df = utils.get_history_df()

        top1, top2 = st.columns(2)
        with top1:
            st.plotly_chart(dash.grade_gauge(prediction, meta["color"]), use_container_width=True)
        with top2:
            st.plotly_chart(dash.risk_meter(meta["risk"]), use_container_width=True)

        mid1, mid2 = st.columns(2)
        with mid1:
            st.plotly_chart(dash.performance_pie(meta["category"]), use_container_width=True)
        with mid2:
            st.plotly_chart(dash.lifestyle_radar(inputs), use_container_width=True)

        bot1, bot2 = st.columns(2)
        with bot1:
            st.plotly_chart(dash.study_analysis_bar(inputs), use_container_width=True)
        with bot2:
            st.plotly_chart(dash.family_analysis_bar(inputs), use_container_width=True)

        st.plotly_chart(dash.attendance_chart(inputs), use_container_width=True)
        st.plotly_chart(dash.performance_timeline(history_df), use_container_width=True)


# ==========================================================
# PDF REPORT PAGE
# ==========================================================

elif page == "📄 PDF Report":

    st.markdown('<h1 class="gradient-text">📄 PDF Report Generation</h1>', unsafe_allow_html=True)

    latest = st.session_state.get("latest_prediction")

    if latest is None:
        st.info(
            "No prediction available to report on. Run a prediction from the 🎯 Prediction page first.",
            icon="ℹ️",
        )
    else:
        inputs = latest["inputs"]
        prediction = latest["prediction"]
        percentage = latest["percentage"]
        meta = latest["meta"]
        recommendations = utils.generate_recommendations(inputs, prediction)

        st.write("Preview of the report contents:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Grade", utils.format_grade(prediction))
        c2.metric("Percentage", utils.format_percentage(percentage))
        c3.metric("Performance", f"{meta['emoji']} {meta['category']}")
        c4.metric("Risk Level", meta["risk"])

        try:
            pdf_bytes = rpt.generate_pdf_report(inputs, prediction, percentage, meta, recommendations)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to generate PDF report: {exc}", icon="🚫")
        else:
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="student_performance_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ==========================================================
# PREDICTION HISTORY PAGE
# ==========================================================

elif page == "📜 Prediction History":

    st.markdown('<h1 class="gradient-text">📜 Prediction History</h1>', unsafe_allow_html=True)

    history_df = utils.get_history_df()

    if history_df.empty:
        st.info("No predictions made yet this session.", icon="ℹ️")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download History as CSV",
                data=utils.history_to_csv_bytes(),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("🗑️ Delete History", use_container_width=True):
                utils.clear_history()
                st.rerun()

    st.caption("Note: history is stored in-session only and will reset if the app restarts.")


# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.markdown('<h1 class="gradient-text">ℹ️ About This Project</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset")
    st.markdown("""
Built on the **UCI Student Performance Dataset**, containing 395 secondary-school
students with 32 academic, demographic, and lifestyle attributes, plus first-,
second-, and final-period grades.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Machine Learning")
    st.markdown("""
- **Algorithm:** Random Forest Regressor
- **Approach:** A single Scikit-Learn `Pipeline` bundles preprocessing
  (categorical encoding + scaling) and the regressor, so raw form inputs
  can be passed directly to `pipeline.predict()`.
- **Target:** Final grade (G3), on a 0-20 scale.
- **Reported accuracy:** ~91% (R² / cross-validated, per the shipped model).
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛠 Technology Stack")
    st.markdown("Python · Streamlit · Scikit-Learn · Pandas · NumPy · Plotly · Joblib · ReportLab")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔭 Future Scope")
    st.markdown("""
- Persist prediction history to a database instead of session state
- Add authentication for multi-teacher / multi-class usage
- Support batch CSV upload for whole-class predictions
- Add model explainability (SHAP) to the Dashboard
- Retrain pipeline periodically on new cohort data
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-text">AI Student Performance Analytics &nbsp;|&nbsp; '
        'Built with Streamlit, Scikit-Learn & Plotly</div>',
        unsafe_allow_html=True,
    )
