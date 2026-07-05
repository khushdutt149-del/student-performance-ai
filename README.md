# 🎓 AI Student Performance Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end **Machine Learning web application** that predicts a student's
final academic grade, quantifies academic risk, and generates personalized,
AI-driven recommendations — built with a dark-themed, glassmorphism dashboard UI.

---

## ✨ Features

- 🎯 **Prediction Engine** — 32-feature Random Forest pipeline predicts final grade (0–20 scale)
- 📊 **Interactive Dashboard** — gauge charts, radar charts, pie charts, and bar charts (Plotly)
- 🧠 **AI Recommendations** — 8–10 dynamic, rule-based suggestions generated per student profile
- 📄 **PDF Reports** — professional, downloadable reports via ReportLab
- 📜 **Prediction History** — in-session table with CSV export and delete option
- 🎨 **Modern UI** — dark theme, blue/purple gradients, glassmorphism cards, animated hover states
- 🛡 **Defensive Error Handling** — missing model file, invalid inputs, empty session state, PDF failures

---

## 🗂 Project Structure

```
Student-Performance-AI/
│
├── app.py                              # Main Streamlit entry point & routing
├── dashboard.py                        # Plotly chart builders
├── report.py                           # ReportLab PDF generation
├── utils.py                            # Categorization, recommendations, history, helpers
├── requirements.txt
├── README.md
└── student_performance_pipeline (1).pkl   # Pre-trained model (place in project root)
```

> **Note:** This build uses a CSS-based glassmorphism theme (gradients, blur,
> rounded cards) directly in `app.py` instead of static image assets — this
> keeps the app fully self-contained and avoids binary files in version control.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/Student-Performance-AI.git
cd Student-Performance-AI
pip install -r requirements.txt
```

Place your trained model file, `student_performance_pipeline (1).pkl`, in the
project root (same folder as `app.py`).

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

### Running in Google Colab

```python
!pip install streamlit pyngrok joblib plotly reportlab
!streamlit run app.py &>/content/logs.txt &

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 🧭 Pages

| Page | Description |
|---|---|
| 🏠 Home | Hero section, project stats, feature list, tech stack, pipeline overview |
| 🎯 Prediction | 4-section form → grade prediction, category, risk, recommendations |
| 📊 Dashboard | Gauges, radar, pie, and bar charts for the latest prediction |
| 📄 PDF Report | Generates and downloads a full PDF report |
| 📜 Prediction History | Session table of all predictions with CSV export |
| ℹ️ About | Dataset, model, tech stack, and future scope |

---

## 🧠 Model

- **Dataset:** UCI Student Performance dataset (395 students, 32 features)
- **Algorithm:** Random Forest Regressor
- **Target:** Final grade (`G3`), 0–20 scale
- **Pipeline:** Single Scikit-Learn `Pipeline` handling preprocessing + inference

The model is loaded once at startup via:

```python
pipeline = joblib.load("student_performance_pipeline (1).pkl")
```

---

## 🚀 Deployment

This app deploys as-is to:
- **Streamlit Community Cloud** — point it at `app.py` in your repo
- **Render / Railway / Fly.io** — use `requirements.txt` and the standard
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` start command
- **Google Colab + ngrok** — for quick demos (see above)

---

## 🔭 Future Scope

- Persist prediction history in a database (SQLite/Postgres) instead of session state
- Multi-user authentication for teachers/schools
- Batch CSV upload for whole-class predictions
- SHAP-based model explainability on the Dashboard
- Scheduled retraining pipeline on new cohort data

---

## 📄 License

MIT License — free to use, modify, and distribute.
