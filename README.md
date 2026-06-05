# 🎓 Student Placement Prediction System

An end-to-end machine learning project that predicts whether a student will be placed based on academic performance, technical skills, and experience. Includes EDA notebooks, model training pipeline, and an interactive Streamlit dashboard.

---

## ⚙️ How It Works

### 1. Data Ingestion
The raw CSV is loaded and cleaned — the `Student_ID` column is dropped, and the target column `Placement_Status` is encoded as `1` (Placed) and `0` (Not Placed).

### 2. Feature Engineering
Four new features are derived from the raw data before training:

| Feature | How It's Computed |
|---|---|
| `Career_Readiness_Score` | Weighted sum: 30% CGPA + 20% Coding + 15% Communication + 15% Aptitude + 10% Internships + 10% Certifications |
| `Academic_Strength` | CGPA + Aptitude Test Score |
| `Technical_Strength` | Coding Skills + Projects + Certifications |
| `Internship_Category` | Bucketed as None (0) / Moderate (1–2) / High (3+) |

### 3. Preprocessing Pipeline
A `ColumnTransformer` is built inside a Scikit-learn `Pipeline`:
- **Numerical features** → `SimpleImputer` (median) → `StandardScaler`
- **Categorical features** → `SimpleImputer` (most frequent) → `OneHotEncoder`

This ensures no data leakage — the pipeline is fit only on training data and applied to the test set.

### 4. Model Training & Selection
Five classifiers are trained and evaluated on an 80/20 stratified split:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |
| LightGBM | Fast gradient boosting |
| CatBoost | Handles categoricals natively |

Each model is scored on Accuracy, Precision, Recall, F1, and ROC-AUC. The model with the highest **ROC-AUC** is automatically selected, then validated with 5-fold stratified cross-validation.

### 5. Prediction
The saved pipeline accepts a raw student DataFrame, applies the same preprocessing, and returns both a binary prediction (Placed / Not Placed) and a probability score (0–100%).

### 6. Streamlit Dashboard
The user fills in a student profile via the sidebar. The app computes the engineered features in real time, calls the saved model, and renders:
- A probability gauge chart
- A horizontal skill profile bar chart
- Personalised strengths and improvement recommendations
- A downloadable plain-text report

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/placement-prediction.git
cd placement-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place `train.csv` inside:
```
data/Student Placement Dataset/train.csv
```

---

## 🔄 Workflow

### Step 1 — Exploratory Data Analysis

Open and run `eda.ipynb` to explore the dataset: distributions, correlations, outliers, and feature insights.

```bash
jupyter notebook eda.ipynb
```

### Step 2 — Model Training

Either run the notebook or the standalone script:

```bash
# Option A: Notebook
jupyter notebook model_training.ipynb

# Option B: Script
python placement.py
```

Both options will:
- Engineer features (Career Readiness Score, Academic Strength, etc.)
- Train 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Evaluate and compare all models
- Save the best model and preprocessor to `models/`

### Step 3 — Launch the Dashboard

```bash
python -m streamlit run app\streamlit_app.py
```

> `streamlit_app.py` lives inside the `app/` folder, so the path `app\streamlit_app.py` is required. Using `python -m streamlit` (instead of just `streamlit`) ensures the app runs under your active Python environment, avoiding any PATH or virtual environment mismatch issues.

Open `http://localhost:8501` in your browser.

---

## 🧠 Models Trained

| Model               | Description                              |
|---------------------|------------------------------------------|
| Logistic Regression | Linear baseline                          |
| Random Forest       | Ensemble of decision trees               |
| XGBoost             | Gradient boosted trees (high accuracy)   |
| LightGBM            | Fast gradient boosting                   |
| CatBoost            | Handles categoricals natively            |

The best model (by ROC-AUC) is automatically selected and saved.

---

## ✨ Features Used

**Raw Features:**
- Age, Gender, Degree, Branch
- CGPA, Aptitude Test Score
- Internships, Projects, Certifications
- Coding Skills, Communication Skills, Soft Skills
- Backlogs

**Engineered Features:**
- `Career_Readiness_Score` — weighted composite of key skills
- `Academic_Strength` — CGPA + Aptitude
- `Technical_Strength` — Coding + Projects + Certifications
- `Internship_Category` — bucketed internship experience (None / Moderate / High)

---

## 📊 Streamlit Dashboard

The dashboard allows you to:
- Input a student's profile via the sidebar
- View skill profile as an interactive bar chart
- Get a placement prediction with probability gauge
- See personalised strengths and improvement recommendations
- Download a plain-text placement report

---

## 📈 Evaluation Metrics

Each model is evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC *(primary selection metric)*

5-fold stratified cross-validation is also performed on the best model.

---

## 📦 Output Artifacts

| File | Description |
|------|-------------|
| `models/placement_prediction_model.pkl` | Best trained pipeline (preprocessor + model) |
| `models/preprocessor.pkl` | Standalone preprocessor |
| `models/label_encoder.pkl` | Label encoder for target variable |
| `models/model_metadata.json` | Model name, metrics, and feature lists |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** — preprocessing, pipelines, evaluation
- **XGBoost / LightGBM / CatBoost** — gradient boosting models
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisation
- **Plotly** — interactive charts in the dashboard
- **Streamlit** — web app
- **Joblib** — model serialisation

---

## 📝 License

This project is for educational purposes. Feel free to use and adapt it.
