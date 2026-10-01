

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==========================================
# CONFIGURATION
# ==========================================

# Use pathlib for cross-platform compatibility
from pathlib import Path

DATA_PATH = Path("data") / "Student Placement Dataset" / "train.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "placement_prediction_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData Info:")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")

# ==========================================
# DATA CLEANING
# ==========================================

print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Drop Student_ID as it's not a feature
df.drop("Student_ID", axis=1, inplace=True)

# Encode target variable
df["Placement_Status"] = df["Placement_Status"].map({
    "Placed": 1,
    "Not Placed": 0
})

print(f"Target variable distribution:")
print(df["Placement_Status"].value_counts())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Career Readiness Score - weighted combination
df["Career_Readiness_Score"] = (
    0.30 * df["CGPA"]
    + 0.20 * df["Coding_Skills"]
    + 0.15 * df["Communication_Skills"]
    + 0.15 * df["Aptitude_Test_Score"]
    + 0.10 * df["Internships"]
    + 0.10 * df["Certifications"]
)

# Academic Strength - combination of academics
df["Academic_Strength"] = (
    df["CGPA"] +
    df["Aptitude_Test_Score"]
)

# Technical Strength - combination of technical skills
df["Technical_Strength"] = (
    df["Coding_Skills"] +
    df["Projects"] +
    df["Certifications"]
)

# Internship Category - categorical bucketing
def categorize_internships(x):
    """Categorize internship experience"""
    if x == 0:
        return "None"
    elif x <= 2:
        return "Moderate"
    else:
        return "High"

df["Internship_Category"] = df["Internships"].apply(categorize_internships)

print("New features created:")
print(df[["Career_Readiness_Score", "Academic_Strength", 
          "Technical_Strength", "Internship_Category"]].head())

# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Placement Distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    x="Placement_Status",
    data=df,
    palette="Set2"
)
plt.title("Placement Status Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Placement Status (0: Not Placed, 1: Placed)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(MODELS_DIR / "placement_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# CGPA Distribution
plt.figure(figsize=(10, 5))
sns.histplot(
    data=df,
    x="CGPA",
    kde=True,
    hue="Placement_Status",
    palette="Set2"
)
plt.title("CGPA Distribution by Placement Status", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MODELS_DIR / "cgpa_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)
plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MODELS_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# PREPARE DATA FOR MODELING
# ==========================================

print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"\nFeatures: {list(X.columns)}")

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("\n" + "="*50)
print("TRAIN TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining set placement rate: {y_train.mean():.2%}")
print(f"Test set placement rate: {y_test.mean():.2%}")

# ==========================================
# DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=RANDOM_STATE
    )
}

# ==========================================
# TRAIN & EVALUATE MODELS
# ==========================================

print("\n" + "="*50)
print("MODEL TRAINING & EVALUATION")
print("="*50)

results = []
best_pipeline = None
best_auc = 0
pipelines = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    pipelines[name] = pipeline
    
    # Make predictions
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })
    
    print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {auc:.4f}")
    
    # Track best model
    if auc > best_auc:
        best_auc = auc
        best_pipeline = pipeline
        best_model_name = name

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ROC_AUC", ascending=False)

print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)
print(results_df.to_string(index=False))

# ==========================================
# BEST MODEL ANALYSIS
# ==========================================

print("\n" + "="*50)
print(f"BEST MODEL: {best_model_name}")
print("="*50)

preds = best_pipeline.predict(X_test)
probs = best_pipeline.predict_proba(X_test)[:, 1]

print(f"\nBest ROC-AUC: {best_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Not Placed", "Placed"]))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight='bold')
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {best_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# MODEL COMPARISON VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.15

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])

plt.xlabel("Models", fontsize=12, fontweight='bold')
plt.ylabel("Score", fontsize=12, fontweight='bold')
plt.title("Model Performance Comparison", fontsize=14, fontweight='bold')
plt.xticks(x + width * 2, results_df["Model"], rotation=45, ha='right')
plt.legend(fontsize=10)
plt.ylim([0, 1.05])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# FEATURE IMPORTANCE (for tree-based models)
# ==========================================

if best_model_name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, columns in best_pipeline.named_steps['preprocessor'].transformers_:
        if name == 'num':
            feature_names.extend(columns)
        else:
            # Get one-hot encoded feature names
            categories = transformer.categories_
            for i, col in enumerate(columns):
                for cat in categories[i]:
                    feature_names.append(f"{col}_{cat}")
    
    model = best_pipeline.named_steps['model']
    importances = model.feature_importances_
    
    # Get top 15 features
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(10, 8))
    plt.barh([feature_names[i] for i in indices], importances[indices], color='steelblue')
    plt.xlabel("Importance", fontsize=12, fontweight='bold')
    plt.title(f"Top 15 Feature Importances - {best_model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# SAVE MODEL & PREPROCESSOR
# ==========================================

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

joblib.dump(best_pipeline, MODEL_PATH)
print(f"✅ Model saved: {MODEL_PATH}")

# Also save just the preprocessor for reference
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"✅ Preprocessor saved: {PREPROCESSOR_PATH}")

# Save model metadata
metadata = {
    "model_name": best_model_name,
    "roc_auc": best_auc,
    "accuracy": accuracy_score(y_test, preds),
    "features": list(X.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

import json
with open(MODELS_DIR / "model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"✅ Metadata saved: {MODELS_DIR / 'model_metadata.json'}")

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_student(student_df):
    """
    Make placement prediction for a student
    
    Parameters:
    -----------
    student_df : pd.DataFrame
        DataFrame with student features
        
    Returns:
    --------
    tuple : (prediction, probability)
        prediction: 0 (Not Placed) or 1 (Placed)
        probability: confidence percentage (0-100)
    """
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(student_df)[0]
        probability = model.predict_proba(student_df)[0][1] * 100
        return prediction, probability
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

# Test the prediction function
print("\n" + "="*50)
print("TESTING PREDICTION FUNCTION")
print("="*50)

test_student = X.iloc[0:1]
pred, prob = predict_student(test_student)

if pred is not None:
    status = "Placed" if pred == 1 else "Not Placed"
    print(f"\nTest Prediction: {status}")
    print(f"Confidence: {prob:.2f}%")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
