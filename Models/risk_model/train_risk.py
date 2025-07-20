import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# === Config ===
DATA_PATH = "Data/risk_factors_cervical_cancer.csv"
MODEL_DIR = "Models/saved_models"
SHAP_DIR = os.path.join(MODEL_DIR, "shap_individual")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

# === Load & Clean Data ===
data = pd.read_csv(DATA_PATH)
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
data.columns = data.columns.str.strip()
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

# === Features/Target ===
X = data.drop(columns=["Biopsy"])
y = data["Biopsy"].astype(int)

# === Feature Types ===
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

# === Preprocessing ===
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]), cat_cols)
])

# === Full Pipeline ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42))
])

# === Stratified Cross-Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipeline, X, y, cv=skf, method='predict')
y_proba = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')[:, 1]

# === Evaluation ===
print("📊 Evaluation with Stratified 5-Fold CV:")
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
print("ROC-AUC Score:", roc_auc_score(y, y_proba))

# === Confusion Matrix ===
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Biopsy", "Biopsy"],
            yticklabels=["No Biopsy", "Biopsy"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "biopsy_confusion_matrix.png"))
plt.close()

# Save classification report
report = classification_report(y, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(MODEL_DIR, "biopsy_classification_report.csv"))

# === SHAP Explainability ===
print("🔍 Generating SHAP explainability...")
pipeline.fit(X, y)
model = pipeline.named_steps["classifier"]
X_processed = pipeline.named_steps["preprocessor"].transform(X)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_processed)

# Feature naming
feature_names = num_cols + cat_cols
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Save global summary
shap.summary_plot(shap_values, X_processed_df, show=False)
plt.title("SHAP Summary (Biopsy)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "biopsy_shap_summary.png"))
plt.close()

# === Export Individual SHAP Plots ===
print("📤 Exporting SHAP force plots for individual patients...")
for i in range(len(X_processed_df)):
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[i], X_processed_df.iloc[i], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"shap_patient_{i+1}.png"))
    plt.close()

# === Export Misclassified Samples ===
misclassified = X.copy()
misclassified["True_Label"] = y
misclassified["Predicted_Label"] = y_pred
misclassified = misclassified[misclassified["True_Label"] != misclassified["Predicted_Label"]]
misclassified.to_csv(os.path.join(MODEL_DIR, "biopsy_misclassified.csv"), index=False)

# === Export Probabilities ===
proba_df = X.copy()
proba_df["Biopsy_Probability"] = y_proba
proba_df.to_csv(os.path.join(MODEL_DIR, "biopsy_probabilities.csv"), index=False)

# === Save Model ===
model_path = os.path.join(MODEL_DIR, "biopsy_pipeline.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"📦 Model saved to: {model_path}")
print("✅ SHAP summary, individual patient plots, confusion matrix, and results exported.")
