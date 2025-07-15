import os
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# === Paths ===
DATA_PATH = "Data/clean_cervical_cancer.csv"
MODEL_DIR = "Models/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# === Simplify Target ===
def simplify_action(text):
    text = text.upper()
    if "REPEAT" in text:
        return "REPEAT"
    elif "VACCINE" in text:
        return "VACCINE"
    elif any(x in text for x in ["BIOPSY", "COLPOSCOPY", "ANNUAL", "ANUAL"]):
        return "FOLLOW_UP"
    else:
        return "OTHER"

df["Recommended Action"] = df["Recommended Action"].apply(simplify_action)

# === Features and Target ===
features = [
    'Age', 'Sexual_Partners', 'First_Sexual_Activity_Age',
    'HPV_Test_Result', 'Pap_Smear_Result', 'Smoking_Status',
    'STDs_History', 'Region', 'Insurance_Covered', 'Screening_Type_Last'
]
target = "Recommended Action"
X = df[features]
y = df[target]

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# === Preprocessing ===
categorical = X.select_dtypes(include="object").columns.tolist()
numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === XGBoost Classifier with Extended Parameters ===
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# === Build Pipeline ===
model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("classifier", xgb_clf)
])

# === Fit Model ===
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=model,
    param_grid={},  # Optional: skip grid for now; can add later
    scoring="accuracy",
    cv=cv,
    verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_name = "XGBoost"
print(f"\n✅ Best model: {best_name}")

# === Evaluation ===
y_pred = best_model.predict(X_test)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test_labels, y_pred_labels))

# === Save Model ===
with open(os.path.join(MODEL_DIR, "cervical_cancer_pipeline.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("📦 Saved: cervical_cancer_pipeline.pkl and label_encoder.pkl")

# === Confusion Matrix ===
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "recommendation_confusion_matrix.png"))
plt.close()

# === Misclassified Samples ===
X_test_copy = X_test.copy()
X_test_copy["True_Label"] = y_test_labels
X_test_copy["Predicted_Label"] = y_pred_labels
misclassified = X_test_copy[X_test_copy["True_Label"] != X_test_copy["Predicted_Label"]]
misclassified.to_csv(os.path.join(MODEL_DIR, "recommendation_misclassified.csv"), index=False)

# === SHAP Summary ===
print("🔍 Generating SHAP explainability...")
model_final = best_model.named_steps["classifier"]
X_train_processed = best_model.named_steps["preprocessor"].transform(X_train)
explainer = shap.TreeExplainer(model_final)
shap_values = explainer.shap_values(X_train_processed)

shap.summary_plot(shap_values, X_train_processed, show=False)
plt.title("SHAP Summary (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "recommendation_shap_summary.png"))
plt.close()
print("✅ SHAP summary plot saved.")
