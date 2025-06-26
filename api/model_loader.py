import pickle
import os

# === Load Biopsy Risk Model ===
def load_biopsy_model():
    model_path = os.path.join("Models", "saved_models", "biopsy_pipeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Biopsy model not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ✅ Feature names with spaces/colons — MATCH the model training data
    features = [
        "Age", "Number of sexual partners", "First sexual intercourse",
        "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)",
        "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
        "IUD", "IUD (years)", "STDs", "STDs (number)",
        "STDs:condylomatosis", "STDs:cervical condylomatosis",
        "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
        "STDs:syphilis", "STDs:pelvic inflammatory disease",
        "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS",
        "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV",
        "STDs: Number of diagnosis", "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis", "Dx:Cancer", "Dx:CIN", "Dx:HPV",
        "Dx", "Hinselmann", "Schiller", "Citology"
    ]
    return model, features

# === Load Recommendation Model ===
def load_recommendation_model():
    model_path = os.path.join("Models", "saved_models", "cervical_cancer_pipeline.pkl")
    encoder_path = os.path.join("Models", "saved_models", "label_encoder.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # ✅ now this is a Pipeline

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    features = [
        'Age', 'Sexual_Partners', 'First_Sexual_Activity_Age',
        'HPV_Test_Result', 'Pap_Smear_Result', 'Smoking_Status',
        'STDs_History', 'Region', 'Insurance_Covered', 'Screening_Type_Last'
    ]

    return model, features, encoder
