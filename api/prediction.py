from fastapi import APIRouter, HTTPException
from api.schemas import (
    BiopsyRiskInput,
    ScreeningRecommendationInput,
    FullAssessmentRequest,
    RecommendationOnlyResponse
)
from api.model_loader import load_biopsy_model, load_recommendation_model, load_label_encoder
import pandas as pd
router = APIRouter()
# === Load models ===
try:
    biopsy_model = load_biopsy_model()
    recommendation_model = load_recommendation_model()
    label_encoder = load_label_encoder()
except Exception as e:
    print("Model loading error:", e)
    biopsy_model = None
    recommendation_model = None
    label_encoder = None

# === FULL ASSESSMENT ENDPOINT ===
@router.post("/predict-full-assessment")
def predict_full_assessment(request: FullAssessmentRequest):
    if not biopsy_model or not recommendation_model or not label_encoder:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        biopsy_dict = request.biopsy_data.dict()
        recommendation_dict = request.recommendation_data.dict()
        # Column mapping for biopsy model
        biopsy_column_mapping = {
            "Age": "Age",
            "Number_of_Pregnancies": "Num of pregnancies",
            "Number_of_sexual_partners": "Number of sexual partners",
            "First_sexual_intercourse_age": "First sexual intercourse",
            "Smokes": "Smokes",
            "Smokes_years": "Smokes (years)",
            "Smokes_packs_per_year": "Smokes (packs/year)",
            "Hormonal_Contraceptives": "Hormonal Contraceptives",
            "Hormonal_Contraceptives_years": "Hormonal Contraceptives (years)",
            "IUD": "IUD",
            "IUD_years": "IUD (years)",
            "STDs": "STDs",
            "STDs_number": "STDs (number)",
            "STDs_condylomatosis": "STDs:condylomatosis",
            "STDs_cervical_condylomatosis": "STDs:cervical condylomatosis",
            "STDs_vaginal_condylomatosis": "STDs:vaginal condylomatosis",
            "STDs_vulvo_perineal_condylomatosis": "STDs:vulvo-perineal condylomatosis",
            "STDs_syphilis": "STDs:syphilis",
            "STDs_pelvic_inflammatory_disease": "STDs:pelvic inflammatory disease",
            "STDs_genital_herpes": "STDs:genital herpes",
            "STDs_molluscum_contagiosum": "STDs:molluscum contagiosum",
            "STDs_AIDS": "STDs:AIDS",
            "STDs_HIV": "STDs:HIV",
            "STDs_Hepatitis_B": "STDs:Hepatitis B",
            "STDs_HPV": "STDs:HPV",
            "STDs_Time_since_first_diagnosis": "STDs: Time since first diagnosis",
            "STDs_Time_since_last_diagnosis": "STDs: Time since last diagnosis",
            "Dx": "Dx",
            "Dx_Cancer": "Dx:Cancer",
            "Dx_CIN": "Dx:CIN",
            "Dx_HPV": "Dx:HPV",
            "Hinselmann": "Hinselmann",
            "Schiller": "Schiller",
            "Citology": "Citology"
        }

        # Rename columns
        renamed_biopsy_data = {
            biopsy_column_mapping.get(k): v for k, v in biopsy_dict.items() if k in biopsy_column_mapping
        }

        # Fill missing columns
        expected_columns = biopsy_model.feature_names_in_
        for col in expected_columns:
            if col not in renamed_biopsy_data:
                renamed_biopsy_data[col] = 0

        # Create DataFrames
        biopsy_df = pd.DataFrame([renamed_biopsy_data])
        recommendation_df = pd.DataFrame([recommendation_dict])

        # Predict
        biopsy_pred = biopsy_model.predict(biopsy_df)[0]
        biopsy_proba = float(biopsy_model.predict_proba(biopsy_df).max())

        encoded_prediction = recommendation_model.predict(recommendation_df)[0]
        recommendation_result = label_encoder.inverse_transform([encoded_prediction])[0]

        return {
            "biopsy_risk": {
                "prediction": int(biopsy_pred),
                "confidence": round(biopsy_proba * 100, 2)
            },
            "screening_recommendation": recommendation_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === RECOMMENDATION ONLY ENDPOINT ===
@router.post("/predict-recommendation-only", response_model=RecommendationOnlyResponse)
def recommend_screening(input_data: ScreeningRecommendationInput):
    if not recommendation_model or not label_encoder:
        raise HTTPException(status_code=500, detail="Recommendation model not available")
    try:
        df = pd.DataFrame([input_data.dict()])
        encoded_prediction = recommendation_model.predict(df)[0]
        recommendation = label_encoder.inverse_transform([encoded_prediction])[0]
        return {"recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
