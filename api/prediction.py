import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas import BiopsyInput, RecommendationInput
from .model_loader import load_biopsy_model, load_recommendation_model

router = APIRouter()

# === Biopsy Risk Prediction ===
@router.post("/predict-biopsy")
def predict_biopsy_endpoint(input_data: BiopsyInput):
    try:
        model, expected_features = load_biopsy_model()

        # Convert input to DataFrame using alias keys (matches model training column names)
        df = pd.DataFrame([input_data.model_dump(by_alias=True)])

        # Fill any missing expected columns with None
        for col in expected_features:
            if col not in df.columns:
                df[col] = None

        # Reorder columns to match model input
        df = df[expected_features]

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "probability_percent": round(probability * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
@router.post("/predict-recommendation")
def predict_recommendation_endpoint(input_data: RecommendationInput):
    try:
        model, features, encoder = load_recommendation_model()
        df = pd.DataFrame([input_data.dict()])

        missing = [col for col in features if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")

        pred_encoded = model.predict(df[features])[0]
        prediction = encoder.inverse_transform([pred_encoded])[0]

        return {
            "recommended_action": prediction,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))