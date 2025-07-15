from fastapi import APIRouter
from api.model_loader import biopsy_model, recommendation_model
from api.schemas import BiopsyRiskInput, ScreeningRecommendationInput
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/predict-full-assessment")
def predict_full_assessment(
        biopsy_data: BiopsyRiskInput,
        recommendation_data: ScreeningRecommendationInput
):
    try:
        # Check for UNKNOWN test results and return rule-based response
        if recommendation_data.HPV_Test_Result.upper() == "UNKNOWN" or recommendation_data.Pap_Smear_Result.upper() == "UNKNOWN":
            return {
                "biopsy_risk": {
                    "prediction": None,
                    "confidence": None
                },
                "screening_recommendation": "RECOMMEND PAP SMEAR AND HPV TESTING"
            }

        biopsy_input_df = biopsy_data.to_dataframe()
        biopsy_prediction = biopsy_model.predict(biopsy_input_df)[0]
        biopsy_prob = biopsy_model.predict_proba(biopsy_input_df)[0].max()

        recommendation_input_df = recommendation_data.to_dataframe()
        recommendation_prediction = recommendation_model.predict(recommendation_input_df)[0]

        return {
            "biopsy_risk": {
                "prediction": int(biopsy_prediction),
                "confidence": round(biopsy_prob, 4)
            },
            "screening_recommendation": recommendation_prediction
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/predict-recommendation-only")
def predict_recommendation_only(data: ScreeningRecommendationInput):
    try:
        # Check for UNKNOWN test results and return rule-based response
        if data.HPV_Test_Result.upper() == "UNKNOWN" or data.Pap_Smear_Result.upper() == "UNKNOWN":
            return {
                "screening_recommendation": "RECOMMEND PAP SMEAR AND HPV TESTING"
            }

        input_df = data.to_dataframe()
        prediction = recommendation_model.predict(input_df)[0]
        return {"screening_recommendation": prediction}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
