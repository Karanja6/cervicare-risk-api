from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

class BiopsyRiskInput(BaseModel):
    Age: float
    Number_of_Pregnancies: int
    Smokes: int
    Hormonal_Contraceptives: int
    STDs: int
    HPV: int
    IUD: int
    STDs_Number: int
    First_sexual_intercourse_age: float
    STDs_condylomatosis: int
    STDs_cervical_condylomatosis: int
    STDs_vaginal_condylomatosis: int
    STDs_vulvo_perineal_condylomatosis: int
    STDs_syphilis: int
    STDs_pelvic_inflammatory_disease: int
    STDs_genital_herpes: int
    STDs_molluscum_contagiosum: int
    STDs_HIV: int
    STDs_Hepatitis_B: int
    STDs_HPV: int
    Dx: int
    Dx_Cancer: int
    Dx_CIN: int

    def to_dataframe(self):
        return pd.DataFrame([self.dict()])


class ScreeningRecommendationInput(BaseModel):
    Age: int
    Sexual_Partners: Optional[int] = None
    First_Sexual_Activity_Age: Optional[float] = None

    # Explicitly allow null or "UNKNOWN" string inputs
    HPV_Test_Result: Optional[str] = Field(default=None)
    Pap_Smear_Result: Optional[str] = Field(default=None)

    Smoking_Status: Optional[str] = None
    STDs_History: Optional[str] = None
    Region: Optional[str] = None
    Insurance_Covered: Optional[str] = None
    Screening_Type_Last: Optional[str] = None

    def to_dataframe(self):
        return pd.DataFrame([self.dict()])


# Optional response models (not strictly required but useful for docs)
class FullAssessmentResponse(BaseModel):
    biopsy_risk_score: Optional[float]
    recommendation: str

class RecommendationOnlyResponse(BaseModel):
    recommendation: str
