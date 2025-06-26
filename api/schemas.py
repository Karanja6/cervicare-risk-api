from pydantic import BaseModel, Field
from typing import Optional

class BiopsyInput(BaseModel):
    Patient_ID: str
    Age: int
    Number_of_sexual_partners: Optional[float] = Field(None, alias="Number of sexual partners")
    First_sexual_intercourse: Optional[float] = Field(None, alias="First sexual intercourse")
    Num_of_pregnancies: Optional[float] = Field(None, alias="Num of pregnancies")

    Smokes: Optional[float] = Field(None, alias="Smokes")
    Smokes_years: Optional[float] = Field(None, alias="Smokes (years)")
    Smokes_packs_per_year: Optional[float] = Field(None, alias="Smokes (packs/year)")

    Hormonal_Contraceptives: Optional[float] = Field(None, alias="Hormonal Contraceptives")
    Hormonal_Contraceptives_years: Optional[float] = Field(None, alias="Hormonal Contraceptives (years)")

    IUD: Optional[float] = Field(None, alias="IUD")
    IUD_years: Optional[float] = Field(None, alias="IUD (years)")

    STDs: Optional[float] = Field(None, alias="STDs")
    STDs_number: Optional[float] = Field(None, alias="STDs (number)")
    STDs_condylomatosis: Optional[float] = Field(None, alias="STDs:condylomatosis")
    STDs_cervical_condylomatosis: Optional[float] = Field(None, alias="STDs:cervical condylomatosis")
    STDs_vaginal_condylomatosis: Optional[float] = Field(None, alias="STDs:vaginal condylomatosis")
    STDs_vulvo_perineal_condylomatosis: Optional[float] = Field(None, alias="STDs:vulvo-perineal condylomatosis")
    STDs_syphilis: Optional[float] = Field(None, alias="STDs:syphilis")
    STDs_pelvic_inflammatory_disease: Optional[float] = Field(None, alias="STDs:pelvic inflammatory disease")
    STDs_genital_herpes: Optional[float] = Field(None, alias="STDs:genital herpes")
    STDs_molluscum_contagiosum: Optional[float] = Field(None, alias="STDs:molluscum contagiosum")
    STDs_AIDS: Optional[float] = Field(None, alias="STDs:AIDS")
    STDs_HIV: Optional[float] = Field(None, alias="STDs:HIV")
    STDs_Hepatitis_B: Optional[float] = Field(None, alias="STDs:Hepatitis B")
    STDs_HPV: Optional[float] = Field(None, alias="STDs:HPV")
    STDs_Number_of_diagnosis: Optional[float] = Field(None, alias="STDs: Number of diagnosis")
    STDs_Time_since_first_diagnosis: Optional[float] = Field(None, alias="STDs: Time since first diagnosis")
    STDs_Time_since_last_diagnosis: Optional[float] = Field(None, alias="STDs: Time since last diagnosis")

    Dx_Cancer: Optional[float] = Field(None, alias="Dx:Cancer")
    Dx_CIN: Optional[float] = Field(None, alias="Dx:CIN")
    Dx_HPV: Optional[float] = Field(None, alias="Dx:HPV")
    Dx: Optional[float] = Field(None, alias="Dx")

    Hinselmann: Optional[float] = Field(None, alias="Hinselmann")
    Schiller: Optional[float] = Field(None, alias="Schiller")
    Citology: Optional[float] = Field(None, alias="Citology")

    class Config:
        populate_by_name = True

class RecommendationInput(BaseModel):
    Age: int
    Sexual_Partners: int
    First_Sexual_Activity_Age: int
    HPV_Test_Result: str
    Pap_Smear_Result: str
    Smoking_Status: str
    STDs_History: str
    Region: str
    Insurance_Covered: str
    Screening_Type_Last: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float


class RecommendationResponse(BaseModel):
    recommended_action: str
    confidence: float
