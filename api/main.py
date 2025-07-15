# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import prediction

app = FastAPI(
    title="CerviCare Risk Assessment API",
    description="Provides endpoints for biopsy risk prediction and cervical cancer screening recommendations.",
    version="1.0.0"
)

# Allow all origins (update for production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(prediction.router)

@app.get("/")
def root():
    return {"message": "CerviCare Risk Assessment API is running"}
