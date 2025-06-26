# api/main.py

from fastapi import FastAPI
from api.prediction import router  # ✅ Import the router

app = FastAPI(
    title="CerviCare ML API",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to CerviCare ML API 🚀"}

# Mount router at /api, so routes are like /api/predict-biopsy
app.include_router(router, prefix="/api")
