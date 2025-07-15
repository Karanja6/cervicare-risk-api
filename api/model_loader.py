import os
import pickle

# Define the base model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Models", "saved_models")

# Correctly use MODEL_DIR in both model paths
BIOPSY_MODEL_PATH = os.path.join(MODEL_DIR, "biopsy_pipeline.pkl")
RECOMMENDATION_MODEL_PATH = os.path.join(MODEL_DIR, "cervical_cancer_pipeline.pkl")

def load_biopsy_model():
    with open(BIOPSY_MODEL_PATH, "rb") as file:
        return pickle.load(file)

def load_recommendation_model():
    with open(RECOMMENDATION_MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load at startup
biopsy_model = load_biopsy_model()
recommendation_model = load_recommendation_model()
