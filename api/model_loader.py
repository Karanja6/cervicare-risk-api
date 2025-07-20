import os
import pickle
# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Models", "saved_models")
BIOPSY_MODEL_PATH = os.path.join(MODEL_DIR, "biopsy_pipeline.pkl")
RECOMMENDATION_MODEL_PATH = os.path.join(MODEL_DIR, "cervical_cancer_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
# Load Biopsy Model
def load_biopsy_model():
    with open(BIOPSY_MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load Recommendation Model
def load_recommendation_model():
    with open(RECOMMENDATION_MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load Label Encoder for Recommendation Predictions
def load_label_encoder():
    with open(LABEL_ENCODER_PATH, "rb") as file:
        return pickle.load(file)
