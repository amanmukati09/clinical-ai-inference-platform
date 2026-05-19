"""
FastAPI inference server for readmission prediction.
Serves the trained PyTorch model with confidence scoring,
input validation, and structured error handling.
"""

import os
import sys
import pickle
import logging
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.readmission_model import ReadmissionModel, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Readmission Prediction API",
    description="Predicts 30-day readmission risk for diabetic patients",
    version="1.0.0"
)

# ── Global model state ──────────────────────────────────────────────────────

MODEL = None
SCALER = None
ENCODERS = None
FEATURE_NAMES = None

ARTIFACT_DIR = os.getenv(
    "ARTIFACT_DIR",
    "/home/aman/clinical-ai-platform/data/processed/artifacts"
)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/aman/clinical-ai-platform/ml-service/models/best_model.pt"
)


@app.on_event("startup")
def load_model():
    global MODEL, SCALER, ENCODERS, FEATURE_NAMES

    logger.info("Loading model artifacts...")

    # Load scaler
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
        SCALER = pickle.load(f)

    # Load encoders
    with open(os.path.join(ARTIFACT_DIR, "encoders.pkl"), "rb") as f:
        ENCODERS = pickle.load(f)

    # Load feature names
    with open(os.path.join(ARTIFACT_DIR, "feature_names.pkl"), "rb") as f:
        FEATURE_NAMES = pickle.load(f)

    # Load PyTorch model
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    config = checkpoint["config"]
    MODEL = ReadmissionModel(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"]
    )
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    logger.info(f"Model loaded. Trained AUC: {checkpoint['val_auc']:.4f}")
    logger.info(f"Features expected: {FEATURE_NAMES}")


# ── Request / Response schemas ───────────────────────────────────────────────

class PatientData(BaseModel):
    """
    Input schema for a single patient encounter.
    All fields map directly to preprocessed feature columns.
    """
    race: str = Field(..., example="Caucasian")
    gender: str = Field(..., example="Female")
    age: int = Field(..., ge=0, le=100, example=55)
    admission_type_id: int = Field(..., ge=1, le=8, example=1)
    discharge_disposition_id: int = Field(..., ge=1, le=30, example=1)
    admission_source_id: int = Field(..., ge=1, le=26, example=7)
    time_in_hospital: int = Field(..., ge=1, le=14, example=3)
    num_lab_procedures: int = Field(..., ge=0, example=41)
    num_procedures: int = Field(..., ge=0, example=0)
    num_medications: int = Field(..., ge=0, example=12)
    number_outpatient: int = Field(..., ge=0, example=0)
    number_emergency: int = Field(..., ge=0, example=0)
    number_inpatient: int = Field(..., ge=0, example=0)
    diag_1: int = Field(..., ge=0, le=8, example=4)
    diag_2: int = Field(..., ge=0, le=8, example=0)
    diag_3: int = Field(..., ge=0, le=8, example=0)
    number_diagnoses: int = Field(..., ge=0, example=9)
    metformin: int = Field(..., ge=0, le=3, example=0)
    repaglinide: int = Field(..., ge=0, le=3, example=0)
    nateglinide: int = Field(..., ge=0, le=3, example=0)
    chlorpropamide: int = Field(..., ge=0, le=3, example=0)
    glimepiride: int = Field(..., ge=0, le=3, example=0)
    acetohexamide: int = Field(..., ge=0, le=3, example=0)
    glipizide: int = Field(..., ge=0, le=3, example=0)
    glyburide: int = Field(..., ge=0, le=3, example=0)
    tolbutamide: int = Field(..., ge=0, le=3, example=0)
    pioglitazone: int = Field(..., ge=0, le=3, example=0)
    rosiglitazone: int = Field(..., ge=0, le=3, example=0)
    acarbose: int = Field(..., ge=0, le=3, example=0)
    miglitol: int = Field(..., ge=0, le=3, example=0)
    troglitazone: int = Field(..., ge=0, le=3, example=0)
    tolazamide: int = Field(..., ge=0, le=3, example=0)
    examide: int = Field(..., ge=0, le=3, example=0)
    citoglipton: int = Field(..., ge=0, le=3, example=0)
    insulin: int = Field(..., ge=0, le=3, example=2)
    glyburide_metformin: int = Field(..., ge=0, le=3, example=0)
    glipizide_metformin: int = Field(..., ge=0, le=3, example=0)
    glimepiride_pioglitazone: int = Field(..., ge=0, le=3, example=0)
    metformin_rosiglitazone: int = Field(..., ge=0, le=3, example=0)
    metformin_pioglitazone: int = Field(..., ge=0, le=3, example=0)
    change: int = Field(..., ge=0, le=1, example=0)
    diabetesMed: int = Field(..., ge=0, le=1, example=1)

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("gender must be Male or Female")
        return v


class PredictionResponse(BaseModel):
    readmission_risk_score: float = Field(
        ..., description="Probability of 30-day readmission (0-1)"
    )
    risk_category: str = Field(
        ..., description="LOW / MEDIUM / HIGH based on score thresholds"
    )
    flag_for_intervention: bool = Field(
        ..., description="True if patient should be flagged for care coordination"
    )
    confidence: str = Field(
        ..., description="Model confidence: HIGH / MEDIUM / LOW"
    )
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_auc: Optional[float]


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_risk_category(score: float) -> str:
    """
    Clinical thresholds — not arbitrary.
    Based on typical care management triage cutoffs:
    - <0.15: baseline population risk, no intervention
    - 0.15-0.30: elevated, schedule follow-up call
    - >0.30: high risk, assign care coordinator
    """
    if score < 0.15:
        return "LOW"
    elif score < 0.30:
        return "MEDIUM"
    else:
        return "HIGH"


def compute_confidence(score: float) -> str:
    """
    Confidence based on distance from decision boundary (0.5).
    Scores near 0.5 = model is uncertain.
    Scores near 0 or 1 = model is confident.
    """
    distance = abs(score - 0.5)
    if distance > 0.3:
        return "HIGH"
    elif distance > 0.15:
        return "MEDIUM"
    else:
        return "LOW"


def preprocess_input(patient: PatientData) -> np.ndarray:
    """Convert PatientData to scaled feature vector."""
    # Build feature dict matching training column order
    feature_map = {
        "race": ENCODERS["race"].transform(
            [patient.race]
        )[0] if "race" in ENCODERS else 0,
        "gender": int(patient.gender == "Male"),
        "age": patient.age,
        "admission_type_id": patient.admission_type_id,
        "discharge_disposition_id": patient.discharge_disposition_id,
        "admission_source_id": patient.admission_source_id,
        "time_in_hospital": patient.time_in_hospital,
        "num_lab_procedures": patient.num_lab_procedures,
        "num_procedures": patient.num_procedures,
        "num_medications": patient.num_medications,
        "number_outpatient": patient.number_outpatient,
        "number_emergency": patient.number_emergency,
        "number_inpatient": patient.number_inpatient,
        "diag_1": patient.diag_1,
        "diag_2": patient.diag_2,
        "diag_3": patient.diag_3,
        "number_diagnoses": patient.number_diagnoses,
        "metformin": patient.metformin,
        "repaglinide": patient.repaglinide,
        "nateglinide": patient.nateglinide,
        "chlorpropamide": patient.chlorpropamide,
        "glimepiride": patient.glimepiride,
        "acetohexamide": patient.acetohexamide,
        "glipizide": patient.glipizide,
        "glyburide": patient.glyburide,
        "tolbutamide": patient.tolbutamide,
        "pioglitazone": patient.pioglitazone,
        "rosiglitazone": patient.rosiglitazone,
        "acarbose": patient.acarbose,
        "miglitol": patient.miglitol,
        "troglitazone": patient.troglitazone,
        "tolazamide": patient.tolazamide,
        "examide": patient.examide,
        "citoglipton": patient.citoglipton,
        "insulin": patient.insulin,
        "glyburide-metformin": patient.glyburide_metformin,
        "glipizide-metformin": patient.glipizide_metformin,
        "glimepiride-pioglitazone": patient.glimepiride_pioglitazone,
        "metformin-rosiglitazone": patient.metformin_rosiglitazone,
        "metformin-pioglitazone": patient.metformin_pioglitazone,
        "change": patient.change,
        "diabetesMed": patient.diabetesMed,
    }

    # Build array in exact feature order from training
    feature_vector = np.array(
        [feature_map[f] for f in FEATURE_NAMES],
        dtype=np.float32
    ).reshape(1, -1)

    # Apply same scaler used during training
    feature_vector = SCALER.transform(feature_vector)
    return feature_vector


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu") \
        if MODEL is not None else None
    return HealthResponse(
        status="healthy" if MODEL is not None else "model not loaded",
        model_loaded=MODEL is not None,
        model_auc=checkpoint["val_auc"] if checkpoint else None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if MODEL is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    try:
        start = time.time()

        features = preprocess_input(patient)
        tensor = torch.FloatTensor(features)
        score = float(MODEL.predict_proba(tensor).item())

        elapsed = (time.time() - start) * 1000

        return PredictionResponse(
            readmission_risk_score=round(score, 4),
            risk_category=compute_risk_category(score),
            flag_for_intervention=score >= 0.15,
            confidence=compute_confidence(score),
            inference_time_ms=round(elapsed, 2),
            model_version="1.0.0-mlp-pytorch"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Prediction error: {str(e)}")


@app.get("/")
def root():
    return {"service": "Clinical AI Inference API", "version": "1.0.0"}