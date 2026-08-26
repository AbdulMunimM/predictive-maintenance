from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"


# ============================================================
# Load deployment artifacts
# ============================================================

# Preprocessors
model1_preprocessor = joblib.load(
    ARTIFACT_DIR / "model1_preprocessor.joblib"
)

model2_preprocessor = joblib.load(
    ARTIFACT_DIR / "model2_preprocessor.joblib"
)


# Model 1 - Machine Failure XGBoost
model1 = XGBClassifier()

model1.load_model(
    str(ARTIFACT_DIR / "model1_xgboost.ubj")
)


# Model 2 - Failure Mode XGBoost models
FAILURE_MODES = [
    "TWF",
    "HDF",
    "PWF",
    "OSF"
]

mode_models = {}

for mode in FAILURE_MODES:

    model = XGBClassifier()

    model.load_model(
        str(ARTIFACT_DIR / f"model2_{mode}.ubj")
    )

    mode_models[mode] = model


# Configuration
config = joblib.load(
    ARTIFACT_DIR / "project_config.joblib"
)


# ============================================================
# Configuration values
# ============================================================

FEATURES = config["features"]

NUMERIC_FEATURES = config["numeric_features"]

CATEGORICAL_FEATURES = config["categorical_features"]

FAILURE_THRESHOLD = config[
    "machine_failure"
]["threshold"]

MODE_THRESHOLDS = config[
    "failure_modes"
]["thresholds"]


# ============================================================
# SHAP
# ============================================================

shap_explainer = shap.TreeExplainer(
    model1
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="AI Predictive Maintenance API",
    description=(
        "AI4I 2020 predictive maintenance system for "
        "machine failure prediction, failure-mode "
        "classification, and model explainability."
    ),
    version="1.0.0"
)


# ============================================================
# Input schema
# ============================================================

class MachineInput(BaseModel):

    Type: str = Field(
        ...,
        description="Machine type: H, M, or L"
    )

    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin"
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed in rpm"
    )

    torque: float = Field(
        ...,
        description="Torque in Nm"
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes"
    )


# ============================================================
# Input validation
# ============================================================

def validate_input(machine: MachineInput):

    if machine.Type not in {"H", "M", "L"}:
        raise HTTPException(
            status_code=400,
            detail="Type must be one of: H, M, L"
        )

    if machine.air_temperature <= 0:
        raise HTTPException(
            status_code=400,
            detail="Air temperature must be greater than 0 K"
        )

    if machine.process_temperature <= 0:
        raise HTTPException(
            status_code=400,
            detail="Process temperature must be greater than 0 K"
        )

    if machine.rotational_speed < 0:
        raise HTTPException(
            status_code=400,
            detail="Rotational speed cannot be negative"
        )

    if machine.torque < 0:
        raise HTTPException(
            status_code=400,
            detail="Torque cannot be negative"
        )

    if machine.tool_wear < 0:
        raise HTTPException(
            status_code=400,
            detail="Tool wear cannot be negative"
        )


# ============================================================
# Feature engineering
# ============================================================

def create_model_features(
    machine: MachineInput
) -> pd.DataFrame:

    data = pd.DataFrame([{
        "Type": machine.Type,
        "Air temperature [K]": machine.air_temperature,
        "Process temperature [K]": machine.process_temperature,
        "Rotational speed [rpm]": machine.rotational_speed,
        "Torque [Nm]": machine.torque,
        "Tool wear [min]": machine.tool_wear
    }])

    # Engineered feature 1
    data["Temp_Diff"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    # Engineered feature 2
    data["Power_W"] = (
        data["Rotational speed [rpm]"]
        * data["Torque [Nm]"]
    )

    # Engineered feature 3
    data["Overstrain_Index"] = (
        data["Torque [Nm]"]
        * data["Tool wear [min]"]
    )

    return data[FEATURES]


# ============================================================
# Maintenance decision
# ============================================================

def generate_maintenance_decision(
    machine_failure,
    failure_modes
):

    predicted_modes = [
        mode
        for mode, result in failure_modes.items()
        if result["predicted"]
    ]

    # Failure predicted + known mode identified
    if machine_failure:

        if predicted_modes:

            mode_text = ", ".join(predicted_modes)

            return {
                "maintenance_status": "CRITICAL",
                "recommended_action": (
                    f"Inspect machine immediately. "
                    f"Likely failure mode(s): {mode_text}."
                )
            }

        # Failure predicted but no known mode identified
        return {
            "maintenance_status": "FAILURE_DETECTED",
            "recommended_action": (
                "Inspect machine because failure risk is high, "
                "but no known failure mode was identified."
            )
        }

    # No failure predicted
    return {
        "maintenance_status": "NORMAL",
        "recommended_action": (
            "Continue normal operation and monitoring."
        )
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model1": "loaded",
        "model2": "loaded"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(machine: MachineInput):

    validate_input(machine)

    # Create the exact nine features
    X = create_model_features(machine)

    # --------------------------------------------------------
    # Model 1 - Machine Failure
    # --------------------------------------------------------

    X_model1 = model1_preprocessor.transform(X)

    failure_probability = float(
        model1.predict_proba(X_model1)[0, 1]
    )

    machine_failure = (
        failure_probability >= FAILURE_THRESHOLD
    )


    # --------------------------------------------------------
    # Model 2 - Failure Modes
    # --------------------------------------------------------

    X_model2 = model2_preprocessor.transform(X)

    failure_modes = {}

    for mode in FAILURE_MODES:

        probability = float(
            mode_models[mode].predict_proba(X_model2)[0, 1]
        )

        predicted = (
            probability >= MODE_THRESHOLDS[mode]
        )

        failure_modes[mode] = {
            "probability": probability,
            "predicted": bool(predicted)
        }


    # --------------------------------------------------------
    # Maintenance decision
    # --------------------------------------------------------

    decision = generate_maintenance_decision(
        bool(machine_failure),
        failure_modes
    )


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "failure_probability": failure_probability,
        "machine_failure": bool(machine_failure),
        "failure_modes": failure_modes,
        "maintenance_status": decision[
            "maintenance_status"
        ],
        "recommended_action": decision[
            "recommended_action"
        ]
    }


# ============================================================
# SHAP explanation endpoint
# ============================================================

@app.post("/explain")
def explain(machine: MachineInput):

    validate_input(machine)

    # Create the exact nine features
    X = create_model_features(machine)

    # Apply the same preprocessing used by Model 1
    X_transformed = model1_preprocessor.transform(X)

    feature_names = (
        model1_preprocessor
        .get_feature_names_out()
    )

    # Calculate SHAP values
    shap_values = shap_explainer.shap_values(
        X_transformed
    )

    values = np.asarray(shap_values)[0]

    # Build contribution list
    contributions = []

    for i, value in enumerate(values):

        feature_name = feature_names[i]

        contributions.append({
            "feature": feature_name,
            "shap_value": float(value)
        })

    # Sort by absolute contribution
    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "base_value": float(
            shap_explainer.expected_value
        ),
        "contributions": contributions[:6]
    }