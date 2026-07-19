"""
Deployment Risk Scorer - Backend API
Wraps the trained XGBoost model + SHAP explainer as a REST API so the
frontend can send deployment details and get back a risk score with reasons.

Run with:  python app.py
Then open: http://127.0.0.1:8000/docs  (interactive API test page)
"""
import pickle
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Deployment Risk Scorer API")

# Allow the frontend (opened as a local file or served separately) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model once at startup ----
with open("risk_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
service_encoder = bundle["service_encoder"]
risk_encoder = bundle["risk_encoder"]
feature_cols = bundle["feature_cols"]
explainer = shap.TreeExplainer(model)

VALID_SERVICES = list(service_encoder.classes_)

# Real metrics from the last training run (see train_model.py output).
# Update these two numbers if you retrain and get a different split.
MODEL_META = {
    "algorithm": "XGBoost",
    "training_samples": 3000,
    "accuracy": 0.797,
    "features": len(feature_cols),
}


class DeploymentInput(BaseModel):
    files_changed: int
    loc_changed: int
    hour: int
    day_of_week: int
    is_weekend: int
    service: str
    incidents_last_week: int
    on_call_available: int
    failed_tests: int


@app.get("/health")
def health():
    return {"status": "online", "model": MODEL_META}


@app.get("/services")
def get_services():
    """List of valid service names for the frontend dropdown."""
    return {"services": VALID_SERVICES}


@app.post("/predict")
def predict(payload: DeploymentInput):
    if payload.service not in VALID_SERVICES:
        raise HTTPException(
            status_code=400,
            detail=f"service must be one of {VALID_SERVICES}",
        )

    row = payload.dict()
    row["service_enc"] = service_encoder.transform([row.pop("service")])[0]
    X_row = pd.DataFrame([row])[feature_cols]

    pred_class = int(model.predict(X_row)[0])
    pred_label = risk_encoder.inverse_transform([pred_class])[0]
    proba = model.predict_proba(X_row)[0]

    shap_values = explainer(X_row)
    contribs = shap_values.values[0, :, pred_class]
    order = np.argsort(-np.abs(contribs))[:5]

    reasons = []
    for i in order:
        reasons.append({
            "feature": feature_cols[i],
            "value": float(X_row.iloc[0, i]),
            "impact": round(float(contribs[i]), 3),
            "direction": "increases" if contribs[i] > 0 else "decreases",
        })

    return {
        "risk": pred_label,
        "probabilities": {
            "Low": round(float(proba[risk_encoder.transform(["Low"])[0]]), 3),
            "Medium": round(float(proba[risk_encoder.transform(["Medium"])[0]]), 3),
            "High": round(float(proba[risk_encoder.transform(["High"])[0]]), 3),
        },
        "reasons": reasons,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
