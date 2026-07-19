"""
Deployment Risk Scorer - Live Demo Script
Run this in front of the panel: pass deployment details, get a risk score
with SHAP-backed reasons. Edit the `deployment` dict below to try your own
scenarios live.
"""
import pandas as pd
import numpy as np
import shap
import pickle

with open("risk_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
service_encoder = bundle["service_encoder"]
risk_encoder = bundle["risk_encoder"]
feature_cols = bundle["feature_cols"]
explainer = shap.TreeExplainer(model)

# ---- EDIT THIS to demo different scenarios live ----
deployment = {
    "files_changed": 80,
    "loc_changed": 1600,
    "hour": 22,
    "day_of_week": 4,
    "is_weekend": 0,
    "service": "checkout",
    "incidents_last_week": 2,
    "on_call_available": 0,
    "failed_tests": 1,
}
# -----------------------------------------------------

row = deployment.copy()
row["service_enc"] = service_encoder.transform([row.pop("service")])[0]
X_row = pd.DataFrame([row])[feature_cols]

pred_class = model.predict(X_row)[0]
pred_label = risk_encoder.inverse_transform([pred_class])[0]
proba = model.predict_proba(X_row)[0]

shap_values = explainer(X_row)
contribs = shap_values.values[0, :, pred_class]
order = np.argsort(-np.abs(contribs))

print("=" * 50)
print(f"  DEPLOYMENT RISK SCORE: {pred_label.upper()}")
print("=" * 50)
print(f"  Low: {proba[risk_encoder.transform(['Low'])[0]]:.0%}  |  "
      f"Medium: {proba[risk_encoder.transform(['Medium'])[0]]:.0%}  |  "
      f"High: {proba[risk_encoder.transform(['High'])[0]]:.0%}")
print("\nWhy:")
for i in order[:5]:
    sign = "increases" if contribs[i] > 0 else "decreases"
    print(f"  - {feature_cols[i]} = {X_row.iloc[0, i]}  ({sign} risk)")
