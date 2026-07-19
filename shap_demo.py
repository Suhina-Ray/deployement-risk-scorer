"""
Deployment Risk Scorer - SHAP Explainability Demo
Loads the trained model and explains one high-risk and one low-risk
example deployment, saving a SHAP waterfall plot for each.
"""
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

with open("risk_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
service_encoder = bundle["service_encoder"]
risk_encoder = bundle["risk_encoder"]
feature_cols = bundle["feature_cols"]

explainer = shap.TreeExplainer(model)


def explain_deployment(example: dict, label: str):
    """example: raw feature dict (service as string, rest numeric)"""
    row = example.copy()
    row["service_enc"] = service_encoder.transform([row.pop("service")])[0]
    X_row = pd.DataFrame([row])[feature_cols]

    pred_class = model.predict(X_row)[0]
    pred_proba = model.predict_proba(X_row)[0]
    pred_label = risk_encoder.inverse_transform([pred_class])[0]

    shap_values = explainer(X_row)
    # shap_values.values shape: (1, n_features, n_classes) for multiclass
    class_idx = pred_class
    contribs = shap_values.values[0, :, class_idx]

    print(f"\n=== {label} ===")
    print(f"Input: {example}")
    print(f"Predicted risk: {pred_label}  "
          f"(P(Low)={pred_proba[risk_encoder.transform(['Low'])[0]]:.2f}, "
          f"P(Medium)={pred_proba[risk_encoder.transform(['Medium'])[0]]:.2f}, "
          f"P(High)={pred_proba[risk_encoder.transform(['High'])[0]]:.2f})")
    print("Top factors driving this prediction:")
    order = np.argsort(-np.abs(contribs))
    for i in order[:6]:
        sign = "+" if contribs[i] > 0 else "-"
        print(f"  {sign} {feature_cols[i]} = {X_row.iloc[0, i]}  (impact {contribs[i]:+.3f})")

    # waterfall plot
    fig = plt.figure(figsize=(7, 4.5))
    shap.plots.waterfall(
        shap.Explanation(
            values=contribs,
            base_values=shap_values.base_values[0, class_idx],
            data=X_row.iloc[0].values,
            feature_names=feature_cols,
        ),
        show=False,
    )
    plt.title(f"SHAP Explanation — Predicted: {pred_label}")
    plt.tight_layout()
    fname = f"shap_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# Example 1: the classic "high risk" deployment for your demo
high_risk_example = {
    "files_changed": 150,
    "loc_changed": 3000,
    "hour": 23,
    "day_of_week": 5,       # Saturday
    "is_weekend": 1,
    "service": "payments",
    "incidents_last_week": 4,
    "on_call_available": 0,
    "failed_tests": 2,
}
explain_deployment(high_risk_example, "High Risk Example")

# Example 2: a routine safe deployment, for contrast
low_risk_example = {
    "files_changed": 3,
    "loc_changed": 40,
    "hour": 14,
    "day_of_week": 2,       # Wednesday
    "is_weekend": 0,
    "service": "notifications",
    "incidents_last_week": 0,
    "on_call_available": 1,
    "failed_tests": 0,
}
explain_deployment(low_risk_example, "Low Risk Example")
