"""
Deployment Risk Scorer - Model Training
Trains an XGBoost classifier on the synthetic deployment dataset,
reports accuracy/precision/recall/confusion matrix, and saves the model.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("deployment_dataset.csv")

# Encode categorical feature
service_encoder = LabelEncoder()
df["service_enc"] = service_encoder.fit_transform(df["service"])

# Encode target. LabelEncoder sorts classes alphabetically (High, Low, Medium) --
# always read the label order back from risk_encoder.classes_, never assume it.
risk_encoder = LabelEncoder()
df["risk_enc"] = risk_encoder.fit_transform(df["risk"])
risk_order = list(risk_encoder.classes_)  # e.g. ['High', 'Low', 'Medium']

feature_cols = [
    "files_changed", "loc_changed", "hour", "day_of_week", "is_weekend",
    "service_enc", "incidents_last_week", "on_call_available", "failed_tests"
]

X = df[feature_cols]
y = df["risk_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
)
model.fit(X_train, y_train)

# ---- Evaluation ----
# Force display order to Low -> Medium -> High regardless of the encoder's
# internal (alphabetical) ordering, by passing explicit `labels=`.
display_order = ["Low", "Medium", "High"]
display_labels = risk_encoder.transform(display_order)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, labels=display_labels, target_names=display_order))

cm = confusion_matrix(y_test, y_pred, labels=display_labels)
print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=display_order, columns=display_order))

# ---- Confusion matrix plot ----
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(display_order)
ax.set_yticks(range(3)); ax.set_yticklabels(display_order)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (Accuracy: {acc:.1%})")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---- Feature importance plot ----
importances = model.feature_importances_
order = np.argsort(importances)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.barh([feature_cols[i] for i in order], importances[order], color="#2563eb")
ax.set_title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ---- Save model + encoders ----
with open("risk_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "service_encoder": service_encoder,
        "risk_encoder": risk_encoder,
        "feature_cols": feature_cols,
    }, f)

print("\nSaved: risk_model.pkl, confusion_matrix.png, feature_importance.png")
