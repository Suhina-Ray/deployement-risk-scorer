"""
Deployment Risk Scorer - Synthetic Dataset Generator
Simulates historical deployment metadata with a realistic (noisy) risk-generating
process, so the downstream model has real signal to learn instead of random labels.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

services = ["payments", "auth", "notifications", "search", "checkout", "reporting"]
critical_services = {"payments", "auth", "checkout"}

rows = []
for _ in range(N):
    files_changed = int(np.random.exponential(scale=15)) + 1
    loc_changed = files_changed * int(np.random.exponential(scale=25)) + np.random.randint(0, 50)
    hour = np.random.randint(0, 24)
    service = np.random.choice(services)
    incidents_last_week = np.random.poisson(0.6 if service in critical_services else 0.2)
    on_call_available = np.random.choice([1, 0], p=[0.75, 0.25])
    day_of_week = np.random.randint(0, 7)  # 0=Mon
    is_weekend = 1 if day_of_week >= 5 else 0
    failed_tests = np.random.poisson(0.3)

    # ---- ground-truth risk score (latent), built from domain logic + noise ----
    score = 0.0
    score += min(files_changed / 100, 1.0) * 2.2
    score += min(loc_changed / 2000, 1.0) * 2.0
    score += 1.5 if (hour >= 22 or hour <= 5) else 0.0
    score += 1.8 if service in critical_services else 0.3
    score += incidents_last_week * 0.9
    score += -1.3 if on_call_available else 1.0
    score += 0.6 if is_weekend else 0.0
    score += failed_tests * 1.4
    score += np.random.normal(0, 0.8)  # noise

    if score < 2.5:
        risk = "Low"
    elif score < 5.0:
        risk = "Medium"
    else:
        risk = "High"

    rows.append({
        "files_changed": files_changed,
        "loc_changed": loc_changed,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "service": service,
        "incidents_last_week": incidents_last_week,
        "on_call_available": on_call_available,
        "failed_tests": failed_tests,
        "risk": risk,
    })

df = pd.DataFrame(rows)
df.to_csv("deployment_dataset.csv", index=False)
print(df["risk"].value_counts())
print(df.head())
print(f"\nTotal rows: {len(df)}")
