# Deployment Risk Scorer

Built for Synergy 2026 by Team 404 Not Found.

## What this is

Right now most teams decide whether a deployment is "risky" based on gut
feeling. Someone senior looks at the change and says "should be fine" or
"let's wait till morning." There's no consistent way to measure it.

This tool takes the details of a deployment (how big the change is, what
time it's going out, which service, how many incidents that service has
had recently, whether anyone's on call) and predicts whether it's Low,
Medium, or High risk. It also explains *why* it made that call, instead of
just spitting out a label you have to trust blindly.

## How it works

```
Deployment details --> XGBoost model --> Risk level (Low/Medium/High)
                                      --> SHAP explanation (why)
```

We trained the model on a synthetic dataset of 3,000 deployments (built
with realistic rules, not just random labels) since we don't have access
to a real company's deployment history. Accuracy on unseen test data is
about 80%. The features and pipeline stay the same either way, so this
swaps over to real GitHub data without much rework.

## What's in this repo

- `generate_dataset.py` - builds the synthetic training data
- `train_model.py` - trains the XGBoost model, prints accuracy and a
  confusion matrix, saves the trained model
- `shap_demo.py` - generates example explanations for a high risk and a
  low risk deployment
- `demo_predict.py` - quick script to test a single prediction from the
  terminal, useful for messing around with inputs
- `app.py` - the backend API (FastAPI) that the frontend actually talks to
- `index.html` - the dashboard, open this in a browser
- `risk_model.pkl` - the already-trained model, so you don't have to
  retrain unless you want to
- `deployment_dataset.csv` - the dataset itself

## Running it yourself

You need Python 3 installed. Then:

```
python -m venv venv
venv\Scripts\activate          (Windows)
pip install fastapi uvicorn xgboost shap scikit-learn pandas matplotlib
```

If you want to regenerate the data and retrain from scratch:

```
python generate_dataset.py
python train_model.py
python shap_demo.py
```

If you just want to run the app with the model that's already trained,
skip straight to:

```
python app.py
```

Leave that running, then open `index.html` in your browser. It talks to
the backend automatically. If the top right shows "API OFFLINE," it means
`app.py` isn't running or crashed, check the terminal for the error.

## Where things currently stand

Working: dataset, model, SHAP explanations, backend API, frontend dashboard.

Not done yet: pulling real deployment data from GitHub instead of synthetic
data, a database to store prediction history, deploying this anywhere
outside a laptop.

