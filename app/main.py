
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
from pathlib import Path

app = FastAPI(title="MLOps Template API")
MODEL = None

class PredictRequest(BaseModel):
    features: list

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load(Path("artifacts")/"model.pkl")
    return MODEL

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model()
    X = np.array(req.features, dtype=float)
    preds = model.predict(X).tolist()
    return {"predictions": preds}
