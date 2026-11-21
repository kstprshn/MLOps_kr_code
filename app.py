"""Q47: REST-сервис для модели (FastAPI)."""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("model.joblib")
features = joblib.load("features.joblib")
app = FastAPI(title="ML model service")


class Request(BaseModel):
    features: dict  # {"SepalLengthCm": 5.1, ...}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame([req.features], columns=features)
    return {"prediction": str(model.predict(df)[0])}
