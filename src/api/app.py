from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_uri = config.get("model_uri")
    if model_uri:
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        model_path = Path(config["model_path"])
        model = joblib.load(model_path)
    yield


app = FastAPI(title="ML Model API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[dict[str, str | int | float]]


class PredictResponse(BaseModel):
    predictions: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame(request.features)
    predictions = model.predict(df)
    return PredictResponse(predictions=predictions.tolist())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.get("port", 8000))
