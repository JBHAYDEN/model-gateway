from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from app.predictor import Predictor
from app.schemas import *
from ml.pipeline import NUMERIC,CATEGORICAL
from ml.train import train_and_save
predictor:Predictor|None=None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global predictor
    try: predictor=Predictor()
    except FileNotFoundError: train_and_save(); predictor=Predictor()
    yield
    predictor=None
app=FastAPI(title="Model Gateway",version="1.0.0",lifespan=lifespan)
@app.get("/health")
def health()->dict[str,str]: return {"status":"ok"}
@app.get("/model-info",response_model=ModelInfoResponse)
def model_info()->ModelInfoResponse:
    return ModelInfoResponse(model_type="LogisticRegression",artifact_loaded=predictor is not None,features=NUMERIC+CATEGORICAL)
@app.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest)->PredictionResponse:
    if predictor is None: raise HTTPException(status_code=503,detail="Model not ready")
    try: return predictor.predict(request)
    except Exception as exc: raise HTTPException(status_code=500,detail="Prediction failed") from exc
