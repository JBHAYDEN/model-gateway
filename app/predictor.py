from pathlib import Path
import joblib,pandas as pd
from app.schemas import PredictionRequest,PredictionResponse
from ml.features import add_behavior_features
class Predictor:
    def __init__(self,model_path:str|Path="artifacts/conversion_model.joblib"):
        self.model_path=Path(model_path)
        if not self.model_path.exists(): raise FileNotFoundError(self.model_path)
        self.model=joblib.load(self.model_path)
    def predict(self,request:PredictionRequest)->PredictionResponse:
        frame=add_behavior_features(pd.DataFrame([request.model_dump(by_alias=False)]))
        label=int(self.model.predict(frame)[0]); probability=float(self.model.predict_proba(frame)[0][1])
        return PredictionResponse(label=label,probability=round(probability,6))
