from pydantic import BaseModel,Field
class PredictionRequest(BaseModel):
    age:float=Field(ge=18,le=100)
    sessions:float=Field(ge=0)
    purchases:float=Field(ge=0)
    average_session_minutes:float=Field(alias="averageSessionMinutes",ge=0)
    channel:str=Field(pattern="^(organic|search|social|referral)$")
class PredictionResponse(BaseModel):
    label:int
    probability:float
class ModelInfoResponse(BaseModel):
    model_type:str
    artifact_loaded:bool
    features:list[str]
