from pydantic import BaseModel
from typing import List, Optional, Dict
class DGARecord(BaseModel):
    H2: float; CH4: float; C2H6: float; C2H4: float; C2H2: float; CO: float; CO2: float; oil_temp: float; load_pct: float
    timestamp: Optional[str]=None
class PredictRequest(BaseModel):
    records: List[DGARecord]
class PredictResponse(BaseModel):
    model_version: str
    predictions: List[str]
    probabilities: List[Dict[str, float]]
