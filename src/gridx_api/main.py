from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, io
from .schemas import PredictRequest, PredictResponse
from .pipelines import load_model, preprocess, predict_and_explain
app=FastAPI(title='GRIDx API', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
model, meta = load_model()
@app.get('/health')
def health(): return {'status':'ok','model_version':meta.get('version','unknown')}
@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    import pandas as pd
    df=pd.DataFrame([r.dict() for r in req.records]); X=preprocess(df)
    labels, proba, expl = predict_and_explain(model, meta, X)
    idx=proba.argmax(axis=1)
    preds=[str(labels[i]) for i in idx]
    probs=[{str(labels[j]):float(p[j]) for j in range(len(labels))} for p in proba]
    return PredictResponse(model_version=meta.get('version','unknown'),predictions=preds,probabilities=probs)
@app.post('/predict_csv', response_model=PredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    df=pd.read_csv(io.BytesIO(await file.read())); X=preprocess(df)
    labels, proba, expl = predict_and_explain(model, meta, X)
    idx=proba.argmax(axis=1)
    preds=[str(labels[i]) for i in idx]
    probs=[{str(labels[j]):float(p[j]) for j in range(len(labels))} for p in proba]
    return PredictResponse(model_version=meta.get('version','unknown'),predictions=preds,probabilities=probs)
