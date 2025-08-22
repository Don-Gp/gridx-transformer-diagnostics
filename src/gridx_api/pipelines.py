import joblib, json
import numpy as np, pandas as pd
from pathlib import Path
MODEL_DIR=Path(__file__).parent/'models'
MODEL_PATH=MODEL_DIR/'dga_baseline.joblib'
DEFAULT_CLASSES=['healthy','thermal_T1','thermal_T2','high_energy','PD']
FEATS=['H2','CH4','C2H6','C2H4','C2H2','CO','CO2','oil_temp','load_pct']

def load_model():
    if MODEL_PATH.exists():
        m=joblib.load(MODEL_PATH); return m, {'version':'dga_baseline_v0','classes':getattr(m,'classes_',DEFAULT_CLASSES)}
    class M: classes_=np.array(DEFAULT_CLASSES); 
    	def predict_proba(self,X):
            n=len(X); p=np.random.default_rng(0).random((n,len(self.classes_))); return p/p.sum(1,keepdims=True)
    return M(), {'version':'mock_v0','classes':DEFAULT_CLASSES}

def preprocess(df):
    df=df.copy()
    for c in FEATS:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
    return df[FEATS]

def predict_and_explain(m, meta, X):
    p=m.predict_proba(X.values)
    labels=np.array(meta.get('classes',DEFAULT_CLASSES))
    return labels, p, [[] for _ in range(len(X))]
