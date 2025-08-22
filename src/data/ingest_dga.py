import argparse, pandas as pd, pathlib
CANON=['timestamp','H2','CH4','C2H6','C2H4','C2H2','CO','CO2','oil_temp','load_pct','fault_label','source']
COLMAPS=[{'hydrogen':'H2','methane':'CH4','ethane':'C2H6','ethylene':'C2H4','acetylene':'C2H2','carbon_monoxide':'CO','carbon_dioxide':'CO2','time':'timestamp','fault':'fault_label'},{'H2':'H2','CH4':'CH4','C2H6':'C2H6','C2H4':'C2H4','C2H2':'C2H2','CO':'CO','CO2':'CO2','timestamp':'timestamp','label':'fault_label'}]

def normalise_cols(df):
    lower={c.lower():c for c in df.columns}
    best={}; score=-1
    for mp in COLMAPS:
        s=sum(1 for k in mp if k in lower)
        if s>score: best, score = mp, s
    rename={lower[k]:v for k,v in best.items() if k in lower}
    df=df.rename(columns=rename)
    for c in CANON:
        if c not in df.columns: df[c]=None
    for c in ['H2','CH4','C2H6','C2H4','C2H2','CO','CO2','oil_temp','load_pct']:
        df[c]=pd.to_numeric(df[c], errors='coerce')
    return df[CANON]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
    a=ap.parse_args(); rawp=pathlib.Path(a.raw)
    frames=[]
    for p in rawp.glob('*.csv'):
        df=pd.read_csv(p); df=normalise_cols(df); df['source']=p.name; frames.append(df)
    if not frames: raise SystemExit('No CSV files found in raw folder')
    out=pd.concat(frames, ignore_index=True); out.to_csv(a.out, index=False)
    print('Wrote', a.out, 'rows:', len(out))
if __name__=='__main__': main()
