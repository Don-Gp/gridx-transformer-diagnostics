import pandas as pd
from src.data.ingest_dga import normalise_cols, CANON

def test_normalise_cols_maps_and_types():
    df = pd.DataFrame({
        'hydrogen': ['1'],
        'methane': ['2'],
        'ethane': ['3'],
        'ethylene': ['4'],
        'acetylene': ['5'],
        'carbon_monoxide': ['6'],
        'carbon_dioxide': ['7'],
        'oil_temp': ['80'],
        'load_pct': ['90'],
        'time': ['2020-01-01'],
        'fault': ['none']
    })
    normalized = normalise_cols(df)
    # ensure columns match canonical order
    assert list(normalized.columns) == CANON
    # numeric fields converted to float
    assert normalized['H2'].dtype.kind in 'fi'
    assert normalized['load_pct'].iloc[0] == 90.0
    # missing column 'source' filled with NaN
    assert normalized['source'].isna().all()