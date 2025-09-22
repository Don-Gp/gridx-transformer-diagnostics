import types
import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def _stub_deps():
    """Insert lightweight stubs for heavy optional dependencies."""
    # tensorflow stub
    tf = types.ModuleType("tensorflow")
    tf.random = types.SimpleNamespace(set_seed=lambda seed: None)
    keras = types.SimpleNamespace(
        models=types.SimpleNamespace(Sequential=object),
        layers=types.SimpleNamespace(
            LSTM=object,
            Dense=object,
            Dropout=object,
            BatchNormalization=object,
            Conv1D=object,
            MaxPooling1D=object,
            Flatten=object,
            Reshape=object,
        ),
        optimizers=types.SimpleNamespace(Adam=object),
        callbacks=types.SimpleNamespace(EarlyStopping=object, ReduceLROnPlateau=object),
        utils=types.SimpleNamespace(to_categorical=lambda y, num_classes=None: y),
    )
    tf.keras = keras
    sys.modules.setdefault("tensorflow", tf)
    sys.modules.setdefault("tensorflow.keras", keras)
    sys.modules.setdefault("tensorflow.keras.models", keras.models)
    sys.modules.setdefault("tensorflow.keras.layers", keras.layers)
    sys.modules.setdefault("tensorflow.keras.optimizers", keras.optimizers)
    sys.modules.setdefault("tensorflow.keras.callbacks", keras.callbacks)
    sys.modules.setdefault("tensorflow.keras.utils", keras.utils)

    # statsmodels stub
    sm = types.ModuleType("statsmodels")
    tsa = types.ModuleType("statsmodels.tsa")
    seasonal = types.ModuleType("statsmodels.tsa.seasonal")
    seasonal.seasonal_decompose = lambda *a, **k: None
    arima_model = types.ModuleType("statsmodels.tsa.arima.model")
    arima_model.ARIMA = object
    tsa.seasonal = seasonal
    tsa.arima = types.SimpleNamespace(model=arima_model)
    sm.tsa = tsa
    sys.modules.setdefault("statsmodels", sm)
    sys.modules.setdefault("statsmodels.tsa", tsa)
    sys.modules.setdefault("statsmodels.tsa.seasonal", seasonal)
    sys.modules.setdefault("statsmodels.tsa.arima", tsa.arima)
    sys.modules.setdefault("statsmodels.tsa.arima.model", arima_model)

    # matplotlib and seaborn stubs
    mpl = types.ModuleType("matplotlib")
    mpl.pyplot = types.SimpleNamespace()
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", mpl.pyplot)
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))


_def_stub = _stub_deps()  # run at import time


def test_train_maintenance_classifier(tmp_path):
    from backend.app.ml_models.ett_predictive_models import ETTPredictiveMaintenance

    # create tiny synthetic dataset
    X = np.random.rand(20, 5)
    y_maint = np.random.randint(0, 2, 20)
    y_urg = np.random.randint(0, 4, 20)
    scaler = StandardScaler().fit(X)

    data = {
        'ett_maintenance': {
            'X_train': X[:12],
            'X_val': X[12:16],
            'X_test': X[16:],
            'y_maintenance_train': y_maint[:12],
            'y_maintenance_val': y_maint[12:16],
            'y_maintenance_test': y_maint[16:],
            'y_urgency_train': y_urg[:12],
            'y_urgency_val': y_urg[12:16],
            'y_urgency_test': y_urg[16:],
            'scaler': scaler,
        }
    }
    data_file = tmp_path / "data.pkl"
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    model = ETTPredictiveMaintenance(data_path=str(data_file))
    clf = model.train_maintenance_classifier(tune_hyperparameters=False)

    # model should produce predictions for test set
    preds = model.results['maintenance_classifier']['predictions']
    assert len(preds) == len(data['ett_maintenance']['y_maintenance_test'])
    assert hasattr(clf, 'predict')