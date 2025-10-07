# Model Training Guide

This project trains models for two datasets:

1. **IEEE DataPort fault detection**
2. **ETT predictive maintenance**

> ℹ️ Configure dataset paths in your `.env` file before training. See the
> "Dataset Configuration" section of the README for the list of supported
> environment variables.

Both datasets are trained independently. After generating the unified dataset via:

```bash
python -m scripts.run_unified_gridx_pipeline
```

run the training scripts:

```bash
# IEEE DataPort models
python -m scripts.run_ieee_training

# ETT predictive maintenance models
python -m scripts.run_ett_training
```

Each script loads its dataset, trains all models, creates visualizations,
saves model artifacts under `backend/app/ml_models/trained/`, and generates
summary reports.

Run both commands so that models for both datasets are available.