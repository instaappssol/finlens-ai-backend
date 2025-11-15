# Models Directory

This directory stores trained ML models for transaction categorization.

## Structure

```
models/
├── transaction_categorizer_latest.joblib    # Latest trained model
├── transaction_categorizer_v1.0.0.joblib    # Versioned models
└── model_metadata.json                      # Model metadata and versioning info
```

## Model Files

- **Format**: `.joblib` (for scikit-learn models) or `.pkl` (for other models)
- **Naming**: `{model_name}_{version}.{extension}`
- **Storage**: Managed by `ModelManager` service

## Metadata

The `model_metadata.json` file tracks:
- Model versions
- Training dates
- Training samples count
- Categories supported
- Current active version

## Usage

Models are automatically loaded by the `CategorizationService` when the API starts.
To train a new model, use the `/transactions/train-model` endpoint.

