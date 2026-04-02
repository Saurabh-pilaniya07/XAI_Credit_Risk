DATASETS = ["german", "bank"]

DATASET_CONFIG = {
    "bank_sample_size": 1500
}


# Model Settings
MODEL_CONFIG = {
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": None
}

# Train/Test Split
SPLIT_CONFIG = {
    "test_size": 0.2,
    "random_state": 42
}

# Output Settings
OUTPUT_CONFIG = {
    "save_plots": True,
    "save_lime": True,
    "save_comparison": True,
    "output_dir": "outputs"
}

# Explainability Settings
EXPLAINABILITY_CONFIG = {
    "use_shap": True,
    "use_lime": True,
    "use_counterfactual": True
}
