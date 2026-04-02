import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_loader import load_german, load_bank
from src.preprocessing import preprocess_german, preprocess_bank
from src.model import train_model
from src.evaluate import evaluate_model
from src.explain_shap import shap_analysis, save_shap_plot
from src.explain_lime import lime_explanation
from src.counterfactual import generate_counterfactual


# ================================
# Setup
# ================================
if not os.path.exists("outputs"):
    os.makedirs("outputs")


# ================================
# Dataset Loop
# ================================
from config import DATASETS, SPLIT_CONFIG, OUTPUT_CONFIG, DATASET_CONFIG
datasets = DATASETS

results = []

for ds in datasets:

    print("\n===============================")
    print(f"Running for dataset: {ds}")
    print("===============================")

    # ----------------------------
    # Load + Preprocess
    # ----------------------------
    if ds == "german":
        df = load_german()
        X, y, df = preprocess_german(df)

    elif ds == "bank":
        df = load_bank(sample_size=DATASET_CONFIG["bank_sample_size"])
        X, y, df = preprocess_bank(df)

    # ----------------------------
    # Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=SPLIT_CONFIG["test_size"],
        random_state=SPLIT_CONFIG["random_state"]
    )

    # ----------------------------
    # Train
    # ----------------------------
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = evaluate_model(y_test, y_pred)

    print("Accuracy:", acc)

    # ----------------------------
    # SHAP
    # ----------------------------
    print("Running SHAP...")
    shap_values = shap_analysis(model, X_test)
    save_shap_plot(shap_values, X_test, ds)

    # ----------------------------
    # LIME
    # ----------------------------
    print("Running LIME...")
    lime_explanation(model, X_train, X_test, ds)

    # ----------------------------
    # Counterfactual
    # ----------------------------
    print("Generating counterfactual...")
    sample = X_test.iloc[0]
    cf = generate_counterfactual(model, sample)

    print("Counterfactual:", cf)

    # ----------------------------
    # Store Results
    # ----------------------------
    results.append({
        "dataset": ds,
        "accuracy": acc,
        "counterfactual_feature": cf.get("feature") if cf else None,
        "counterfactual_factor": cf.get("factor") if cf else None
    })


# ================================
# Final Comparison Table
# ================================
print("\n===============================")
print("FINAL COMPARISON")
print("===============================")

results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)

print(results_df)

# Save comparison
results_df.to_csv("outputs/comparison_results.csv", index=False)

print("\nResults saved to outputs/comparison_results.csv")