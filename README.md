# Explainable AI & Fairness Analysis in Credit Decision Systems

---

## Overview

This project develops a **multi-dataset explainable AI pipeline** to analyze how machine learning models make decisions in financial applications such as credit approval and marketing campaigns.

The system integrates:

* Fairness analysis (bias detection and implications)
* Explainability techniques (SHAP, LIME)
* Counterfactual reasoning (decision boundary analysis)

The pipeline is evaluated across two datasets:

* German Credit Dataset (benchmark fairness dataset)
* Bank Marketing Dataset (real-world behavioral dataset)

---

## Objectives

* Analyze bias and fairness risks in credit decision systems
* Understand model behavior using explainability techniques
* Identify decision-driving features through counterfactual analysis
* Compare model behavior across datasets
* Examine trade-offs between interpretability, fairness, and performance

---

## Datasets

### German Credit Dataset (https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv)

* Benchmark dataset widely used in fairness research
* Task: credit risk prediction
* Key features: age, amount, duration, employment

---

### Bank Marketing Dataset (https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv)

* Real-world dataset from banking campaigns
* Task: customer response prediction
* Key features: duration, campaign, macroeconomic indicators

---

## Methodology

### Model

* Random Forest Classifier
* Trained independently on each dataset
* Ensures controlled and comparable evaluation

---

### Explainability Techniques

#### SHAP (Global + Local)

* Identifies global feature importance
* Explains contribution of features to predictions

#### LIME (Local Explanation)

* Explains individual predictions
* Outputs stored as `.html` files

#### Counterfactual Explanations

* Identifies minimal feature changes to flip predictions
* Reveals decision sensitivity and boundary behavior

---

### Multi-Dataset Evaluation

Each dataset is processed using the same pipeline to ensure:

* Controlled experimentation
* Comparable results
* Robust analysis across domains

---

## Results

| Dataset        | Accuracy | Primary Decision Driver | Type        |
| -------------- | -------- | ----------------------- | ----------- |
| Bank Marketing | 0.923    | duration                | Behavioral  |
| German Credit  | 0.680    | age                     | Demographic |

---

## SHAP Explainability Visualizations

### German Credit Dataset — Feature Importance

Age is the dominant decision driver, representing a protected demographic attribute.

![SHAP German](outputs/shap_german.png)

---

### Bank Marketing Dataset — Feature Importance

Duration (call length) is the dominant driver, reflecting user behavior.

![SHAP Bank](outputs/shap_bank.png)

---

## Key Observation

The two datasets exhibit fundamentally different decision patterns:

* German Credit → decisions driven by **who the individual is (age)**
* Bank Marketing → decisions driven by **what the individual does (behavior)**

This distinction directly affects fairness risk assessment.

---

## Cross-Dataset Analysis: Why Decision Drivers Differ

### Core Finding

| Dataset        | Top Feature | Feature Type             | Fairness Risk |
| -------------- | ----------- | ------------------------ | ------------- |
| German Credit  | age         | Demographic (protected)  | High          |
| Bank Marketing | duration    | Behavioral (unprotected) | Low           |

---

### Interpretation

**German Credit → Age-Driven Decisions**
The model relies on a protected demographic attribute (age) for decision-making.
Even if predictive, this introduces potential discrimination risk under:

* EU AI Act (high-risk systems)
* GDPR (sensitive data considerations)

---

**Bank Marketing → Duration-Driven Decisions**
Call duration reflects user engagement and behavioral patterns.
This signal is not inherently linked to demographic identity, making it a lower fairness-risk factor.

---

### Research Question

> Does dataset context determine fairness risk more than the algorithm itself?

If true, fairness evaluation must be:

* Domain-specific
* Context-aware
* Not limited to generic technical metrics

---

### Why Explainability Alone Is Not Enough

Explainability methods (e.g., SHAP) reveal:

* Which features influence decisions

However, they do not determine:

* Whether those decisions are ethically acceptable

This requires integration with:

* Policy frameworks
* Ethical reasoning
* Domain knowledge

---

## Outputs

* SHAP plots (`outputs/`) — committed to repository
* LIME explanations — generated locally as `.html` files (not tracked due to size)
* Counterfactual analysis results
* Multi-dataset comparison CSV

---

## Configuration

All pipeline settings are controlled through `config.py`.

This allows modification of datasets, model behavior, and explainability parameters without altering core logic.

### Configuration Parameters

| Parameter               | Description                                     |
| ----------------------- | ----------------------------------------------- |
| `DATASETS`              | Datasets to run (`german`, `bank`)              |
| `DATASET_CONFIG`        | Dataset-specific settings (e.g., sampling size) |
| `MODEL_CONFIG`          | Model hyperparameters                           |
| `SPLIT_CONFIG`          | Train/test split settings                       |
| `OUTPUT_CONFIG`         | Controls saving of outputs and visualizations   |
| `EXPLAINABILITY_CONFIG` | Toggles SHAP, LIME, and counterfactual analysis |
---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* SHAP
* LIME
* Matplotlib

---

## Project Structure

```id="8mkc7y"
xai-multi-dataset/
│
├── data/
├── notebooks/
├── src/
├── outputs/
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash id="t2yq3s"
pip install -r requirements.txt
python main.py
```

---

## Limitations

* Counterfactual explanations depend on perturbation strategy
* LIME explanations may vary across runs
* SHAP assumes feature independence
* Results are dataset-dependent

---

## Ethical Considerations

* Explainability does not guarantee fairness
* Sensitive attributes require careful handling
* Responsible AI must balance:

  * Accuracy
  * Fairness
  * Transparency

---

## Policy Relevance

This project aligns with responsible AI principles and supports:

* Bias monitoring in financial systems
* Transparent decision-making
* Compliance with regulations such as the EU AI Act

---

## Research Positioning

This work represents a shift from:

**Model performance → Model understanding and ethical evaluation**

It demonstrates that:

* Model behavior varies across datasets
* Explainability reveals decision logic
* Fairness risks depend on feature usage

---

## Conclusion

Two models with similar performance can exhibit fundamentally different ethical risks depending on their decision drivers.

* German Credit → High fairness risk (demographic reliance)
* Bank Marketing → Lower fairness risk (behavioral reliance)

This highlights the need for:

* Context-aware AI evaluation
* Domain-specific fairness assessment
* Policy-aligned deployment of machine learning systems
