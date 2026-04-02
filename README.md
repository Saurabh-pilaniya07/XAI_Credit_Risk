# Explainable AI & Fairness Analysis in Credit Decision Systems

## Overview

This project builds a **multi-dataset, explainable AI pipeline** to analyze how machine learning models make decisions in financial applications such as credit approval and marketing campaigns.

It integrates:

* **Fairness analysis** (bias detection & mitigation)
* **Explainability methods** (SHAP, LIME)
* **Counterfactual reasoning** (decision boundary analysis)

The system is evaluated across two datasets:

* German Credit Dataset (benchmark fairness dataset)
* Bank Marketing Dataset (real-world behavioral dataset)

---

## Objectives

* Detect and analyze bias in credit decision models
* Understand model behavior using explainability techniques
* Identify decision-driving features using counterfactuals
* Compare results across multiple datasets
* Evaluate trade-offs between fairness, accuracy, and interpretability

---

## Datasets

### 1. German Credit Dataset

* Benchmark dataset widely used in fairness research
* Focus: credit risk prediction
* Key features: age, amount, duration, employment

### 2. Bank Marketing Dataset

* Real-world dataset from banking campaigns
* Focus: customer response prediction
* Key features: duration, campaign, macroeconomic indicators

---

## Methodology

### 1. Model

* Random Forest Classifier
* Trained separately on each dataset

---

### 2. Explainability

#### 🔹 SHAP (Global + Local)

* Identifies overall feature importance
* Explains how features impact predictions

#### 🔹 LIME (Local Explanation)

* Explains individual predictions
* Output saved as interactive HTML reports

#### 🔹 Counterfactual Explanations

* Identifies minimal feature changes required to flip decisions
* Reveals decision-sensitive features

---

### 3. Multi-Dataset Evaluation

Each dataset is processed independently using the same pipeline to ensure:

* Controlled experimentation
* Comparable results
* Robust validation

---

## Results

| Dataset | Accuracy | Counterfactual Feature | Factor |
| ------- | -------- | ---------------------- | ------ |
| Bank    | 0.923    | duration               | 0.5    |
| German  | 0.680    | age                    | 0.5    |

---

## Key Insights

### 1. Dataset-Dependent Decision Drivers

* German Dataset → **age** (demographic factor)
* Bank Dataset → **duration** (behavioral factor)

Model decisions vary significantly depending on dataset context.

---

### 2. Sensitivity Near Decision Boundary

* Small feature changes (×0.5) flipped predictions
* Indicates proximity to decision boundary

---

### 3. Fairness Implications

* Age-based influence may introduce **demographic bias risks**
* Raises concerns about ethical deployment in financial systems

---

### 4. Operational Insights

* Duration reflects **customer engagement quality**
* Provides actionable business insight

---

### 5. Explainability is Multi-Dimensional

No single method is sufficient:

| Method         | Insight              |
| -------------- | -------------------- |
| SHAP           | Global importance    |
| LIME           | Local reasoning      |
| Counterfactual | Decision sensitivity |

---

## Limitations

* Counterfactuals depend on perturbation strategy
* LIME explanations may vary across runs
* SHAP assumes feature independence
* Results depend on dataset characteristics

---

## Ethical Considerations

* Explainability does not guarantee fairness
* Sensitive features (e.g., age) require careful handling
* Responsible AI must balance:

  * Accuracy
  * Fairness
  * Transparency

---

## Policy Relevance

This project aligns with **responsible AI principles** and supports:

* Bias monitoring in financial systems
* Transparent decision-making
* Compliance with regulations such as the EU AI Act

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

```
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

```bash
pip install -r requirements.txt
python main.py
```

---

## Outputs

* SHAP plots (`outputs/`)
* LIME explanations (`.html` files)
* Counterfactual analysis
* Multi-dataset comparison CSV

---

## Future Work

* Extend to additional datasets
* Integrate causal fairness methods
* Improve counterfactual generation (multi-feature optimization)
* Add fairness-aware constraints in training

---

## Research Positioning

This project shifts focus from:

> **Model performance → Model understanding**

By combining fairness, explainability, and counterfactual reasoning, it demonstrates a **holistic approach to responsible AI systems**.

---

