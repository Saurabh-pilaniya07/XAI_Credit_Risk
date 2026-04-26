# Research Findings: Explainable AI & Fairness in Credit Decision Systems

## Summary
This project builds a multi-dataset XAI pipeline using SHAP, LIME, and
counterfactual analysis to understand how ML models make decisions in
financial contexts. The core finding is that decision drivers — and
therefore fairness risks — are fundamentally domain-dependent.

**Dataset details:**

| Dataset | Shape | Train/Test | Features | Test Accuracy | CV Accuracy | Test ROC-AUC |
|---------|-------|-----------|----------|--------------|-------------|-------------|
| German Credit | (1000, 16) | 800 / 200 | 16 (after preprocessing) | 0.695 | 0.690 ± 0.024 | 0.682 |
| Bank Marketing | (1500, 10) | 1200 / 300 | 10 (after preprocessing) | 0.863 | 0.903 ± 0.008 | 0.831 |

---

## Finding 1: Decision Drivers Are Domain-Dependent

- German Credit dataset: **age** is the primary decision driver (demographic)
- Bank Marketing dataset: **duration** is the primary decision driver (behavioral)

The same algorithm (Random Forest), applied to two financial datasets, 
produces models with fundamentally different fairness risk profiles — 
not because of the algorithm, but because of what each domain rewards.

**Implication:** Fairness risk cannot be assessed at the algorithm level. 
It must be assessed at the domain + data + deployment context level.

---

## Finding 2: Models Are Sensitive Near Decision Boundaries — German Credit Far More Than Bank

The full counterfactual analysis (perturbation factors: ×0.3, ×0.5, ×0.7, ×0.9) revealed
dramatically different boundary sensitivity between the two datasets:

**German Credit — Top decision-sensitive features:**

| Feature | Factor | Flip Rate | Fairness Implication |
|---------|--------|-----------|---------------------|
| `amount` | ×0.3 | **38%** | Loan amount sits very close to boundary |
| `age` | ×0.3 | **26%** | Protected attribute — high risk |
| `amount` | ×0.5 | 22% | Sensitivity persists across mild perturbation |
| `age` | ×0.5 | 20% | Age influence robust across factors |
| `duration` | ×0.7 | 18% | Loan term also near boundary |

**Bank Marketing — All features (much lower sensitivity):**

| Max Flip Rate | Features | Interpretation |
|--------------|---------|----------------|
| **2%** | `age`, `duration`, `euribor3m`, `nr.employed` | Very robust — model is far from boundary |

**The critical contrast:**
German Credit's maximum flip rate (38%) is **19× higher** than Bank Marketing's (2%).
This means German Credit decisions are structurally more fragile — small real-world
changes in loan amount or applicant age can flip credit outcomes for nearly 4 in 10 applicants.

This indicates that many applicants sit close to the decision boundary in German Credit —
meaning small real-world changes (a slightly different loan amount, one more year of age)
can change outcomes. This is particularly concerning for high-stakes financial decisions
and directly relevant to EU AI Act Art. 14 (human oversight) requirements.

---

## Finding 3: Explainability Methods Reveal Different Things

| Method | What It Shows | What It Cannot Show |
|--------|--------------|---------------------|
| SHAP | Which features matter globally | Whether those features are appropriate |
| LIME | Why a specific prediction was made | Whether the reasoning is consistent |
| Counterfactual | What would change the outcome | Whether that change is realistic |

No single method is sufficient. A complete explainability audit requires 
all three — and even then, the results require human and regulatory judgment.

---

## Finding 4: High Accuracy Does Not Mean Low Fairness Risk

| Dataset | Test Accuracy | CV Accuracy | ROC-AUC | Primary Feature | Fairness Risk |
|---------|--------------|-------------|---------|-----------------|---------------|
| Bank Marketing | **0.863** | 0.903 ± 0.008 | 0.831 | `duration` (behavioral) | Lower |
| German Credit | **0.695** | 0.690 ± 0.024 | 0.682 | `age` (demographic) | **High** |

The Bank Marketing model achieves higher accuracy (test: 0.863, CV: 0.903) while
relying on `duration` — a feature that could be **gamed** (longer calls could be
artificially extended to inflate approval probability).

The German Credit model at lower accuracy (test: 0.695, CV: 0.690) relies on `age`
— a feature that is **immutable and protected**: applicants cannot change their age
to receive a better credit decision. Age ranked 2nd in counterfactual sensitivity
with a 26% flip rate at ×0.3 perturbation.

Higher accuracy did not correlate with lower fairness risk. The higher-accuracy model
(Bank Marketing) has a different failure mode — gameable features — while the
lower-accuracy model (German Credit) has a worse failure mode — protected characteristic reliance.

---

## Open Research Questions

1. Does domain context systematically predict whether AI systems will 
   exhibit demographic vs behavioural decision patterns?
2. Can counterfactual analysis be used to design fairer feature 
   engineering pipelines before model training?
3. How should EU AI Act Article 13 (transparency) requirements account 
   for domain-specific explainability differences?

---

## Connection to EU AI Act

Both use cases (credit scoring, marketing) involve automated decisions 
about individuals. Under EU AI Act Annex III, credit scoring is 
explicitly high-risk. This project demonstrates that:

- Technical explainability (SHAP/LIME) is necessary but not sufficient
- Domain-aware fairness audits are required
- Counterfactual analysis provides actionable compliance evidence

---

*Part of a broader portfolio on Responsible AI: Fairness, 
Explainability, and Governance.*