# Research Findings: Explainable AI & Fairness in Credit Decision Systems

## Summary
This project builds a multi-dataset XAI pipeline using SHAP, LIME, and 
counterfactual analysis to understand how ML models make decisions in 
financial contexts. The core finding is that decision drivers — and 
therefore fairness risks — are fundamentally domain-dependent.

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

## Finding 2: Models Are Sensitive Near Decision Boundaries

A ×0.5 perturbation to the primary feature flipped predictions in both datasets:
- Bank: halving `duration` changed the prediction
- German: halving `age` changed the prediction

This indicates that many applicants sit close to the decision boundary — 
meaning small real-world changes (one more year of age, a slightly 
shorter call) can change outcomes. This is particularly concerning for 
high-stakes financial decisions.

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

The Bank Marketing model achieved 0.923 accuracy while relying on 
`duration` — a feature that could be manipulated (longer calls could 
be gamed). The German model at 0.680 accuracy uses `age` — a protected 
characteristic it should not rely on.

Higher accuracy did not correlate with lower fairness risk. In fact, 
the higher-accuracy model may be learning a spurious correlation.

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