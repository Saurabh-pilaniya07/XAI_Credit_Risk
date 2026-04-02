import pandas as pd

def generate_counterfactual(model, sample):

    sample = sample.copy()
    sample_df = pd.DataFrame([sample], columns=sample.index)
    original = model.predict(sample_df)[0]

    for feature in sample.index:

        for factor in [0.5, 1.5, 2]:

            temp = sample.copy()
            temp[feature] = sample[feature] * factor

            temp_df = pd.DataFrame([temp], columns=sample.index)
            new_pred = model.predict(temp_df)[0]

            if new_pred != original:
                return {
                    "feature": feature,
                    "factor": factor,
                    "from": int(original),
                    "to": int(new_pred)
                }

    return None
