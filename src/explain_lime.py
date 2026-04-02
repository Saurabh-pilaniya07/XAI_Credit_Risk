from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

def lime_explanation(model, X_train, X_test, name):

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['Bad', 'Good'],
        mode='classification'
    )

    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    )

    exp.save_to_file(f"outputs/lime_{name}.html")

    return exp