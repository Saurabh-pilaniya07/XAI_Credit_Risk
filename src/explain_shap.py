import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return shap_values


def save_shap_plot(shap_values, X_test, name):
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap_{name}.png", bbox_inches='tight')
    plt.close()