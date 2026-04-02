import pandas as pd

def load_german():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
    return pd.read_csv(url)


def load_bank(sample_size=1500):
    url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv"
    df = pd.read_csv(url, sep=";")

    # Sample data
    df = df.sample(n=sample_size, random_state=42)

    return df