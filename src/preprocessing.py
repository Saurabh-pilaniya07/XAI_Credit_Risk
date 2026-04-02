import pandas as pd

def preprocess_german(df):

    df['target'] = df['credit_risk'].apply(lambda x: 1 if x == 1 else 0)
    df['age_group'] = df['age'].apply(lambda x: 'young' if x < 25 else 'adult')

    features = [
        'age', 'amount', 'duration',
        'credit_history', 'employment_duration',
        'housing', 'job'
    ]

    X = df[features]
    X = pd.get_dummies(X, drop_first=True)

    y = df['target']

    return X, y, df


import pandas as pd

def preprocess_bank(df):

    df['target'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    df['age_group'] = df['age'].apply(lambda x: 'young' if x < 25 else 'adult')

    features = [
        'age',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'emp.var.rate',
        'cons.price.idx',
        'cons.conf.idx',
        'euribor3m',
        'nr.employed'
    ]

    X = df[features]

    # No need for get_dummies (all numeric)
    y = df['target']

    return X, y, df