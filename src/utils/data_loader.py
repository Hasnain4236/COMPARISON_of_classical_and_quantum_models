import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_california_housing

def load_benchmark(name):
    """
    Loads a standard benchmark dataset.
    Returns: X (features), y (target), task_type (str)
    """
    if name == "Iris":
        data = load_iris()
        return data.data, data.target, "Classification"
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        return data.data, data.target, "Classification"
    elif name == "Wine":
        data = load_wine()
        return data.data, data.target, "Classification"
    elif name == "California Housing":
        data = fetch_california_housing()
        # For demo purposes, we might want to subsample this as it's large for quantum sim
        return data.data[:500], data.target[:500], "Regression"
    else:
        raise ValueError(f"Unknown dataset: {name}")

def preprocess_data(X, y, task_type):
    """
    Standard scaling and splitting.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    # Quantum models often prefer features in [0, 1] or [-1, 1] or [0, 2pi]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
