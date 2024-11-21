import numpy as np
import pandas as pd


def get_preprocessed_data():
    """
    Load and preprocess the dataset.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        tuple: Preprocessed training and testing data.

    """
    filepath = "dataset/Concrete_Data.xls"
    try:
        data = pd.read_excel(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    # Separate features (x) and target (y)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Standardise features (Z-score normalisation)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_standardised = (x - x_mean) / x_std

    # Detect and cap outliers
    Q1 = np.percentile(x_standardised, 25, axis=0)
    Q3 = np.percentile(x_standardised, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    x_capped = np.clip(x_standardised, lower_bound, upper_bound)

    # Shuffle and split into training and testing sets
    dataset = np.hstack((x_capped, y.reshape(-1, 1)))
    np.random.seed(42)
    np.random.shuffle(dataset)

    split_index = int(0.7 * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    train_features = train_data[:, :-1]
    train_target = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_target = test_data[:, -1]

    return train_features, train_target, test_features, test_target
