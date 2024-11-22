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
    filepath = "dataset/Concrete_Data.xls" # dataset file path
    try:
        data = pd.read_excel(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}") #flags error if not found

    # Separate features and target 
    features = data.iloc[:, :-1].values # all collumns except the last are features
    target = data.iloc[:, -1].values # only the last column is the target

    # Standardise features (Z-score normalisation)
    x_mean = np.mean(features, axis=0) # calculate mean
    x_std = np.std(features, axis=0) #calculate standard deviation
    x_standardised = (features - x_mean) / x_std #apply standarisation

    # Detect and cap outliers
    Q1 = np.percentile(x_standardised, 25, axis=0) #calculate first quartile
    Q3 = np.percentile(x_standardised, 75, axis=0) # calculate third quartile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR # defines lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR# defines upper bound for outliers
    x_capped = np.clip(x_standardised, lower_bound, upper_bound) # cap all outliers to the bounds instead of removing.

    # Shuffle dataset randomly
    dataset = np.hstack((x_capped, target.reshape(-1, 1)))
    np.random.seed(42)
    np.random.shuffle(dataset)

    #split into 70% train 30% test
    split_index = int(0.7 * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    #separate features and targets for each set
    train_features = train_data[:, :-1]
    train_target = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_target = test_data[:, -1]

    return train_features, train_target, test_features, test_target
