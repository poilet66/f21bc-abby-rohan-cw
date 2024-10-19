import numpy as np
import pandas as pd #ignore: import-untyped

"""
===========================================
         Data Loading and Separation
===========================================
"""
# Load the dataset from the folder
data = pd.read_excel("dataset/Concrete_Data.xls")

# Separate features (x) and target (y)
x = data.iloc[
    :, :-1
].values  # Extract features from the dataset, [:, :-1] selects all rows and all columns except the last one (features x).
y = data.iloc[
    :, -1
].values  # Extract target variable (compressive strength), [:, -1] selects all rows and the last column (target variable y).

"""
===========================================
        Standardisation (Z-score)
===========================================
"""
# Compute the mean and standard deviation for each feature
# (x, axis=0) applies the operation along the rows (axis 0), so it processes each column independently.
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

# Apply Z-score normalisation to standardise the features
x_standardised = (x - x_mean) / x_std

"""
============================================
        Outlier Detection and Capping
============================================
"""
# Calculate the first quartile (Q1) and third quartile (Q3) for each feature
Q1 = np.percentile(x_standardised, 25, axis=0)
Q3 = np.percentile(x_standardised, 75, axis=0)
IQR = Q3 - Q1  # Compute the interquartile range (IQR)

# Define the lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers to the lower and upper bounds
x_capped = np.clip(x_standardised, lower_bound, upper_bound)

"""
===========================================
            Train-Test Split
===========================================
"""
# Combine the features (x) and target (y) into a single dataset
dataset = np.hstack((x_capped, y.reshape(-1, 1)))

# Shuffle the dataset to ensure randomisation of rows
np.random.seed(42)  # Set a seed for reproducibility
np.random.shuffle(dataset)

# Split the dataset into 70% training and 30% testing
split_index = int(0.7 * len(dataset))
train_data = dataset[:split_index]  # Training data
test_data = dataset[split_index:]  # Testing data

# Separate features (x) and target (y) for both training and testing sets
x_train = train_data[:, :-1]  # Training features
y_train = train_data[:, -1]  # Training target
x_test = test_data[:, :-1]  # Testing features
y_test = test_data[:, -1]  # Testing target


# Return training and testing data
def get_preprocessed_data():
    return x_train, y_train, x_test, y_test
