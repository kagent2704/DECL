import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Corrected Dataset URL (Google Drive Mirror)
file_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names based on dataset documentation
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "Skin_Fold", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load dataset
data = pd.read_csv(file_url, header=None, names=column_names)

# Dataset Preview
print("Dataset Preview:")
print(data.head())

# Missing Values Check
print("\nMissing values in each column:\n", data.isnull().sum())

# Drop Rows with Any NaN
data_drop_any_nan = data.dropna()
print("\nDataset after dropping rows with any NaN:")
print(data_drop_any_nan)

# Drop Rows Where All Values Are NaN
data_drop_all_nan = data.dropna(how='all')
print("\nDataset after dropping rows where all values are NaN:")
print(data_drop_all_nan)

# Drop Rows with More Than 2 NaN
data_drop_thresh = data.dropna(thresh=data.shape[1] - 2)
print("\nDataset after dropping rows with more than 2 NaN:")
print(data_drop_thresh)

# Drop Rows with NaN in 'Skin_Fold'
data_drop_column_nan = data.dropna(subset=['Skin_Fold'])
print("\nDataset after dropping rows with NaN in the 'Skin_Fold' column:")
print(data_drop_column_nan)

# Fill NaN with Default Value (0)
data_fill_default = data.fillna(0)
print("\nDataset after filling NaN with default value (0):")
print(data_fill_default)

# Impute Missing Values with Mean ('Skin_Fold')
data_impute_mean = data.copy()
data_impute_mean['Skin_Fold'] = data['Skin_Fold'].fillna(data['Skin_Fold'].mean())
print("\nDataset after imputing missing values in 'Skin_Fold' with mean:")
print(data_impute_mean)

# Impute Missing Values with Median ('Skin_Fold')
data_impute_median = data.copy()
data_impute_median['Skin_Fold'] = data['Skin_Fold'].fillna(data['Skin_Fold'].median())
print("\nDataset after imputing missing values in 'Skin_Fold' with median:")
print(data_impute_median)

# Find Duplicates
duplicates = data.duplicated()
print("\nDuplicates in the Dataset:")
print(duplicates)

# Remove Duplicates
data_no_duplicates = data.drop_duplicates()
print("\nDataset after removing duplicates:")
print(data_no_duplicates)

# Remove Redundant Rows in 'Skin_Fold' Column
data_no_redundancy = data.drop_duplicates(subset=['Skin_Fold'])
print("\nDataset after removing redundancy in the 'Skin_Fold' column:")
print(data_no_redundancy)
