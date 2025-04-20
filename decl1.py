import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

# Fetch dataset from UCI ML Repository (Wine Dataset ID = 109)
wine = fetch_ucirepo(id=109)  

# Extract features (X) and targets (y)
data = wine.data.features 
data['target'] = wine.data.targets  # Add target column

# Print available column names to check correctness
print("\nAvailable Columns in Dataset:\n", data.columns)

# Standardizing column names
data.columns = data.columns.str.replace(' ', '_').str.lower()

# Basic Dataset Information
print("\nDataset Shape:", data.shape)
print("\nFirst 5 rows of the dataset:\n", data.head())
print("\nData Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Check Missing Values
print("\nMissing values in each column:\n", data.isnull().sum())

# Statistical Analysis
print("\nMean of Proline:\n", data['proline'].mean())  
print("\nMedian of Ash:\n", data['ash'].median())  
print("\nMode of Hue:\n", data['hue'].mode().iloc[0])  
print("\nStandard Deviation of Proline:\n", data['proline'].std())  

# Histogram: Alcohol Distribution
plt.figure(figsize=(8, 5))
plt.hist(data['alcohol'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Alcohol')
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: Alcohol vs Malic Acid (Correcting Column Name)
malic_acid_col = [col for col in data.columns if "malic" in col.lower()]
print("\nIdentified Malic Acid Column:", malic_acid_col)

if malic_acid_col:
    plt.figure(figsize=(8, 5))
    plt.scatter(data['alcohol'], data[malic_acid_col[0]], color='red', alpha=0.7)
    plt.title('Alcohol vs Malic Acid')
    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    plt.show()
else:
    print("Malic Acid column not found!")
