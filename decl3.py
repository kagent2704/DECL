import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Corrected Dataset URL (Google Drive Mirror)
file_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names based on dataset documentation
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "Skin_Fold", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load dataset from web
df = pd.read_csv(file_url, header=None, names=column_names)

# Original Dataset Preview
print("Original Dataset:")
print(df.head())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values (Fill with Mean)
df.fillna(df.mean(), inplace=True)
print("\nDataset after handling missing values (filled with mean):")
print(df.head())

# Identify Duplicates
duplicates = df[df.duplicated()]
print("\nDuplicates:")
print(duplicates)

# Remove Duplicate Rows
df = df.drop_duplicates()
print("\nDataset after removing duplicates:")
print(df)

# Perform Pearson Correlation Analysis
correlation_matrix = df.corr(method='pearson')
print("\nCorrelation Matrix (Pearson):")
print(correlation_matrix)

# Display Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Data Visualization
# Histogram Plot
df.hist(figsize=(12, 10), bins=20)
plt.suptitle('Histogram for Each Feature')
plt.show()

# Scatter Plot
if 'Pregnancies' in df.columns and 'Glucose' in df.columns:
    plt.scatter(df['Pregnancies'], df['Glucose'])
    plt.xlabel('Pregnancies')
    plt.ylabel('Glucose')
    plt.title('Scatter Plot: Pregnancies vs Glucose')
    plt.show()

# Create a Binary Category for Glucose Levels
df['Glucose Category'] = (df['Glucose'] > 120).map({True: 'High', False: 'Low'})

# Bar Graph
plt.figure(figsize=(8, 5))
df['Glucose Category'].value_counts().plot(kind='bar', color=['green', 'orange'], edgecolor='black')
plt.title('Glucose Categories Count', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Data Scaling
scaler = MinMaxScaler()
zscore_scaler = StandardScaler()

# Select Only Numeric Columns for Scaling
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Apply Min-Max Scaling
scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)
print("\nDataset after Min-Max Scaling:")
print(scaled_df.head())

# Apply Z-Score Scaling
zscore_df = pd.DataFrame(zscore_scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)
print("\nDataset after Z-Score Scaling:")
print(zscore_df.head())

# Data Smoothing Using Binning
if 'Glucose' in df.columns:
    bins = 5  # Number of bins
    df['binned_Glucose'] = pd.cut(df['Glucose'], bins=bins, labels=False)
    print("\nDataset after Binning on 'Glucose':")
    print(df[['Glucose', 'binned_Glucose']].head())

# Feature Reduction with Selection
if 'Glucose Category' in df.columns:
    y = df['Glucose Category'].map({'High': 1, 'Low': 0})
    X = df.drop(['Glucose Category'], axis=1, errors='ignore')
    selector = SelectKBest(score_func=f_classif, k=3)
    selector.fit(X, y)

    print("\nSelected Features:")
    print(X.columns[selector.get_support()])
else:
    print("Target column not found; skipping feature selection.")
