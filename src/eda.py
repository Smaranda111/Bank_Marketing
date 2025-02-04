import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define dataset path
DATA_PATH = "data/bank.csv"

# Check if the dataset exists
if not os.path.exists(DATA_PATH):
    print(f"⚠️ Dataset not found! Please download it and place it in: {DATA_PATH}")
    exit()

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n🔹 Column Names:")
print(df.columns)

# Display basic info
print("\n🔹 Dataset Overview:")
print(df.info())

print("\n🔹 First 5 Rows:")
print(df.head())

# Check missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Define the target variable
TARGET_COLUMN = "deposit"

# Check target variable distribution
if TARGET_COLUMN in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[TARGET_COLUMN])
    plt.title("Target Variable Distribution")
    plt.xlabel("Deposit Subscription (Yes/No)")
    plt.ylabel("Count")
    plt.savefig("results/target_distribution.png")
    print("✅ Saved target variable distribution as 'results/target_distribution.png'")
else:
    print(f"\n⚠️ No target column named '{TARGET_COLUMN}' found.")

# Identify column datatypes
print("\n🔹 Column Data Types:")
print(df.dtypes)

# Convert categorical columns to numerical
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes 

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("results/correlation_heatmap.png")  # Save the heatmap
print("✅ Saved correlation heatmap as 'results/correlation_heatmap.png'")

