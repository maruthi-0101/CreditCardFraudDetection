import pandas as pd

# Load dataset
df = pd.read_csv("creditcard.csv")

# Check for missing values
print("\n🔍 Checking for Missing Values:")
print(df.isnull().sum())

# Check fraud vs. non-fraud counts
print("\n📊 Class Distribution:")
print(df["Class"].value_counts())

# Print sample fraud & safe transactions
print("\n✅ Sample Safe Transactions:")
print(df[df["Class"] == 0].head())

print("\n⚠️ Sample Fraudulent Transactions:")
print(df[df["Class"] == 1].head())