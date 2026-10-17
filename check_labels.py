import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/phishing_features.csv")

# Count each label
print("\n--- Label Counts ---")
print(df["label"].value_counts())

# Show examples from each label
print("\n--- Examples with Label 0 ---")
print(df[df["label"] == 0][["url", "label"]].head(10).to_string(index=False))

print("\n--- Examples with Label 1 ---")
print(df[df["label"] == 1][["url", "label"]].head(10).to_string(index=False))

# Display records with missing TLD values
print("\n--- Examples with Missing TLD ---")
print(df[df["tld"].isna()][["url", "label", "tld"]].head(10).to_string(index=False))