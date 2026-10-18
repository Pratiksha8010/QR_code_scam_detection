import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the original dataset
df = pd.read_csv("dataset/phishing_features.csv")

print("Original dataset shape:", df.shape)

# Step 2: Extract hostnames from URLs
df["domain"] = (
    df["url"]
    .astype(str)
    .str.extract(r"^(?:https?://)?([^/:?#]+)", expand=False)
    .str.lower()
)

# Step 3: Remove rows where hostname extraction failed
df = df.dropna(subset=["domain"]).copy()

# Step 4: Split the dataset into training and testing sets
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Step 5: Get unique hostnames from each set
train_domains = set(train_df["domain"])
test_domains = set(test_df["domain"])

# Step 6: Find hostnames appearing in both sets
shared_domains = train_domains.intersection(test_domains)

# Step 7: Display the results
print("\n--- Domain Split Analysis ---")

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nUnique training hostnames:", len(train_domains))
print("Unique testing hostnames:", len(test_domains))

print("\nHostnames present in both sets:", len(shared_domains))

if len(test_domains) > 0:
    overlap_percentage = (
        len(shared_domains) / len(test_domains)
    ) * 100

    print(
        "Percentage of test hostnames also in training:",
        round(overlap_percentage, 2),
        "%"
    )