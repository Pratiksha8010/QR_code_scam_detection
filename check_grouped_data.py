import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load the original dataset
df = pd.read_csv("dataset/phishing_features.csv")

# 2. Extract hostname from each URL
df["domain"] = (
    df["url"]
    .astype(str)
    .str.extract(r"^(?:https?://)?([^/:?#]+)", expand=False)
    .str.lower()
)

# 3. Remove records without a hostname
df = df.dropna(subset=["domain"]).copy()

print("Dataset shape:", df.shape)

# 4. Split the dataset by hostname
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(df, df["label"], groups=df["domain"])
)

train_data = df.iloc[train_indices]
test_data = df.iloc[test_indices]

# 5. Display training and testing records
print("\n--- Dataset Split ---")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

# 6. Display label distribution
print("\n--- Training Label Distribution ---")
print(train_data["label"].value_counts().sort_index())

print("\n--- Testing Label Distribution ---")
print(test_data["label"].value_counts().sort_index())

# 7. Check hostname overlap
train_domains = set(train_data["domain"])
test_domains = set(test_data["domain"])

common_domains = train_domains.intersection(test_domains)

print("\n--- Hostname Overlap Check ---")
print("Unique training hostnames:", len(train_domains))
print("Unique testing hostnames:", len(test_domains))
print("Shared hostnames:", len(common_domains))