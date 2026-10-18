import pandas as pd

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

# Step 4: Display the overall label distribution
print("\n--- Overall Label Distribution ---")
print(df["label"].value_counts().sort_index())

# Step 5: Count each label for every hostname
domain_label_counts = pd.crosstab(
    df["domain"],
    df["label"]
)

# Step 6: Count hostnames containing each label
print("\n--- Hostname Label Analysis ---")

for label in sorted(df["label"].unique()):
    count = (domain_label_counts[label] > 0).sum()

    print(f"Hostnames containing label {label}: {count}")

# Step 7: Find hostnames containing both labels
if 0 in domain_label_counts.columns and 1 in domain_label_counts.columns:
    mixed_domains = domain_label_counts[
        (domain_label_counts[0] > 0)
        & (domain_label_counts[1] > 0)
    ]

    print(
        "\nHostnames containing both labels:",
        len(mixed_domains)
    )

# Step 8: Analyze hostname frequency
domain_sizes = df.groupby("domain").size()

print("\n--- Hostname Statistics ---")
print("Total unique hostnames:", df["domain"].nunique())

print(
    "Hostnames with only one URL:",
    (domain_sizes == 1).sum()
)

print(
    "Hostnames with multiple URLs:",
    (domain_sizes > 1).sum()
)