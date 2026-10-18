import pandas as pd

# Load the original dataset
df = pd.read_csv("dataset/phishing_features.csv")

print("Dataset shape:", df.shape)

# Check the number of records in each class
print("\nLabel distribution:")
print(df["label"].value_counts())

# Check how many unique URLs exist
print("\nUnique URLs:", df["url"].nunique())

# Check for duplicate URLs
print("\nDuplicate URLs:", df["url"].duplicated().sum())

# Extract the hostname from each URL
df["domain"] = (
    df["url"]
    .astype(str)
    .str.extract(r"^(?:https?://)?([^/:?#]+)", expand=False)
    .str.lower()
)

# Count unique domains
print("\nUnique hostnames:", df["domain"].nunique())

# Check whether the same hostname has different labels
domain_label_counts = df.groupby("domain")["label"].nunique()

conflicting_domains = domain_label_counts[domain_label_counts > 1]

print("\nHostnames with conflicting labels:", len(conflicting_domains))

print("\nExamples of conflicting hostnames:")
print(conflicting_domains.head(10))
# Show URLs belonging to hostnames with conflicting labels
print("\n--- Conflicting Hostname Details ---")

conflicting_domain_names = conflicting_domains.index

conflicting_rows = df[df["domain"].isin(conflicting_domain_names)]

print(
    conflicting_rows[["url", "domain", "label"]]
    .sort_values(by=["domain", "label"])
    .to_string(index=False)
)
# ----------------------------------
# Check the largest hostnames
# ----------------------------------

print("\n--- Top 10 Most Common Hostnames ---")

print(
    df["domain"]
    .value_counts()
    .head(10)
)


# ----------------------------------
# Check labels for the top hostnames
# ----------------------------------

print("\n--- Labels for the Top 10 Hostnames ---")

top_domains = df["domain"].value_counts().head(10).index

print(
    df[df["domain"].isin(top_domains)]
    .groupby(["domain", "label"])
    .size()
    .unstack(fill_value=0)
)