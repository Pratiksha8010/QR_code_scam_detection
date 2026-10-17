import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

# Load the original dataset
input_path = Path("dataset/phishing_features.csv")

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)
print("Missing TLD values before cleaning:", df["tld"].isna().sum())


# Function to extract TLD from a URL
def extract_tld(url):
    if pd.isna(url):
        return None

    url = str(url).strip()

    # Handle URLs that do not contain http:// or https://
    if "://" not in url:
        url = "//" + url

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname

    if not hostname:
        return None

    # Remove a trailing dot and convert to lowercase
    hostname = hostname.rstrip(".").lower()

    # Extract the last domain part
    parts = hostname.split(".")

    if len(parts) >= 2:
        return parts[-1]

    return None


# Fill only missing TLD values
missing_tld = df["tld"].isna()

df.loc[missing_tld, "tld"] = (
    df.loc[missing_tld, "url"].apply(extract_tld)
)

# Save the cleaned dataset
output_path = Path("dataset/cleaned_phishing_features.csv")

df.to_csv(output_path, index=False)

print("\nMissing TLD values after cleaning:", df["tld"].isna().sum())
print("Prepared dataset shape:", df.shape)
print("Cleaned dataset saved to:", output_path)