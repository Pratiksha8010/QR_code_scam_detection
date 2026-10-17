import pandas as pd
from pathlib import Path

# Location of the dataset
dataset_path = Path("dataset/phishing_features.csv")

# Check whether the dataset exists
if not dataset_path.exists():
    print("Dataset file not found!")
    print("Check the filename and dataset folder.")
else:
    # Load the CSV file
    df = pd.read_csv(dataset_path)

    # Display the first five records
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Display the number of rows and columns
    print("\n--- Dataset Shape ---")
    print(df.shape)

    # Display all column names
    print("\n--- Column Names ---")
    print(df.columns.tolist())

    # Check missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Check duplicate records
    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    # Check class distribution
    if "label" in df.columns:
        print("\n--- Label Distribution ---")
        print(df["label"].value_counts(dropna=False))
    else:
        print("\nLabel column not found!")