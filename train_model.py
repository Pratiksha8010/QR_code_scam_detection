import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_phishing_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Separate features and labels
X = df.drop(columns=["url", "label"])
y = df["label"]


# Convert categorical TLD values into numeric columns
X = pd.get_dummies(X, columns=["tld"], dtype=int)


print("\nFeatures prepared successfully!")
print("Number of feature columns:", X.shape[1])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n--- Dataset Split ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())