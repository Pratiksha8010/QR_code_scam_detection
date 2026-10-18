import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# 1. Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_phishing_features.csv")

print("Dataset loaded:", df.shape)

# 2. Extract hostnames from URLs
df["domain"] = (
    df["url"]
    .astype(str)
    .str.extract(r"^(?:https?://)?([^/:?#]+)", expand=False)
    .str.lower()
)

# Remove records without a hostname
df = df.dropna(subset=["domain"]).copy()

print("Dataset after hostname extraction:", df.shape)

# 3. Separate features, labels, and hostname groups
X = df.drop(columns=["url", "label", "domain"])
y = df["label"]
groups = df["domain"]

print("\nFeatures used:", X.columns.tolist())

# 4. Split the dataset by hostname
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_domains = groups.iloc[train_indices]
test_domains = groups.iloc[test_indices]

# 5. Display dataset split information
print("\n--- Dataset Split ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\n--- Training Label Distribution ---")
print(y_train.value_counts().sort_index())

print("\n--- Testing Label Distribution ---")
print(y_test.value_counts().sort_index())

shared_domains = set(train_domains) & set(test_domains)

print("\n--- Hostname Overlap Check ---")
print("Shared hostnames:", len(shared_domains))

# 6. Preprocess the TLD feature
preprocessor = ColumnTransformer(
    transformers=[
        (
            "tld_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["tld"]
        )
    ],
    remainder="passthrough"
)

# 7. Create the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 8. Create a pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# 9. Train the model
print("\nTraining the grouped Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# 10. Make predictions on the test dataset
y_pred = pipeline.predict(X_test)

# 11. Display classification report
print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Legitimate (0)",
            "Potentially Malicious (1)"
        ],
        zero_division=0
    )
)

# 12. Display confusion matrix
print("\n--- Confusion Matrix ---")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

# 13. Calculate balanced accuracy
balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

print(
    "\nBalanced Accuracy:",
    round(balanced_accuracy, 4)
)

# 14. Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    pipeline,
    "models/qr_scam_grouped_model.pkl"
)

print("\nGrouped model saved successfully!")
print("Location: models/qr_scam_grouped_model.pkl")