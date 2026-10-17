import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# ----------------------------------
# 1. Load the cleaned dataset
# ----------------------------------

df = pd.read_csv("dataset/cleaned_phishing_features.csv")

print("Dataset loaded:", df.shape)


# ----------------------------------
# 2. Prepare features and labels
# ----------------------------------

X = df.drop(columns=["url", "label"])
y = df["label"]

# Convert TLD text values into numeric columns
X = pd.get_dummies(X, columns=["tld"], dtype=int)

# Store feature names for later prediction
feature_names = X.columns.tolist()


# ----------------------------------
# 3. Split the dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ----------------------------------
# 4. Create the Random Forest model
# ----------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ----------------------------------
# 5. Train the model
# ----------------------------------

print("\nTraining the Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ----------------------------------
# 6. Evaluate the model
# ----------------------------------

y_pred = model.predict(X_test)

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

print("\n--- Confusion Matrix ---")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

print(
    "\nBalanced Accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4)
)


# ----------------------------------
# 7. Save the trained model
# ----------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/qr_scam_model.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print("\nModel saved successfully!")
print("Location: models/qr_scam_model.pkl")