import pandas as pd
import joblib

# Load the grouped model
model = joblib.load("models/qr_scam_grouped_model.pkl")

print("Grouped model loaded successfully!")

# Load sample URLs and their features from the dataset
df = pd.read_csv("dataset/cleaned_phishing_features.csv")

# Select one sample for each label, if available
samples = []

for label in [0, 1]:
    sample = df[df["label"] == label].head(1)

    if not sample.empty:
        samples.append(sample)

# Test the model on the selected samples
for sample in samples:
    url = sample["url"].iloc[0]

    # Prepare the features in the same format as training
    X = sample.drop(columns=["url", "label"])

    prediction = model.predict(X)[0]

    print("\nURL:", url)
    print("Actual label:", sample["label"].iloc[0])
    print("Predicted label:", prediction)

    if prediction == 0:
        print("Prediction: Legitimate according to the model")
    else:
        print("Prediction: Potentially malicious according to the model")