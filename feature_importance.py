import joblib
import pandas as pd

# 1. Load the trained grouped model
model = joblib.load("models/qr_scam_grouped_model.pkl")

print("Grouped model loaded successfully!")

# 2. Get the trained Random Forest classifier
classifier = model.named_steps["classifier"]

# 3. Get the preprocessing component
preprocessor = model.named_steps["preprocessor"]

# 4. Get the feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# 5. Get feature importance scores
importance_scores = classifier.feature_importances_

# 6. Create a DataFrame for the results
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

# 7. Sort features from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# 8. Display the top 15 features
print("\n--- Top 15 Important Features ---")
print(importance_df.head(15).to_string(index=False))

# 9. Save the complete results
importance_df.to_csv(
    "feature_importance_results.csv",
    index=False
)

print("\nComplete results saved to feature_importance_results.csv")