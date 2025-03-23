import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# ✅ Load dataset
df = pd.read_csv("creditcard.csv")

# ✅ Drop unnecessary columns if any extra exist
df = df.drop(columns=["Time"], errors="ignore")  # Ignore if "Time" doesn't exist

# ✅ Load trained model & scaler
try:
    model = tf.keras.models.load_model("fraud_model_14.keras")
    scaler = joblib.load("scaler.pkl")  # Load the same scaler used in training
    print("✅ Model and Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)

# ✅ Get example fraud & safe transactions
fraud_cases = df[df["Class"] == 1].drop(columns=["Class"], errors="ignore").values[:5]
safe_cases = df[df["Class"] == 0].drop(columns=["Class"], errors="ignore").values[:5]

# ✅ Ensure test data matches training feature count (29 features)
expected_feature_count = scaler.n_features_in_  # Get correct feature count

fraud_cases = np.array([row[:expected_feature_count] for row in fraud_cases])
safe_cases = np.array([row[:expected_feature_count] for row in safe_cases])

# ✅ Normalize test data
fraud_cases_scaled = scaler.transform(fraud_cases)
safe_cases_scaled = scaler.transform(safe_cases)

# ✅ Get predictions
fraud_predictions = model.predict(fraud_cases_scaled)
safe_predictions = model.predict(safe_cases_scaled)

# ✅ Convert predictions to percentage
fraud_probabilities = fraud_predictions.flatten() * 100
safe_probabilities = safe_predictions.flatten() * 100

# ✅ Display results
print("\n✅ **Safe Transaction Predictions**")
for i, prob in enumerate(safe_probabilities):
    print(f"   - Safe Transaction {i+1}: Fraud Probability = {prob:.2f}%")

print("\n⚠️ **Fraudulent Transaction Predictions**")
for i, prob in enumerate(fraud_probabilities):
    print(f"   - Fraud Transaction {i+1}: Fraud Probability = {prob:.2f}%")

# ✅ Final Summary
print("\n🚀 **Model Testing Completed!**")