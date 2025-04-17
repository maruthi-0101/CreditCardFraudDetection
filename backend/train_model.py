from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# ✅ Step 1: Load Dataset
df = pd.read_csv("transactions.csv")

# ✅ Step 2: Remove "Time" Column
df = df.drop(columns=["Time"])

# ✅ Step 3: Separate Features & Labels
X = df.drop(columns=["Class"])  # ✅ Keep only 29 features
y = df["Class"]

# ✅ Step 4: Apply SMOTE to Handle Class Imbalance
smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Increase fraud cases to 40%
X_balanced, y_balanced = smote.fit_resample(X, y)

# ✅ Step 5: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# ✅ Step 6: Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Save the scaler
joblib.dump(scaler, "scaler.pkl")

# ✅ Step 7: Define an Improved Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(29,)),  # ✅ Ensure 29 features
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for fraud probability
])


# ✅ Step 8: Use Class Weights to Force Learning
class_weights = {0: 1, 1: 10}  # Fraud cases weighted 10x more

# ✅ Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train the Model
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights)

# ✅ Evaluate Performance
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\n🔍 Final Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
print(f"🔍 Final Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# ✅ Save the Model
model.save("fraud_model_15.keras")
print("✅ Model trained and saved as fraud_model_fixed.keras")
