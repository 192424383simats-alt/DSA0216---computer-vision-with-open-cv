import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print("========== MODULE 3 - SVM CLASSIFICATION ==========\n")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("tremor_data.csv")

X = data[["Amplitude", "Frequency"]]
y = data["Label"]

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Scale whole dataset
X_scaled = scaler.transform(X)

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Compute Mean Values
# -----------------------------
mean_amplitude = X["Amplitude"].mean()
mean_frequency = X["Frequency"].mean()

print("\nMean Amplitude :", round(mean_amplitude,2))
print("Mean Frequency :", round(mean_frequency,2))

# -----------------------------
# Prediction Using Mean
# -----------------------------
mean_scaled = scaler.transform([[mean_amplitude, mean_frequency]])
mean_prediction = model.predict(mean_scaled)[0]

prediction_label = "Parkinson's Risk" if mean_prediction == 1 else "Normal"

print("Prediction Based on Mean:", prediction_label)

# -----------------------------
# Plot Graph
# -----------------------------
plt.figure(figsize=(8,6))

# Plot Normal class
plt.scatter(
    data[data["Label"] == 0]["Amplitude"],
    data[data["Label"] == 0]["Frequency"],
    color='green',
    label="Normal"
)

# Plot Tremor class
plt.scatter(
    data[data["Label"] == 1]["Amplitude"],
    data[data["Label"] == 1]["Frequency"],
    color='red',
    label="Parkinson's Risk"
)

# Highlight Mean Point
plt.scatter(
    mean_amplitude,
    mean_frequency,
    color='blue',
    s=200,
    label="Mean Sample"
)

plt.xlabel("Amplitude (px)")
plt.ylabel("Frequency (Hz)")
plt.title("SVM Classification (Mean-Based Prediction)")
plt.legend()
plt.grid(True)

plt.show()
