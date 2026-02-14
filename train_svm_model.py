import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

print("Training SVM Model...\n")

# -----------------------------------
# Load Dataset
# -----------------------------------
try:
    data = pd.read_csv("tremor_data.csv")
except:
    print("tremor_data.csv not found.")
    exit()

if not {"Amplitude", "Frequency", "Label"}.issubset(data.columns):
    print("Dataset format incorrect.")
    exit()

X = data[["Amplitude", "Frequency"]]
y = data["Label"]

# -----------------------------------
# Scale Features
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# Train SVM
# -----------------------------------
model = SVC(kernel="linear")
model.fit(X_scaled, y)

# -----------------------------------
# Save Model and Scaler
# -----------------------------------
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training complete.")
print("Files saved:")
print(" - svm_model.pkl")
print(" - scaler.pkl")
