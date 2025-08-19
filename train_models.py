import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("teen_phone_addiction_dataset.csv")

# Binarize addiction level (0–10 scale → 0/1)
df["Addiction"] = (df["Addiction_Level"] >= 7.0).astype(int)

# Select features (you can expand this list later)
X = df[[
    "Screen_Time_Before_Bed",
    "Time_on_Social_Media",
    "Sleep_Hours",
    "Academic_Performance"
]]
y = df["Addiction"]

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train models
models = {
    "logistic": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    model.fit(X_scaled, y_resampled)
    joblib.dump(model, f"{name}_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("✅ All models trained and saved successfully.")