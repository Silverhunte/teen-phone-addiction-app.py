import streamlit as st
import numpy as np
import joblib as jb

# Load models
scaler = joblib.load("scaler.pkl")
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl")
}

# UI
st.title("📱 Teen Phone Addiction Predictor")
st.markdown("Enter behavioral and academic inputs to predict addiction risk.")

screen_time = st.slider("Screen Time Before Bed (hours)", 0.0, 5.0, 0.5)
social_media = st.slider("Time on Social Media (hours)", 0.0, 5.0, 1.0)
sleep = st.slider("Sleep Duration (hours)", 0.0, 12.0, 8.0)
academic = st.slider("Academic Performance (%)", 0, 100, 75)

input_data = np.array([[screen_time, social_media, sleep, academic]])
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Addiction Level"):
    votes = []
    for name, model in models.items():
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        votes.append(pred)
        color = "🟩" if pred == 0 else "🟥"
        st.write(f"**{name}**: {color} {'Not Addicted' if pred == 0 else 'Addicted'} (Confidence: {prob:.2f})")

    # Ensemble vote
    final_vote = round(np.mean(votes))

    # Behavioral override
    override_addicted = screen_time >= 4.5 or social_media >= 4.5
    if override_addicted:
        final_vote = 1
        st.markdown("⚠️ **Behavioral Override Triggered**: High screen/social media time")

    # Final verdict
    st.markdown("---")

    st.subheader(f"🧠 Ensemble Verdict: {'🟥 Addicted' if final_vote == 1 else '🟩 Not Addicted'}")
