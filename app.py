import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Flood Prediction & Climate Risk Analysis", layout="centered")

st.title("🌊 Flood Prediction & Climate Risk Analysis")
st.write("Predict Flood Probability based on environmental, climatic, and human-induced factors. Provide your region's conditions below.")

# Default input features (simplified for usability)
default_features = {
    "MonsoonIntensity": 5,
    "Urbanization": 5,
    "AgriculturalPractices": 5,
    "CoastalVulnerability": 5,
    "PopulationScore": 5,
    "TopographyDrainage": 5,
    "ClimateChange": 5,
    "Encroachments": 5,
    "Landslides": 5,
    "WetlandLoss": 5,
    "RiverManagement": 5,
    "DamsQuality": 5,
    "IneffectiveDisasterPreparedness": 5,
    "Watersheds": 5,
    "InadequatePlanning": 5,
    "Deforestation": 5,
    "Siltation": 5,
    "DrainageSystems": 5,
    "DeterioratingInfrastructure": 5,
    "PoliticalFactors": 5
}

# Create input sliders dynamically
st.sidebar.header("⚙️ Input Parameters")
user_input = {}
for feature, default in default_features.items():
    user_input[feature] = st.sidebar.slider(feature, 1, 10, default)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Feature engineering (same as in source code)
input_df["Rainfall_Deforestation"] = input_df["MonsoonIntensity"] * input_df["Deforestation"]
input_df["River_Dams"] = input_df["RiverManagement"] * input_df["DamsQuality"]
input_df["Urban_Planning"] = input_df["Urbanization"] * input_df["InadequatePlanning"]
input_df["Climate_Coastal"] = input_df["ClimateChange"] * input_df["CoastalVulnerability"]
input_df["Climate_Wetlands"] = input_df["ClimateChange"] * input_df["WetlandLoss"]
# Ensure input columns are in the same order as during training
feature_order = [
    "MonsoonIntensity","TopographyDrainage","RiverManagement","Deforestation",
    "Urbanization","ClimateChange","DamsQuality","Siltation","AgriculturalPractices",
    "Encroachments","IneffectiveDisasterPreparedness","DrainageSystems","CoastalVulnerability",
    "Landslides","Watersheds","DeterioratingInfrastructure","PopulationScore","WetlandLoss",
    "InadequatePlanning","PoliticalFactors",
    "Rainfall_Deforestation","River_Dams","Urban_Planning","Climate_Coastal","Climate_Wetlands"
]

# Reorder input_df
input_df = input_df[feature_order]

# Then scale and predict
input_scaled = scaler.transform(input_df)
prediction = xgb_model.predict(input_scaled)[0]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = xgb_model.predict(input_scaled)[0]

# Display result
st.subheader("📊 Prediction Result")
st.metric("🌊 Predicted Flood Probability", f"{prediction:.3f}")

# Risk level
if prediction < 0.4:
    risk = "🟢 Low Flood Risk – Safe"
elif prediction < 0.6:
    risk = "🟠 Moderate Flood Risk – Caution Advised"
else:
    risk = "🔴 High Flood Risk – Take Precautions"

st.info(risk)

st.markdown("""
---
ℹ️ **About This Project**

**Project:** Flood Prediction & Climate Risk Analysis  
**Model:** XGBoost Regressor + StandardScaler  
**Tech:** Python, Streamlit, Scikit-learn  
**Goal:** Help communities & policymakers prepare for flood disasters.  
👨‍💻 Developed as part of an AI & ML project.
""")













