# app.py
# Flood Prediction & Climate Risk Analysis
# Full Streamlit app built only from source code logic

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Flood Prediction & Climate Risk Analysis",
                   layout="wide")

st.title("🌊 Flood Prediction & Climate Risk Analysis")
st.write("This app predicts the probability of floods using trained ML model "
         "and engineered features based on your source code.")

# ------------------ Load Model & Scaler ------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.error("❌ Model or Scaler not found! Place xgb_model.pkl and scaler.pkl in app folder.")
    st.stop()

# ------------------ Feature Definitions ------------------
BASE_FEATURES = [
    "MonsoonIntensity","TopographyDrainage","RiverManagement","Deforestation",
    "Urbanization","ClimateChange","DamsQuality","Siltation","AgriculturalPractices",
    "Encroachments","IneffectiveDisasterPreparedness","DrainageSystems",
    "CoastalVulnerability","Landslides","Watersheds","DeterioratingInfrastructure",
    "PopulationScore","WetlandLoss","InadequatePlanning","PoliticalFactors"
]

ENGINEERED = [
    "Rainfall_Deforestation","River_Dams","Urban_Planning",
    "Climate_Coastal","Climate_Wetlands"
]

FEATURE_ORDER = BASE_FEATURES + ENGINEERED

# ------------------ Input Section ------------------
st.sidebar.header("📥 Input Parameters")
inputs = {}

for feat in BASE_FEATURES:
    inputs[feat] = st.sidebar.slider(feat, 1, 10, 5)

# ------------------ Build Data ------------------
df = pd.DataFrame([inputs])

# Engineered features (from source code)
df["Rainfall_Deforestation"] = df["MonsoonIntensity"] * df["Deforestation"]
df["River_Dams"] = df["RiverManagement"] * df["DamsQuality"]
df["Urban_Planning"] = df["Urbanization"] * df["InadequatePlanning"]
df["Climate_Coastal"] = df["ClimateChange"] * df["CoastalVulnerability"]
df["Climate_Wetlands"] = df["ClimateChange"] * df["WetlandLoss"]

# Reorder columns to match training
df = df[FEATURE_ORDER]

# ------------------ Prediction ------------------
if st.button("🔮 Predict Flood Probability"):
    X_scaled = scaler.transform(df)
    prob = model.predict(X_scaled)[0]

    # Risk category
    if prob < 0.4:
        risk = "🟢 Low"
    elif prob < 0.7:
        risk = "🟠 Moderate"
    else:
        risk = "🔴 High"

    st.subheader("Prediction Result")
    st.metric("Flood Probability", f"{prob:.3f}")
    st.write(f"**Risk Level:** {risk}")

    # ------------------ Visualization ------------------
    st.subheader("Input Summary")
    st.dataframe(df.T.rename(columns={0: "Value"}))

    # Simple input pie chart
    above = (df[BASE_FEATURES] > 5).sum(axis=1).iloc[0]
    below = (df[BASE_FEATURES] < 5).sum(axis=1).iloc[0]
    neutral = len(BASE_FEATURES) - above - below

    fig, ax = plt.subplots()
    ax.pie([above, neutral, below],
           labels=[f"Above 5 ({above})", f"=5 ({neutral})", f"Below 5 ({below})"],
           autopct='%1.0f%%', startangle=90)
    ax.set_title("Input Levels")
    st.pyplot(fig)

    # Export option
    out = df.copy()
    out["PredictedFloodProbability"] = prob
    st.download_button("Download Results (CSV)",
                       data=out.to_csv(index=False),
                       file_name="flood_prediction.csv",
                       mime="text/csv")

# ------------------ Insights ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This app was generated fully using the **source code’s features, "
                 "engineered inputs, and trained model.**")













