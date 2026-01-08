# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import shap
from statsmodels.tsa.arima.model import ARIMA
from fpdf import FPDF

st.set_page_config(page_title="Aadhaar Analytics Dashboard", layout="wide")

st.title("📊 Aadhaar Enrolment & Update Analytics")

# ---------------------------
# 1️⃣ Data Loading
# ---------------------------
st.header("Step 1: Load Data")
try:
    df = pd.read_csv("aadhaar_data.csv")
except FileNotFoundError:
    st.error("⚠️ Data file not found! Make sure 'aadhaar_data.csv' exists in the project folder.")
    st.stop()

if df.empty:
    st.error("⚠️ Data file is empty. Please check the CSV content.")
    st.stop()

st.write("✅ Data loaded successfully!")
st.dataframe(df.head())

# ---------------------------
# 2️⃣ Validate Required Columns
# ---------------------------
required_cols = ['State', 'Year', 'Month', 'Enrolments', 'Updates']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"⚠️ Missing required columns: {missing_cols}")
    st.stop()

# ---------------------------
# 3️⃣ User Filters
# ---------------------------
st.header("Step 2: Filter Data")
states = df['State'].unique().tolist()
selected_state = st.selectbox("Select State", states)
state_df = df[df['State'] == selected_state]

if state_df.empty:
    st.warning(f"⚠️ No data available for {selected_state}. Try a different state.")
    st.stop()

# ---------------------------
# 4️⃣ Feature Engineering
# ---------------------------
state_df = state_df.sort_values(['Year', 'Month'])
state_df['Enrolment_Lag1'] = state_df['Enrolments'].shift(1)
state_df['Update_Ratio'] = state_df['Updates'] / state_df['Enrolments'].replace(0, np.nan)
state_df.fillna(0, inplace=True)

st.subheader("Feature Engineered Data")
st.dataframe(state_df.head())

# ---------------------------
# 5️⃣ Anomaly Detection
# ---------------------------
st.header("Step 3: Detect Anomalies")
if state_df.shape[0] < 2:
    st.warning("⚠️ Not enough data points for anomaly detection.")
else:
    iso = IsolationForest(contamination=0.05, random_state=42)
    state_df['anomaly'] = iso.fit_predict(state_df[['Enrolments', 'Updates']])
    st.write("✅ Anomaly detection complete")
    st.dataframe(state_df[['Year', 'Month', 'Enrolments', 'Updates', 'anomaly']].head())

# ---------------------------
# 6️⃣ Predictive Modeling
# ---------------------------
st.header("Step 4: Predictive Modeling")
X = state_df[['Enrolment_Lag1', 'Update_Ratio']]
y = state_df['Enrolments']

if X.shape[0] < 2:
    st.warning("⚠️ Not enough data for model training.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("✅ RandomForest trained successfully")
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    st.bar_chart(importance_df.set_index('Feature'))

    # SHAP explanations
    if X_test.shape[0] > 0:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        st.subheader("SHAP Summary (Test Data)")
        st.pyplot(shap.summary_plot(shap_values, X_test, show=False))
    else:
        st.warning("⚠️ No test data available for SHAP explanation.")

# ---------------------------
# 7️⃣ Time Series Forecasting
# ---------------------------
st.header("Step 5: Forecast Enrolments")
if len(state_df['Enrolments']) < 3:
    st.warning("⚠️ Not enough data points for ARIMA forecasting.")
else:
    arima_model = ARIMA(state_df['Enrolments'], order=(1,1,1))
    arima_fit = arima_model.fit()
    forecast = arima_fit.forecast(steps=3)
    st.write("✅ ARIMA forecast for next 3 months")
    st.write(forecast)

# ---------------------------
# 8️⃣ Visualizations
# ---------------------------
st.header("Step 6: Visualizations")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(state_df['Enrolments'], label="Enrolments")
ax.plot(state_df['Updates'], label="Updates")
ax.set_title(f"{selected_state} Enrolments & Updates")
ax.set_xlabel("Time")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Optional: Plotly choropleth (if geojson available)
# st.header("National Choropleth Map")
# fig_map = px.choropleth(df, locations="State", color="Enrolments")
# st.plotly_chart(fig_map)

# ---------------------------
# 9️⃣ PDF Report Generation
# ---------------------------
st.header("Step 7: Generate PDF Report")
if st.button("Generate PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Aadhaar Analytics Report - {selected_state}", ln=True)
    pdf.cell(0, 10, "Enrolments & Updates Summary", ln=True)
    pdf.ln(5)
    for idx, row in state_df.iterrows():
        pdf.cell(0, 10, f"{row['Year']}-{row['Month']}: Enrolments={row['Enrolments']}, Updates={row['Updates']}, Anomaly={row['anomaly']}", ln=True)
    pdf.output(f"{selected_state}_aadhaar_report.pdf")
    st.success(f"✅ PDF report generated: {selected_state}_aadhaar_report.pdf")

st.info("🎉 All steps completed successfully!")
