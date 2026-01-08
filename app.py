# app.py
# Hackathon-Ready Aadhaar Analytics (Full Pipeline + Policy + Visuals + Reports)
# --------------------------------------------------
# Runs in TWO modes:
# 1) Streamlit: streamlit run app.py
# 2) CLI: python app.py
# --------------------------------------------------

# ---------------- Safe Imports ----------------
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Optional LSTM
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# PDF report
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ---------------- Utility ----------------
def ui_write(msg):
    if STREAMLIT_AVAILABLE:
        st.write(msg)
    else:
        print(msg)

# ---------------- Data Loading ----------------
def load_data():
    return pd.read_csv("aadhaar_sample_data.csv")

try:
    df = load_data()
except Exception:
    # Auto-generate demo data
    dates = pd.date_range(start="2023-01-31", periods=24, freq="ME")
    df = pd.DataFrame({
        "Month": dates,
        "State": ["UP", "MH", "DL", "RJ"] * 6,
        "Total_Enrolments": np.random.randint(70000, 170000, size=len(dates)),
        "Total_Updates": np.random.randint(120000, 450000, size=len(dates)),
    })

# ---------------- Preprocessing & Feature Engineering ----------------
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Ratios & temporal features
df['Update_Enrolment_Ratio'] = df['Total_Updates'] / (df['Total_Enrolments'] + 1)
df['Month_Num'] = df['Month'].dt.month
df['Quarter'] = df['Month'].dt.quarter
df['Year'] = df['Month'].dt.year

# Lag & rolling features
df['Enroll_Lag_1'] = df['Total_Enrolments'].shift(1)
df['Enroll_Lag_3'] = df['Total_Enrolments'].shift(3)
df['Enroll_Roll_Mean_3'] = df['Total_Enrolments'].rolling(3).mean()
df['Enroll_Roll_Std_3'] = df['Total_Enrolments'].rolling(3).std()

# State encoding
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Drop NaNs
df = df.dropna().reset_index(drop=True)

# ---------------- UI Preview ----------------
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Hackathon Aadhaar Analytics", layout="wide")
    st.title("🚀 Hackathon Aadhaar Analytics Dashboard")
    st.dataframe(df.head())
else:
    print(df.head())

# ---------------- State Visualization ----------------
state_cols = [c for c in df.columns if c.startswith('State_')]
state_load = df[state_cols].sum()
fig, ax = plt.subplots()
ax.barh(state_load.index, state_load.values)
ax.set_title("State-wise Enrolment Load")
if STREAMLIT_AVAILABLE:
    st.pyplot(fig)
else:
    plt.show()

# ---------------- Anomaly Detection ----------------
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Total_Enrolments', 'Total_Updates', 'Update_Enrolment_Ratio']])
anomalies = df[df['Anomaly']==-1]
ui_write("🚨 Detected Anomalies")
ui_write(anomalies[['Month','Total_Enrolments','Total_Updates','Update_Enrolment_Ratio']])

# Policy Interpretation
def interpret_anomaly(row):
    if row['Update_Enrolment_Ratio']>3: return "Mass updates / scheme linkage"
    if row['Total_Enrolments']>df['Total_Enrolments'].quantile(0.95): return "Enrolment drive / infra scaling"
    if row['Total_Enrolments']<df['Total_Enrolments'].quantile(0.05): return "Access gap / disruption"
    return "Operational irregularity"

if not anomalies.empty:
    anomalies['Policy_Interpretation'] = anomalies.apply(interpret_anomaly, axis=1)
    ui_write("📜 Policy Interpretation")
    ui_write(anomalies[['Month','Policy_Interpretation']])

# ---------------- ARIMA Forecast ----------------
series = df.groupby('Month')['Total_Enrolments'].sum().asfreq('ME')
arima_fit = ARIMA(series, order=(1,1,1)).fit()
forecast = arima_fit.forecast(6)

# ---------------- ML Regression ----------------
target = 'Total_Enrolments'
features = [c for c in df.columns if c not in ['Month',target,'Anomaly']]
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
rf = RandomForestRegressor(n_estimators=300,max_depth=12,random_state=42)
rf.fit(X_train,y_train)
preds = rf.predict(X_test)
r2 = r2_score(y_test,preds)
ui_write({'MAE':round(mean_absolute_error(y_test,preds),2),'RMSE':round(np.sqrt(mean_squared_error(y_test,preds)),2),'R2':round(r2,3)})

# ---------------- SHAP Global & Per-State ----------------
if SHAP_AVAILABLE:
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'Feature':X_train.columns,'Mean_SHAP_Importance':shap_importance}).sort_values('Mean_SHAP_Importance',ascending=False)
    ui_write("🔍 SHAP Feature Importance")
    ui_write(shap_df.head(10))
    if STREAMLIT_AVAILABLE:
        fig_shap,ax=plt.subplots()
        ax.barh(shap_df['Feature'][:10],shap_df['Mean_SHAP_Importance'][:10])
        ax.set_title("Top SHAP Features")
        st.pyplot(fig_shap)
    # Per-state impact
    state_features = [c for c in X_train.columns if c.startswith('State_')]
    if state_features:
        ui_write("📍 SHAP Per-State Impact")
        for sf in state_features:
            impact = np.abs(shap_values[:,X_train.columns.get_loc(sf)]).mean()
            ui_write(f"{sf}: {round(impact,2)}")
else:
    ui_write("SHAP skipped")

# ---------------- District Drilldown ----------------
if 'District' in df.columns:
    ui_write("🏘️ District-level Analytics")
    district_summary = df.groupby('District')[target].agg(['mean','sum','std']).reset_index()
    ui_write(district_summary.head())
else:
    ui_write("No District column, using State-level")

# ---------------- Plotly State Heatmap ----------------
if PLOTLY_AVAILABLE and 'State' in df.columns:
    heat_df = df.groupby('State')[target].sum().reset_index()
    fig_map = px.choropleth(heat_df, locations='State', color=target, title='India State-wise Aadhaar Enrolment Heatmap')
    if STREAMLIT_AVAILABLE:
        st.plotly_chart(fig_map,use_container_width=True)

# ---------------- PDF Report ----------------
if PDF_AVAILABLE:
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=10)
    pdf.cell(0,10,"Hackathon Aadhaar Analytics Report",ln=True)
    pdf.cell(0,8,f"R2 Score: {round(r2,3)}",ln=True)
    pdf.cell(0,8,f"Anomalies Detected: {len(anomalies)}",ln=True)
    pdf.output("aadhaar_report.pdf")
    ui_write("📄 PDF report generated")

# ---------------- UIDAI Dataset Guide ----------------
ui_write("📡 Real UIDAI Dataset Integration")
ui_write("Steps: Download UIDAI CSV, rename columns to Month, State, District(optional), Total_Enrolments, Total_Updates, place as aadhaar_sample_data.csv, re-run app")

# ---------------- Sanity Checks ----------------
assert r2>0
assert 'Total_Enrolments' in df.columns
ui_write("Pipeline executed successfully ✅")
