import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AEP Energy Forecaster", layout="wide")
st.title("⚡ AEP Hourly Load Forecaster")

model=joblib.load('/home/sriaparna/AI/energy_forecast/energy_xgb_model.pkl')
last_point = pd.read_csv('last_known_point.csv')

def predict_future(model, start_point, steps):

    predictions = []
    curr_lag_1h = start_point['AEP_MW'].iloc[0]
    curr_lag_24h = start_point['lag_24h'].iloc[0]
    curr_rolling = start_point['rolling_mean_24h'].iloc[0]
    curr_time = datetime.now() 

    for i in range(steps):
        features = pd.DataFrame([[
            curr_time.hour,
            curr_time.weekday(),
            (curr_time.month-1)//3 + 1,
            curr_time.month,
            curr_time.year,
            curr_lag_1h,
            curr_lag_24h,
            curr_rolling
        ]], columns=['hour', 'dayofweek', 'quarter', 'month', 'year', 'lag_1h', 'lag_24h', 'rolling_mean_24h'])
        pred = model.predict(features)[0]
        predictions.append(pred)
        curr_lag_1h = pred  
        curr_time += timedelta(hours=1)
        
    return predictions
st.sidebar.header("Forecast Settings")
hours_to_forecast = st.sidebar.slider("Hours to Predict", 1, 48, 24)
if st.sidebar.button('Generate Future Load Forecast'):
    with st.spinner('Calculating recursive load projections...'):
 
        forecast_results = predict_future(model, last_point, hours_to_forecast)
  
        chart_times = [datetime.now() + timedelta(hours=i) for i in range(hours_to_forecast)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_times, 
            y=forecast_results, 
            mode='lines+markers', 
            name='Predicted MW',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.update_layout(
            title=f"Projected Energy Demand for next {hours_to_forecast} Hours",
            xaxis_title="Time",
            yaxis_title="Consumption (MW)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_view=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            
            peak_val = max(forecast_results) / 1000 
            st.metric("Peak Demand", f"{peak_val:.1f}k MW")

        with col2:
            low_val = min(forecast_results) / 1000
            st.metric("Lowest Demand", f"{low_val:.1f}k MW")

        with col3:
            avg_val = np.mean(forecast_results) / 1000
            st.metric("Average Load", f"{avg_val:.1f}k MW")

        st.info(f"**Business Action:** Based on the peak of {max(forecast_results):,.0f} MW, "
                "grid operators should ensure spinning reserves are at 15% capacity.")
else:
    st.info("Adjust the horizon in the sidebar and click 'Generate' to see the future grid demand.")

