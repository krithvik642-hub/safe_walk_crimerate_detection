import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import numpy as np

# ─── Theme toggle & CSS ───
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=1)

if theme == "Dark":
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #e0e0e0; }
        [data-testid="stSidebar"] { background-color: #161b22; }
        .stPlotlyChart { background-color: #111 !important; }
        h1, h2, h3, p, div { color: #ffffff !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #000000; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Safe Walk • Chicago Crimes", page_icon="🚶‍♂️📊", layout="wide")

st.title("🚶‍♂️ Safe Walk – Chicago Crimes Explorer & 2026+ Forecast")

# ─── Load data ───
@st.cache_data
def load_data():
    path = "/Users/akumaresan/Downloads/archive/crimes.csv"
    df = pd.read_csv(path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    return df

df = load_data()

# ─── Sidebar filters ───
with st.sidebar:
    st.header("Filters")
    year_range = st.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()), (df['Year'].max()-10, df['Year'].max()))
    crime_types = sorted(df['Primary Type'].unique())
    selected_types = st.multiselect("Crime Types", crime_types, default=crime_types[:10])

filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
if selected_types:
    filtered = filtered[filtered['Primary Type'].isin(selected_types)]

# ─── Metrics ───
st.subheader("Overview")
cols = st.columns(4)
cols[0].metric("Total Crimes", f"{len(filtered):,}")
cols[1].metric("Years", f"{year_range[0]} – {year_range[1]}")
cols[2].metric("Top Crime Type", filtered['Primary Type'].mode()[0] if not filtered.empty else "N/A")
cols[3].metric("Most Common Day", filtered['DayOfWeek'].mode()[0] if not filtered.empty else "N/A")

# ─── Tabs ───
tab_trends, tab_map, tab_forecast = st.tabs(["Trends", "Map", "ML Forecast & Reduction"])

with tab_trends:
    st.subheader("Trends Over Years")

    # Yearly line chart with hollow markers
    yearly = filtered.groupby('Year').size().reset_index(name='Count')
    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(
        x=yearly['Year'], y=yearly['Count'],
        mode='lines+markers',
        name='Crimes per Year',
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=12, color='#ff4b4b', line=dict(width=3, color='white')),  # Hollow
        hovertemplate='Year: %{x}<br>Crimes: <b>%{y:,}</b><extra></extra>'
    ))
    fig_year.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0' if theme == "Dark" else '#000000',
                           hovermode='x unified')
  st.plotly_chart(fig_year, use_container_width=True)

    # Day of week heatmap
    dow_hour = filtered.groupby(['DayOfWeek', filtered['Date'].dt.hour]).size().unstack(fill_value=0)
    fig_heat = go.Figure(data=go.Heatmap(
        z=dow_hour.values,
        x=dow_hour.columns,
        y=dow_hour.index,
        colorscale='Blues',
        showscale=True
    ))
    fig_heat.update_layout(title="Crimes by Day & Hour", xaxis_title="Hour", yaxis_title="Day", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
with tab_trends:
    st.subheader("Trends Over Years")

    # Yearly trend
    yearly = filtered.groupby('Year').size().reset_index(name='Count')

    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(
        x=yearly['Year'],
        y=yearly['Count'],
        mode='lines+markers',
        name='Crimes per Year',
        line=dict(width=3),
        marker=dict(size=10),
        hovertemplate='Year: %{x}<br>Crimes: %{y}<extra></extra>'
    ))

    fig_year.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        hovermode="x unified"
    )

    st.plotly_chart(fig_year, use_container_width=True)

    # Heatmap: Day vs Hour
    filtered['Hour'] = filtered['Date'].dt.hour
    dow_hour = filtered.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=dow_hour.values,
        x=dow_hour.columns,
        y=dow_hour.index,
        colorscale='Blues'
    ))

    fig_heat.update_layout(
        title="Crimes by Day and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week"
    )

    st.plotly_chart(fig_heat, use_container_width=True)

with tab_map:
    st.subheader("Crime Locations")
    if 'Latitude' in filtered.columns and 'Longitude' in filtered.columns:
       fig_map = px.scatter_map( filtered.sample(min(10000, len(filtered))), lat="Latitude", lon="Longitude", color="Primary Type", hover_name="Primary Type",zoom=10, height=600, opacity=0.6 )

       fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":0,"l":0,"b":0}
      )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No Lat/Lon columns – map skipped")

with tab_forecast:
    st.subheader("ML Forecast (Prophet) & Crime Reduction Scenarios")

    # Monthly data for Prophet
    monthly = filtered.groupby('Month').size().reset_index(name='y')
    monthly['ds'] = monthly['Month'].dt.to_timestamp()
    monthly = monthly[['ds', 'y']]

    if len(monthly) > 12:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(monthly)
        future = m.make_future_dataframe(periods=60, freq='M')  # 5 years
        forecast = m.predict(future)

        # Baseline forecast chart
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#00cc96')))
        fig_fc.add_trace(go.Scatter(x=monthly['ds'], y=monthly['y'], mode='markers', name='Historical', marker=dict(color='#ff4b4b', size=8, line=dict(width=2, color='white'))))
        fig_fc.update_layout(title="Monthly Crimes Forecast (2026+)", xaxis_title="Date", yaxis_title="Crimes")
        st.plotly_chart(fig_fc, use_container_width=True)

        # Reduction scenarios
        st.markdown("**Crime Reduction Scenarios** (e.g., interventions)")
        reduction_pct = st.slider("Assumed reduction % (due to policing/community programs)", 0, 50, 10)
        forecast_reduced = forecast.copy()
        forecast_reduced['yhat_reduced'] = forecast['yhat'] * (1 - reduction_pct / 100)

        st.metric("Projected 2026 Crimes (baseline)", f"{forecast[forecast['ds'].dt.year == 2026]['yhat'].sum():,.0f}")
        st.metric(f"With {reduction_pct}% reduction", f"{forecast_reduced[forecast_reduced['ds'].dt.year == 2026]['yhat_reduced'].sum():,.0f}", delta=f"-{(forecast['yhat'].sum() * reduction_pct / 100):.0f}")

        st.info("This uses Facebook Prophet for forecasting. Reduction is hypothetical.")
    else:
        st.warning("Not enough data for reliable forecast (need >1 year monthly)")

st.caption("Data: Chicago Crimes • ML: Prophet • Hover over charts for details • Theme toggle in sidebar")
