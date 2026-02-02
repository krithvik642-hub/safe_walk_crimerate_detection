import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

# ────────────────────────────────────────────────
# Theme toggle + CSS for black input text
# ────────────────────────────────────────────────
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=1)

st.markdown(f"""
    <style>
    /* Base dark theme */
    [data-testid="stAppViewContainer"] {{
        background-color: {'#0e1117' if theme == 'Dark' else '#ffffff'};
        color: {'#e0e0e0' if theme == 'Dark' else '#000000'};
    }}
    [data-testid="stSidebar"] {{
        background-color: {'#161b22' if theme == 'Dark' else '#f0f2f6'};
    }}

    /* BLACK TEXT in all input fields – very important */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiselect > div > div > div {{
        color: #000000 !important;
        background-color: #ffffff !important;
    }}

    /* Placeholder text readable */
    .stTextInput > div > div > input::placeholder {{
        color: #555555 !important;
    }}

    /* Chart background */
    .stPlotlyChart {{
        background-color: {'#111' if theme == 'Dark' else '#ffffff'} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Safe Walk • Chicago Crimes", layout="wide")

st.title("🚶‍♂️ Safe Walk – Chicago Crimes Explorer & Forecast")
st.caption("Data from: /Users/akumaresan/Downloads/archive/crimes.csv")

# ─── Load & prepare data ───
@st.cache_data
def load_data():
    path = "/Users/akumaresan/Downloads/archive/crimes.csv"
    df = pd.read_csv(path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M')
    df['MonthName'] = df['Date'].dt.month_name()
    df['DayOfWeek'] = df['Date'].dt.day_name()
    return df.dropna(subset=['Date'])

df = load_data()

# ─── Sidebar filters ───
with st.sidebar:
    st.header("Filters")

    min_y, max_y = int(df['Year'].min()), int(df['Year'].max())
    year_range = st.slider("Year range", min_y, max_y, (max_y-10, max_y))

    crime_types = sorted(df['Primary Type'].unique())
    selected_types = st.multiselect("Crime types", crime_types, default=["BATTERY", "THEFT", "ASSAULT", "CRIMINAL DAMAGE"])

# ─── Apply filters ───
filtered = df[
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
]

if selected_types:
    filtered = filtered[filtered['Primary Type'].isin(selected_types)]

# ─── Quick stats ───
st.subheader("Overview")
cols = st.columns(4)
cols[0].metric("Total Incidents", f"{len(filtered):,}")
cols[1].metric("Time period", f"{year_range[0]} – {year_range[1]}")
cols[2].metric("Most common crime", filtered['Primary Type'].mode()[0] if not filtered.empty else "-")
cols[3].metric("Most common day", filtered['DayOfWeek'].mode()[0] if not filtered.empty else "-")

# ─── Tabs ───
tab_trends, tab_map, tab_forecast = st.tabs(["📈 Trends", "🗺️ Map", "🔮 Forecast 2026+"])

# ─── Trends ───
with tab_trends:
    st.subheader("Crimes per Year")

    yearly = filtered.groupby('Year').size().reset_index(name='Count')
    fig_year = px.line(
        yearly, x='Year', y='Count',
        title="Total crimes per year",
        markers=True,
        color_discrete_sequence=['#ff4b4b']
    )
    fig_year.update_traces(marker=dict(size=10, line=dict(width=2, color='white')))
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("Crimes by Day of Week")
    dow = filtered['DayOfWeek'].value_counts().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    ).reset_index(name='Count')
    fig_dow = px.bar(dow, x='DayOfWeek', y='Count', color='Count')
    st.plotly_chart(fig_dow, use_container_width=True)

# ─── Map ───
with tab_map:
    st.subheader("Crime Locations")

    if 'Latitude' in filtered.columns and 'Longitude' in filtered.columns:
        sample_df = filtered.sample(min(12000, len(filtered)))
        fig_map = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="Primary Type",
            hover_name="Primary Type",
            hover_data=["Date", "Block", "Description"],
            zoom=10,
            height=650,
            opacity=0.6
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No Latitude / Longitude columns found in dataset")

# ─── Forecast ───
with tab_forecast:
    st.subheader("Crime Forecast 2026–2030 (Prophet)")

    # Monthly aggregation
    monthly = filtered.groupby('Month').size().reset_index(name='y')
    monthly['ds'] = monthly['Month'].dt.to_timestamp()
    monthly = monthly[['ds', 'y']].sort_values('ds')

    if len(monthly) > 12:
        with st.spinner("Fitting Prophet model (this may take 10–30 seconds)..."):
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            m.fit(monthly)

            future = m.make_future_dataframe(periods=60, freq='M')
            forecast = m.predict(future)

        # Plot forecast
        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=monthly['ds'], y=monthly['y'],
            mode='markers', name='Actual monthly',
            marker=dict(color='#ff4b4b', size=8, line=dict(width=2, color='white'))
        ))

        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='#00cc96', width=3)
        ))

        fig_fc.update_layout(
            title="Monthly Crimes – Historical + Forecast (2026–2030)",
            xaxis_title="Date",
            yaxis_title="Number of Crimes",
            hovermode='x unified'
        )

        st.plotly_chart(fig_fc, use_container_width=True)

        # Summary numbers
        future_2026 = forecast[forecast['ds'].dt.year == 2026]['yhat'].sum()
        future_2027 = forecast[forecast['ds'].dt.year == 2027]['yhat'].sum()
        future_2028 = forecast[forecast['ds'].dt.year == 2028]['yhat'].sum()

        st.markdown(f"**Estimated annual crimes (Prophet forecast):**")
        st.metric("2026", f"{int(round(future_2026)):,}")
        st.metric("2027", f"{int(round(future_2027)):,}")
        st.metric("2028", f"{int(round(future_2028)):,}")

        st.info("Forecast uses Facebook Prophet • Monthly aggregation • No external regressors")
    else:
        st.warning("Not enough monthly data for reliable forecasting (need at least 12+ months)")

st.markdown("---")
st.caption("Made with Streamlit • Data: Chicago Open Data • Forecast: Prophet • Text inputs are black")
