import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ─── Page config + dark mode ───
st.set_page_config(
    page_title="Safe Walk • Chicago Crimes 2026 Trend",
    page_icon="🚶‍♂️📉",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    .stPlotlyChart {
        background-color: #111 !important;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚶‍♂️ Safe Walk – Chicago Crime Trends & 2026 Projection")
st.markdown("Interactive dashboard with dark mode, outlined markers, hover cursor & simple 2026 forecast")

# ─── Data loading ───
@st.cache_data
def load_crimes():
    # Update this path if your file is elsewhere
    file_path = "crimes.csv"  # or "archive/crimes.csv" or full path
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Year'] = df['Date'].dt.year.astype("Int64")
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}\nPlease place crimes.csv in the project folder or correct the path.")
        st.stop()

df = load_crimes()

# ─── Sidebar filters ───
with st.sidebar:
    st.header("Controls")

    # Year range
    if 'Year' in df.columns and not df['Year'].isna().all():
        min_y = int(df['Year'].min())
        max_y = int(df['Year'].max())
        year_range = st.slider(
            "Year Range",
            min_y, max_y,
            (max_y - 7, max_y),
            key="year_filter"
        )

    # Primary Type
    if 'Primary Type' in df.columns:
        types = sorted(df['Primary Type'].dropna().unique())
        selected_types = st.multiselect(
            "Crime Types",
            types,
            default=types[:10]
        )

# ─── Apply filters ───
filtered = df.copy()
if 'Year' in filtered.columns:
    filtered = filtered[(filtered['Year'] >= year_range[0]) & (filtered['Year'] <= year_range[1])]

if 'Primary Type' in filtered.columns and selected_types:
    filtered = filtered[filtered['Primary Type'].isin(selected_types)]

# ─── Prepare yearly summary for trend ───
if 'Year' in filtered.columns:
    yearly = filtered.groupby('Year').size().reset_index(name='Total Crimes')
    yearly = yearly.sort_values('Year')
else:
    yearly = pd.DataFrame()  # fallback if no Year

# ─── 2026 simple prediction (linear on recent 5 years) ───
pred_2026 = None
if len(yearly) >= 5:
    recent = yearly.tail(5)
    x = recent['Year'].values
    y = recent['Total Crimes'].values
    coeffs = np.polyfit(x, y, 1)
    pred_2026 = int(round(coeffs[0] * 2026 + coeffs[1]))
    if pred_2026 < 0:
        pred_2026 = 0

# ─── Interactive Plotly chart ───
st.subheader("Crime Trend & 2026 Forecast")

fig = go.Figure()

# Main trend line + hollow markers
fig.add_trace(
    go.Scatter(
        x=yearly['Year'],
        y=yearly['Total Crimes'],
        mode='lines+markers',
        name='Total Crimes',
        line=dict(color='#ff5555', width=3),
        marker=dict(
            size=12,
            color='#ff5555',
            line=dict(width=2.5, color='white'),  # hollow effect
            symbol='circle'
        ),
        hovertemplate='Year: %{x}<br>Crimes: <b>%{y:,}</b><extra></extra>'
    )
)

# Prediction point + line if enabled
if pred_2026 is not None:
    fig.add_trace(
        go.Scatter(
            x=[yearly['Year'].iloc[-1], 2026],
            y=[yearly['Total Crimes'].iloc[-1], pred_2026],
            mode='lines+markers',
            name='2026 Projection',
            line=dict(color='#ffaa00', width=3, dash='dot'),
            marker=dict(
                size=16,
                color='#ffaa00',
                symbol='diamond',
                line=dict(width=3, color='white')
            ),
            hovertemplate=f'Year: 2026<br>Projected Crimes: <b>{pred_2026:,}</b> (linear trend)<extra></extra>'
        )
    )

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#e0e0e0',
    xaxis=dict(gridcolor='rgba(120,120,120,0.3)', title="Year"),
    yaxis=dict(gridcolor='rgba(120,120,120,0.3)', title="Number of Crimes"),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(30,30,40,0.95)', font_color='#ffffff'),
    showlegend=True,
    legend=dict(bgcolor='rgba(30,30,40,0.8)', bordercolor='#444'),
    height=580,
    margin=dict(l=50, r=50, t=40, b=60)
)

st.plotly_chart(fig, use_container_width=True)

# ─── Summary cards ───
st.markdown("---")
cols = st.columns(3)
cols[0].metric("Total Crimes (filtered)", f"{len(filtered):,}")
cols[1].metric("Years Shown", f"{year_range[0]} – {year_range[1]}")
if pred_2026 is not None:
    cols[2].metric("Projected 2026 Crimes", f"{pred_2026:,}", delta="linear trend", delta_color="inverse")

st.caption("Data: crimes.csv • Projection: simple linear fit on recent years • Hover over points for details")

# Optional: raw data preview
with st.expander("View Filtered Data Sample"):
    st.dataframe(filtered.head(1500))

