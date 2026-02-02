import streamlit as st
import pandas as pd
import plotly.express as px

# ────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Safe Walk - Chicago Crimes",
    page_icon="🚶‍♂️🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Title & intro
# ────────────────────────────────────────────────
st.title("🚶‍♂️ Safe Walk – Chicago Crime Explorer")
st.markdown("""
Interactive dashboard to help understand crime patterns in Chicago  
and support safer walking decisions.
""")

# ────────────────────────────────────────────────
# Data loading with caching
# ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Chicago crimes data…")
def load_data():
    # Change this path if your file is in a different location
    possible_paths = [
        "crimes.csv",
        "archive/crimes.csv",
        "/Users/akumaresan/Downloads/archive/crimes.csv",
        "data/crimes.csv"
    ]

    for path in possible_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year.astype("Int64")
            return df
        except FileNotFoundError:
            continue

    st.error("Could not find crimes.csv in any common location.\nPlease place the file in the project folder or update the path in the code.")
    st.stop()

df = load_data()

# ────────────────────────────────────────────────
# Sidebar – filters
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    # Year slider
    if 'Year' in df.columns and not df['Year'].isna().all():
        min_y = int(df['Year'].min())
        max_y = int(df['Year'].max())
        year_range = st.slider(
            "Year range",
            min_y, max_y,
            (max_y - 5, max_y),
            key="year_slider"
        )

    # Primary Type multiselect
    if 'Primary Type' in df.columns:
        types = sorted(df['Primary Type'].dropna().unique())
        selected_types = st.multiselect(
            "Crime types",
            options=types,
            default=types[:12]
        )

# ────────────────────────────────────────────────
# Apply filters
# ────────────────────────────────────────────────
filtered = df.copy()

if 'Year' in filtered.columns:
    filtered = filtered[
        (filtered['Year'] >= year_range[0]) &
        (filtered['Year'] <= year_range[1])
    ]

if 'Primary Type' in filtered.columns and selected_types:
    filtered = filtered[filtered['Primary Type'].isin(selected_types)]

# ────────────────────────────────────────────────
# Key numbers row
# ────────────────────────────────────────────────
st.subheader("Overview")
cols = st.columns(4)

cols[0].metric("Total incidents", f"{len(filtered):,}")
cols[1].metric("Time period", f"{year_range[0]} – {year_range[1]}" if 'Year' in filtered else "–")
if 'Primary Type' in filtered.columns:
    cols[2].metric("Most common type", filtered['Primary Type'].mode()[0] if not filtered.empty else "–")
cols[3].metric("Unique crime types", filtered['Primary Type'].nunique() if 'Primary Type' in filtered else "–")

# ────────────────────────────────────────────────
# Main content – tabs
# ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Map", "📊 Trends", "🏆 Top crimes"])

with tab1:
    st.subheader("Crime locations")
    if all(col in filtered.columns for col in ['Latitude', 'Longitude']):
        # Scatter map
        fig_map = px.scatter_mapbox(
            filtered.sample(min(8000, len(filtered))),
            lat="Latitude",
            lon="Longitude",
            color="Primary Type" if 'Primary Type' in filtered else None,
            hover_name="Primary Type",
            hover_data=['Year', 'District', 'Beat'],
            zoom=9.5,
            height=650,
            opacity=0.55
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No Latitude / Longitude columns found → map not available")

with tab2:
    st.subheader("Trends over time")

    if 'Year' in filtered.columns:
        yearly = filtered.groupby('Year').size().reset_index(name='Count')
        fig_year = px.line(yearly, x='Year', y='Count',
                           title="Crimes per year",
                           markers=True)
        st.plotly_chart(fig_year, use_container_width=True)

    if 'Primary Type' in filtered.columns:
        type_counts = filtered['Primary Type'].value_counts().head(12)
        fig_pie = px.pie(type_counts, values=type_counts.values,
                         names=type_counts.index,
                         title="Distribution of selected crime types")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.subheader("Most frequent crime types")

    if 'Primary Type' in filtered.columns:
        top_n = st.slider("Show top N types", 5, 20, 10)
        top = filtered['Primary Type'].value_counts().head(top_n).reset_index()
        top.columns = ['Type', 'Count']

        fig_bar = px.bar(top, x='Count', y='Type',
                         orientation='h',
                         title=f"Top {top_n} crime types",
                         color='Count',
                         color_continuous_scale='reds')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No 'Primary Type' column found")

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("Safe Walk Project • Data: City of Chicago • Built with ❤️ & Streamlit")
