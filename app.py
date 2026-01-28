import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Waterfowl Hunting Forecast",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦆 California Waterfowl Hunting Forecast")
st.markdown("Interactive tool to predict hunting activity based on 15 years of harvest data")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load all necessary CSV files"""
    historical = pd.read_csv('Historical_Baseline.csv')
    coordinates = pd.read_csv('Refuge_Coordinates.csv')
    current_year = pd.read_csv('24-25 harvest F.F.csv')
    return historical, coordinates, current_year

historical, coordinates, current_year = load_data()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def calculate_season_week(date):
    """Calculate season week from date (season starts Oct 1)"""
    year = date.year
    if date.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    season_start = datetime(start_year, 10, 1)
    week = ((date - season_start).days // 7) + 1
    return week

def predict_activity(date, location, historical_df, current_year_df):
    """
    Predict hunting activity for a given date and location.
    
    Returns:
        dict: Contains prediction data or error message
    """
    season_week = calculate_season_week(date)
    
    # Get historical baseline
    hist_row = historical_df[
        (historical_df['Area Name'].str.upper() == location.upper()) & 
        (historical_df['Season_Week'] == season_week)
    ]
    
    if hist_row.empty:
        return {
            'error': f"No historical data for {location} in Season Week {season_week}"
        }
    
    hist_avg = hist_row['15yr_Avg_Ducks'].iloc[0]
    top_species = hist_row['Top_Species'].iloc[0]
    prob_success = hist_row['Prob_Successful_Hunt'].iloc[0]
    
    # Get current year data
    current_year_df_copy = current_year_df.copy()
    current_year_df_copy['Date'] = pd.to_datetime(
        current_year_df_copy['Date'], 
        format='%m/%d',
        errors='coerce'
    )
    current_year_df_copy['Date'] = current_year_df_copy['Date'].apply(
        lambda d: d.replace(year=2024) if pd.notna(d) and d.month >= 10 else d.replace(year=2025) if pd.notna(d) else pd.NaT
    )
    current_year_df_copy['Season_Week'] = current_year_df_copy['Date'].apply(
        lambda d: calculate_season_week(d) if pd.notna(d) else np.nan
    )
    
    # Filter for location and week
    loc_week_df = current_year_df_copy[
        (current_year_df_copy['Area Name'].str.upper() == location.upper()) &
        (current_year_df_copy['Season_Week'] == season_week)
    ]
    
    current_avg = loc_week_df['Average Ducks'].mean() if not loc_week_df.empty else 0.0
    
    # Determine activity level
    if current_avg > hist_avg:
        activity = 'High'
        trend = f"↑ Up {((current_avg - hist_avg) / hist_avg * 100):.1f}% from average"
    else:
        activity = 'Low'
        trend = f"↓ Down {((hist_avg - current_avg) / hist_avg * 100):.1f}% from average"
    
    return {
        'error': None,
        'activity': activity,
        'species': top_species,
        'historical_avg': hist_avg,
        'current_avg': current_avg,
        'prob_success': prob_success,
        'trend': trend,
        'season_week': season_week
    }

# ============================================================================
# SIDEBAR - INPUTS
# ============================================================================

st.sidebar.header("🎯 Forecast Settings")

date = st.sidebar.date_input(
    "Select Date",
    value=datetime.today(),
    help="Choose the date you plan to hunt"
)

location = st.sidebar.selectbox(
    "Select Refuge/Area",
    sorted(coordinates['Area Name'].unique()),
    help="Choose your hunting location"
)

# ============================================================================
# MAIN CONTENT - PREDICTION
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Activity Forecast")
    
    prediction = predict_activity(date, location, historical, current_year)
    
    if prediction.get('error'):
        st.warning(prediction['error'])
    else:
        activity = prediction['activity']
        species = prediction['species']
        
        # Display main prediction in large text
        if activity == 'High':
            st.success(f"### {location} - {activity} Activity Expected")
        else:
            st.info(f"### {location} - {activity} Activity Expected")
        
        # Summary text
        summary = f"{location} is predicted to have **{activity}** activity this week, primarily consisting of **{species}**, based on 15 years of harvest data."
        st.markdown(summary)
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Historical Avg Ducks",
                f"{prediction['historical_avg']:.2f}",
                help="15-year average for this week/location"
            )
        
        with metric_col2:
            st.metric(
                "Current Year Avg",
                f"{prediction['current_avg']:.2f}",
                delta=f"{prediction['current_avg'] - prediction['historical_avg']:.2f}"
            )
        
        with metric_col3:
            st.metric(
                "Success Probability",
                f"{prediction['prob_success']:.1%}",
                help="Probability of successful hunt based on historical data"
            )
        
        st.markdown(f"**Trend:** {prediction['trend']}")

with col2:
    st.subheader("📍 Details")
    st.markdown(f"""
    **Date:** {date.strftime('%B %d, %Y')}
    
    **Location:** {location}
    
    **Season Week:** {prediction.get('season_week', 'N/A')}
    
    **Top Species:** {prediction.get('species', 'N/A')}
    """)

# ============================================================================
# MAP VIEW
# ============================================================================

st.subheader("🗺️ Interactive Map - All Refuges (Season Week {})".format(
    calculate_season_week(date)
))

# Filter baseline for the week
week_num = calculate_season_week(date)
week_data = historical[historical['Season_Week'] == week_num]

# Get coordinates
coords_dict = dict(zip(coordinates['Area Name'], zip(coordinates['Latitude'], coordinates['Longitude'])))

# Create map centered on California
m = folium.Map(
    location=[37.5, -119.5],
    zoom_start=6,
    tiles="OpenStreetMap"
)

# Add markers for all refuges
for _, row in week_data.iterrows():
    area = row['Area Name']
    if area in coords_dict:
        lat, lon = coords_dict[area]
        prob = row['Prob_Successful_Hunt']
        avg_ducks = row['15yr_Avg_Ducks']
        
        # Color based on success probability
        if prob > 0.5:
            color = 'green'
            icon_text = '✓'
        else:
            color = 'orange'
            icon_text = '!'
        
        # Highlight selected location
        if area.upper() == location.upper():
            color = 'blue'
            weight = 3
        else:
            weight = 2
        
        popup_text = f"""
        <b>{area}</b><br>
        Avg Ducks: {avg_ducks:.2f}<br>
        Success Prob: {prob:.1%}
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=area,
            icon=folium.Icon(color=color, prefix='fa', icon='duck')
        ).add_to(m)

# Display map
st_folium(m, width=700, height=500)

# ============================================================================
# FOOTER - DATA INFO
# ============================================================================

st.divider()
st.caption("Data Source: 15 years of California waterfowl harvest data (2006-2024)")
st.caption("Season runs October 1 - December 31 | Weeks calculated from season start")
