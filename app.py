import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Waterfowl Hunting Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NAV CONFIGURATION - loadable/persisted via JSON + editable in sidebar admin
from pathlib import Path
import json

CONFIG_PATH = Path("nav_config.json")

DEFAULT_NAV_CONFIG = {
    "brand": "California Waterfowl Hunting Forecast",
    "nav_bg": "#0b6623",
    "nav_border": "#084f1d",
    "nav_links": [["Forecast", "#forecast"], ["Map", "#map"], ["About", "#about"]],
    "mission_title": "Mission",
    "mission_text": "Provide clear, data-driven forecasts and access to historical harvest data to help hunters plan ethically and safely.",
    "sources_title": "Sources",
    "sources_text": "15 years of California waterfowl harvest data (2006-2024). CSVs available in the project repository."
}
def load_nav_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as fh:
                cfg = json.load(fh)
            # ensure defaults for missing keys
            for k, v in DEFAULT_NAV_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception:
            return DEFAULT_NAV_CONFIG.copy()
    return DEFAULT_NAV_CONFIG.copy()


def save_nav_config(cfg_dict):
    try:
        with open(CONFIG_PATH, "w") as fh:
            json.dump(cfg_dict, fh, indent=2)
        return True
    except Exception:
        return False

# Load config at startup
nav_config = load_nav_config()

NAV_BRAND = nav_config["brand"]
NAV_BG = nav_config["nav_bg"]
NAV_BORDER = nav_config["nav_border"]
NAV_LINKS = [tuple(item) for item in nav_config.get("nav_links", [])]

MISSION_TITLE = nav_config["mission_title"]
MISSION_TEXT = nav_config["mission_text"]
SOURCES_TITLE = nav_config["sources_title"]
SOURCES_TEXT = nav_config["sources_text"]

# ------------------
# Admin UI (sidebar) to edit nav settings and autosave to JSON
# ------------------
with st.sidebar.expander("Navbar settings (admin)", expanded=False):
    st.write("Customize the top navigation bar. Changes autosave to nav_config.json when modified.")

    brand_input = st.text_input("Brand text", value=NAV_BRAND)
    bg_input = st.color_picker("Background color", value=NAV_BG)
    border_input = st.color_picker("Border/accent color", value=NAV_BORDER)

    links_default = "\n".join([f"{label}|{href}" for label, href in NAV_LINKS])
    links_input = st.text_area("Links (one per line, format: Label|#anchor or URL)", value=links_default, height=80)

    mission_title_input = st.text_input("Mission title", value=MISSION_TITLE)
    mission_text_input = st.text_area("Mission text", value=MISSION_TEXT, height=100)

    sources_title_input = st.text_input("Sources title", value=SOURCES_TITLE)
    sources_text_input = st.text_area("Sources text", value=SOURCES_TEXT, height=100)

    # Parse links here for both autosave and manual save
    parsed_links_for_save = []
    for line in (links_input or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if '|' in line:
            label, href = line.split('|', 1)
            parsed_links_for_save.append([label.strip(), href.strip()])
        else:
            label = line.strip()
            href = f"#{label.replace(' ', '-').lower()}"
            parsed_links_for_save.append([label, href])

    # If nothing was provided, fall back to current links
    if not parsed_links_for_save:
        parsed_links_for_save = [[label, href] for label, href in NAV_LINKS]

    cfg_to_write = {
        "brand": brand_input,
        "nav_bg": bg_input,
        "nav_border": border_input,
        "nav_links": parsed_links_for_save,
        "mission_title": mission_title_input,
        "mission_text": mission_text_input,
        "sources_title": sources_title_input,
        "sources_text": sources_text_input
    }

    # Autosave if config changed from last loaded config
    try:
        if cfg_to_write != nav_config:
            ok = save_nav_config(cfg_to_write)
            if ok:
                st.info("Navbar settings autosaved")
                # update in-memory config so autosave doesn't repeat unnecessarily
                nav_config = cfg_to_write
            else:
                st.error("Autosave failed")
    except Exception:
        # Fall back silently in case of unexpected comparison errors
        pass

    # Manual Save / Reset buttons remain for explicit control
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("Save navbar settings"):
            ok = save_nav_config(cfg_to_write)
            if ok:
                st.success("Navbar settings saved to nav_config.json")
                nav_config = cfg_to_write
            else:
                st.error("Failed to save navbar settings")
    with col_reset:
        if st.button("Reset to defaults"):
            try:
                if CONFIG_PATH.exists():
                    CONFIG_PATH.unlink()
                st.experimental_rerun()
            except Exception:
                st.error("Failed to reset configuration")

# Parse links input (for immediate UI update even before save)
parsed_links = []
for line in (links_input or "").splitlines():
    line = line.strip()
    if not line:
        continue
    if '|' in line:
        label, href = line.split('|', 1)
        parsed_links.append((label.strip(), href.strip()))
    else:
        label = line.strip()
        href = f"#{label.replace(' ', '-').lower()}"
        parsed_links.append((label, href))

# Use admin inputs if provided, otherwise fall back to loaded config
NAV_BRAND = brand_input or NAV_BRAND
NAV_BG = bg_input or NAV_BG
NAV_BORDER = border_input or NAV_BORDER
NAV_LINKS = parsed_links if parsed_links else NAV_LINKS
MISSION_TITLE = mission_title_input or MISSION_TITLE
MISSION_TEXT = mission_text_input or MISSION_TEXT
SOURCES_TITLE = sources_title_input or SOURCES_TITLE
SOURCES_TEXT = sources_text_input or SOURCES_TEXT

# Build links HTML from NAV_LINKS
links_html = '\n    '.join([f'<a href="{href}">{label}</a>' for label, href in NAV_LINKS])

# Rebuild nav HTML using the (possibly updated) variables
nav_html = f"""
<style>
.navbar{{position:fixed;top:0;left:0;right:0;background-color:{NAV_BG};color:#fff;z-index:9999;padding:8px 16px;display:flex;align-items:center;justify-content:space-between;font-family:Arial, sans-serif;border-bottom:4px solid {NAV_BORDER}}}
.navbar .brand{{font-weight:700;font-size:18px}}
.navbar .menu{{display:flex;gap:12px;align-items:center}}
.navbar a{{color:#fff;text-decoration:none;padding:6px 8px;border-radius:4px}}
.navbar .dropdown{{position:relative}}
.navbar .dropdown-content{{display:none;position:absolute;top:36px;left:0;background:{NAV_BG};min-width:260px;padding:12px;border-radius:6px;box-shadow:0 4px 8px rgba(0,0,0,0.2)}}
.navbar .dropdown:hover .dropdown-content{{display:block}}
.navbar .dropdown-content p{{margin:0;color:#e6f5ea}}
.main-content{{padding-top:68px}}
.navbar a:hover{{background:rgba(255,255,255,0.06)}}
@media (max-width:600px){{.navbar{{flex-direction:column;align-items:flex-start}}.navbar .menu{{flex-wrap:wrap}}}}
</style>

<div class="navbar">
  <div class="brand">{NAV_BRAND}</div>
  <div class="menu">
    {links_html}
    <div class="dropdown">
      <a href="javascript:void(0)">{MISSION_TITLE} ▾</a>
      <div class="dropdown-content">
        <p><strong>{MISSION_TITLE}</strong></p>
        <p>{MISSION_TEXT}</p>
      </div>
    </div>
    <div class="dropdown">
      <a href="javascript:void(0)">{SOURCES_TITLE} ▾</a>
      <div class="dropdown-content">
        <p><strong>Data Sources</strong></p>
        <p>{SOURCES_TEXT}</p>
      </div>
    </div>
  </div>
</div>
<div class="main-content"></div>
"""

st.markdown(nav_html, unsafe_allow_html=True)

st.title("California Waterfowl Hunting Forecast!")
st.markdown("Interactive tool to predict hunting activity based on 15 years of harvest data")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_all_harvest_data():
    """
    Automatically find and load all harvest CSV files in the directory.
    Combines them into a single master DataFrame with standardized dates.
    """
    import os
    import re
    
    directory = '.'
    all_files = os.listdir(directory)
    
    # Find all harvest CSV files (matching pattern like "06-07 Harvest F.F.csv" or "24-25 harvest F.F.csv")
    harvest_files = []
    for file in all_files:
        if file.endswith('.csv'):
            # Check if it contains harvest data (skip Historical_Baseline, Refuge_Coordinates, etc.)
            if 'harvest' in file.lower() and 'combined' not in file.lower() and 'baseline' not in file.lower():
                harvest_files.append(file)
    
    # Sort files by year range
    def extract_year_range(filename):
        match = re.search(r'(\d{2})-(\d{2})', filename)
        if match:
            return int(match.group(1))
        return 0
    
    harvest_files.sort(key=extract_year_range)
    
    combined_df = pd.DataFrame()
    
    for file in harvest_files:
        try:
            filepath = os.path.join(directory, file)
            
            # Extract year range from filename
            match = re.search(r'(\d{2})-(\d{2})', file)
            if match:
                start_year = 2000 + int(match.group(1))
                end_year = 2000 + int(match.group(2))
            else:
                continue
            
            # Detect header row
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                header = 0 if 'Area Name' in first_line else 1
            
            # Read CSV with empty-file handling
            try:
                df = pd.read_csv(filepath, header=header, on_bad_lines='skip')
            except pd.errors.EmptyDataError:
                st.warning(f"Skipping empty CSV file: {file}")
                continue
            
            # Ensure required columns exist
            if 'Date' not in df.columns or 'Area Name' not in df.columns:
                st.warning(f"Skipping {file} - missing required columns ('Date' or 'Area Name')")
                continue
            
            # Standardize and parse dates
            def parse_date(date_str):
                if pd.isna(date_str) or not isinstance(date_str, str):
                    return pd.NaT
                try:
                    month, day = map(int, date_str.split('/'))
                    # Season runs Oct-Dec, so Oct-Dec = current year, Jan-Sep = next year
                    year = start_year if month >= 10 else end_year
                    return pd.Timestamp(year=year, month=month, day=day)
                except:
                    return pd.NaT
            
            df['Date'] = df['Date'].apply(parse_date)
            
            # Calculate Season_Week
            season_start = pd.Timestamp(year=start_year, month=10, day=1)
            df['Season_Week'] = ((df['Date'] - season_start).dt.days // 7) + 1
            
            # Standardize Area Name
            df['Area Name'] = df['Area Name'].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Standardize species columns
            species_cols = [col for col in df.columns if 'Species' in col]
            for col in species_cols:
                if col in df.columns:
                    df[col] = df[col].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Append to combined
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Could not load {file}: {str(e)}")
            continue
    
    # Drop rows with invalid dates or missing required columns
    combined_df = combined_df.dropna(subset=['Date', 'Area Name'])
    
    return combined_df

def compute_historical_baseline(master_df):
    """
    Compute a historical baseline DataFrame from the aggregated master data if the precomputed baseline is missing.
    """
    results = []
    if master_df is None or master_df.empty:
        return pd.DataFrame(columns=['Area Name','Season_Week','15yr_Avg_Ducks','Top_Species','Prob_Successful_Hunt'])

    for (area, week), group in master_df.groupby(['Area Name', 'Season_Week']):
        # Historical average of Average Ducks
        avg_ducks = group['Average Ducks'].mean() if 'Average Ducks' in group.columns else float('nan')

        # Determine top species (use '#1 Species' or fallback '#1 Duck Species')
        if '#1 Species' in group.columns:
            species_series = group['#1 Species'].fillna(pd.NA)
        elif '#1 Duck Species' in group.columns:
            species_series = group['#1 Duck Species'].fillna(pd.NA)
        else:
            species_series = pd.Series([pd.NA] * len(group))

        top_species = species_series.mode().iloc[0] if not species_series.mode().empty else None

        # Probability of successful hunt: proportion of years with above-average yearly averages
        if 'Date' in group.columns and 'Average Ducks' in group.columns:
            yearly_avg = group.groupby(group['Date'].dt.year)['Average Ducks'].mean()
            overall_mean = yearly_avg.mean() if not yearly_avg.empty else float('nan')
            prob_success = float((yearly_avg > overall_mean).mean()) if not yearly_avg.empty else float('nan')
        else:
            prob_success = float('nan')

        results.append({
            'Area Name': area,
            'Season_Week': int(week),
            '15yr_Avg_Ducks': avg_ducks,
            'Top_Species': top_species,
            'Prob_Successful_Hunt': prob_success
        })

    return pd.DataFrame(results)

@st.cache_data
def load_data():
    """Load all necessary CSV files (with error handling and fallback computation)"""
    # Load coordinates (required)
    try:
        coordinates = pd.read_csv('Refuge_Coordinates.csv')
    except Exception as e:
        st.error(f"Could not load Refuge_Coordinates.csv: {e}")
        raise

    # Load aggregated master harvest data
    master_data = load_all_harvest_data()

    # Try to load precomputed historical baseline; if not available or empty, compute from master_data
    import os
    hb_path = 'Historical_Baseline.csv'
    if os.path.exists(hb_path) and os.path.getsize(hb_path) > 0:
        try:
            historical = pd.read_csv(hb_path)
            # If read but empty or lacks expected columns, fallback
            if historical.empty or 'Area Name' not in historical.columns:
                raise pd.errors.EmptyDataError
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            st.warning("Historical_Baseline.csv appears empty or malformed. Computing baseline from aggregated harvest data.")
            historical = compute_historical_baseline(master_data)
    else:
        st.warning("Historical_Baseline.csv missing or empty. Computing baseline from aggregated harvest data.")
        historical = compute_historical_baseline(master_data)

    return historical, coordinates, master_data

historical, coordinates, master_data = load_data()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def calculate_season_week(date):
    """Calculate season week from date (season starts Oct 1). Accepts date, datetime, or pandas Timestamp."""
    # normalize to pandas Timestamp to avoid mixed-type arithmetic
    date = pd.to_datetime(date)
    year = date.year
    if date.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    season_start = pd.Timestamp(year=start_year, month=10, day=1)
    week = int(((date - season_start).days // 7) + 1)
    return week

def predict_activity(date, location, historical_df, master_data_df):
    """
    Predict hunting activity for a given date and location using master aggregated data.
    
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
    
    # Get current year data from master aggregated data
    # Filter for the selected location and season week
    loc_week_df = master_data_df[
        (master_data_df['Area Name'].str.upper() == location.upper()) &
        (master_data_df['Season_Week'] == season_week)
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

st.sidebar.header("Forecast Settings")

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
    st.markdown('<a id="forecast"></a>', unsafe_allow_html=True)
    st.subheader("Activity Forecast")
    
    prediction = predict_activity(date, location, historical, master_data)
    
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

st.markdown('<a id="map"></a>', unsafe_allow_html=True)
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

st.markdown('<a id="about"></a>', unsafe_allow_html=True)
st.subheader("About")
st.markdown("This application compiles and visualizes 15 years of California waterfowl harvest data to produce weekly forecasts and reference maps. See project documentation for details and data sources.")

# ============================================================================
# FOOTER - DATA INFO
# ============================================================================

st.divider()
st.caption("Data Source: 15 years of California waterfowl harvest data (2006-2024)")
st.caption("Season runs October 1 - December 31 | Weeks calculated from season start")
