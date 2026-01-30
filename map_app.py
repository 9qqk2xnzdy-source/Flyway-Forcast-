import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
from datetime import datetime

# Load baseline data
baseline = pd.read_csv('/Users/kylesilva/Desktop/F.F CSV Years/Historical_Baseline.csv')

# Approximate coordinates for areas (lat, lon)
coords = {
    'ASH CREEK': (39.5, -120.5),
    'BUTTE VALLEY': (39.3, -121.6),
    'CHINA ISLAND': (38.1, -121.7),
    'COLUSA': (39.2, -122.0),
    'DELEVAN': (37.4, -120.8),
    'DELTA - TWITCHELL': (37.8, -121.5),
    'FINNEY-RAMER': (36.0, -119.2),
    'GADWALL': (39.1, -121.8),
    'GOLD HILLS': (39.1, -120.8),
    'GRAY LODGE': (39.0, -122.0),
    'GRIZZLY ISLAND': (38.1, -121.8),
    'HONEY LAKE - DAKIN': (40.2, -120.2),
    'HONEY LAKE - FLEMING': (40.2, -120.2),
    'HONEY LAKE-DAKIN': (40.2, -120.2),
    'HONEY LAKE-FLEMING': (40.2, -120.2),
    'HOWARD SLOUGH': (39.4, -121.9),
    'ISLAND SLOUGH': (39.4, -121.9),
    'JOICE ISLAND': (39.1, -121.8),
    'KERN': (35.3, -119.0),
    'KESTERSON': (37.3, -120.9),
    'LITTLE DRY CREEK': (39.3, -121.6),
    'LLANO SECO': (39.3, -121.6),
    'LOS BANOS': (37.1, -120.8),
    'MENDOTA': (36.8, -120.4),
    'MERCED': (37.3, -120.5),
    'SACRAMENTO': (38.6, -121.5),
    'SALT SLOUGH': (39.4, -121.9),
    'SALTON SEA NWR': (33.2, -115.8),
    'SAN JACINTO': (33.8, -117.0),
    'SAN LUIS': (37.1, -120.8),
    'SHASTA VALLEY': (40.7, -122.3),
    'SONNY BONO': (33.2, -115.8),
    'SUTTER': (39.2, -121.7),
    'VOLTA': (37.1, -120.8),
    'WEST BEAR CREEK': (37.3, -120.5),
    'WILLOW CREEK': (40.9, -123.6),
    'WISTER': (37.3, -120.5),
    'YOLO': (38.7, -121.8),
    # Add more as needed
}

st.title("California Hunting Areas Interactive Map")

# Date picker
date = st.date_input("Select Date", value=datetime.today())

# Search bar
search = st.text_input("Search Area (leave blank for all)")

# Calculate Season_Week
year = date.year
if date.month >= 10:
    start_year = year
else:
    start_year = year - 1
season_start = datetime(start_year, 10, 1)
week = ((date - season_start).days // 7) + 1

# Filter baseline for the week
week_data = baseline[baseline['Season_Week'] == week]

# Create map
m = folium.Map(location=[37.5, -119.5], zoom_start=6)

for _, row in week_data.iterrows():
    area = row['Area Name']
    if area in coords and (not search or search.lower() in area.lower()):
        lat, lon = coords[area]
        prob = row['Prob_Successful_Hunt']
        color = 'green' if prob > 0.5 else 'red'
        folium.Marker(
            location=[lat, lon],
            popup=f"{area}<br>Prob Success: {prob:.2f}",
            icon=folium.Icon(color=color)
        ).add_to(m)

# Display map
map_data = st_folium(m, width=700, height=500)

# Sidebar for clicked marker
if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    # Find closest area
    for area, (a_lat, a_lon) in coords.items():
        if abs(a_lat - lat) < 0.1 and abs(a_lon - lon) < 0.1:  # tolerance
            st.sidebar.header(f"Summary for {area} (Week {week})")
            row = week_data[week_data['Area Name'] == area]
            if not row.empty:
                top_species = row['Top_Species'].values[0]
                avg_ducks = row['15yr_Avg_Ducks'].values[0]
                prob = row['Prob_Successful_Hunt'].values[0]
                st.sidebar.write(f"Top Species: {top_species}")
                st.sidebar.write(f"15yr Avg Ducks: {avg_ducks:.2f}")
                st.sidebar.write(f"Prob Successful Hunt: {prob:.2f}")
            break