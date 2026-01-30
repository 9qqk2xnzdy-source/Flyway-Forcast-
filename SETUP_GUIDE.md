# Waterfowl Hunting Forecast - Streamlit App

## Project Structure

```
/Your Project Folder/
├── app.py                          # Main Streamlit application
├── Historical_Baseline.csv         # 15-year historical data summaries
├── 24-25 harvest F.F.csv          # Current season harvest data
├── Refuge_Coordinates.csv         # Latitude/longitude for all refuges
├── requirements.txt               # Python dependencies
└── [other CSV files]              # Historical harvest data by year
```

## Installation & Setup

### 1. Install Python Dependencies

Open Terminal and navigate to your project folder:

```bash
cd "/Users/kylesilva/Desktop/F.F CSV Years"
```

Install required packages:

```bash
pip3 install -r requirements.txt
```

Or install individually:

```bash
pip3 install streamlit==1.28.1
pip3 install streamlit-folium==0.17.0
pip3 install folium==0.14.0
pip3 install pandas==2.0.3
pip3 install numpy==1.24.3
```

### 2. Run the Streamlit App

In Terminal:

```bash
streamlit run app.py
```

This will:
- Start a local server (usually at `http://localhost:8501`)
- Automatically open the app in your default browser
- Show live updates as you edit the code

### 3. Using the App

**Left Sidebar:**
- Select a date using the date picker
- Choose a refuge/area from the dropdown

**Main Content:**
- Displays the prediction: "High" or "Low" activity
- Shows key metrics (historical average, current year average, success probability)
- Displays the trend comparing current year to historical average

**Interactive Map:**
- Shows all refuges for the selected week
- Green markers = high success probability (>50%)
- Orange markers = lower success probability
- Blue marker = your selected refuge
- Click markers to see detailed information

## How the Prediction Works

The app:

1. **Calculates Season Week** from your selected date (Season = Oct 1 - Dec 31)
2. **Retrieves Historical Baseline** for that location and week (15-year average)
3. **Gets Current Year Data** from the 2024-25 harvest records
4. **Compares Trend** (current year average vs. historical average)
5. **Determines Activity Level** based on the comparison

### Output Example:
```
"Ash Creek is predicted to have High activity this week, 
primarily consisting of MALL, based on 15 years of harvest data."
```

## Files Reference

### app.py
- Main Streamlit application
- Contains prediction logic
- Generates interactive map and dashboard

### Historical_Baseline.csv
- Columns: Area Name, Season_Week, 15yr_Avg_Ducks, Top_Species, Prob_Successful_Hunt
- Data for all refuges across all 17 weeks of season

### 24-25 harvest F.F.csv
- Current season (2024-2025) harvest data
- Used to compare against historical baseline

### Refuge_Coordinates.csv
- Columns: Area Name, Latitude, Longitude
- Used to place markers on the map

### requirements.txt
- Lists all Python packages needed for the app
- Run `pip install -r requirements.txt` to install everything at once

## Troubleshooting

**"Module not found" error:**
```bash
pip3 install streamlit
```

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Data not loading:**
- Verify all CSV files are in the same folder as `app.py`
- Check file names match exactly (case-sensitive for Linux/Mac)
- Run `python3 baseline.py` and `python3 process_csv.py` to regenerate data if needed

## Stopping the App

In Terminal: Press `Ctrl + C`

## Tips

- The app automatically caches data for faster loading
- Edit `app.py` and save - Streamlit will reload automatically
- You can add more refuges to `Refuge_Coordinates.csv` as needed
- Modify colors, layout, or calculations in `app.py` anytime
