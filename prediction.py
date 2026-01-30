import pandas as pd
import datetime

def predict_activity(date_str, location):
    """
    Predicts hunting activity for a given date and location based on historical data.

    Args:
        date_str (str): Date in 'YYYY-MM-DD' format.
        location (str): The area name, e.g., 'Ash Creek'.

    Returns:
        str: Prediction summary.
    """
    # Parse the date
    try:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

    # Calculate Season_Week assuming season starts October 1st
    season_start = datetime.datetime(date.year, 10, 1)
    days_diff = (date - season_start).days
    season_week = days_diff // 7 + 1

    # Load historical baseline
    try:
        hist_df = pd.read_csv('Historical_Baseline.csv')
    except FileNotFoundError:
        return "Historical_Baseline.csv not found."

    # Filter for location and season_week
    hist_row = hist_df[(hist_df['Area Name'].str.upper() == location.upper()) & (hist_df['Season_Week'] == season_week)]
    if hist_row.empty:
        return f"No historical data available for {location} in Season Week {season_week}."

    hist_avg = hist_row['15yr_Avg_Ducks'].iloc[0]
    top_species = hist_row['Top_Species'].iloc[0]

    # Load current year data (24-25)
    try:
        current_df = pd.read_csv('24-25 harvest F.F.csv')
    except FileNotFoundError:
        return "24-25 harvest F.F.csv not found."

    # Parse dates, assuming MM/DD format, and assign year
    current_df['Date'] = pd.to_datetime(current_df['Date'], format='%m/%d')
    # Dates are from Oct to Dec, so all in the same year, but to be general
    current_df['Date'] = current_df['Date'].apply(lambda d: d.replace(year=2024) if d.month >= 10 else d.replace(year=2025))

    # Calculate Season_Week for current data
    current_df['Season_Week'] = current_df['Date'].apply(lambda d: ((d - datetime.datetime(d.year, 10, 1)).days // 7) + 1)

    # Filter for location
    loc_df = current_df[current_df['Area Name'].str.upper() == location.upper()]

    # Get current average for the season_week
    if loc_df.empty:
        current_avg = 0.0
    else:
        week_df = loc_df[loc_df['Season_Week'] == season_week]
        if week_df.empty:
            current_avg = 0.0
        else:
            current_avg = week_df['Average Ducks'].mean()

    # Determine activity level
    if current_avg > hist_avg:
        activity = 'High'
    else:
        activity = 'Low'

    # Generate summary
    summary = f"{location} is predicted to have {activity} activity this week, primarily consisting of {top_species}, based on 15 years of harvest data."
    return summary

# Example usage
if __name__ == "__main__":
    # Test with a sample date and location
    print(predict_activity('2024-10-23', 'Ash Creek'))