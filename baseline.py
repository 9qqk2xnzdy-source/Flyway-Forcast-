import pandas as pd

def calculate_historical_baseline(df):
    results = []
    for (area, week), group in df.groupby(['Area Name', 'Season_Week']):
        avg_ducks_15yr = group['Average Ducks'].mean()
        
        # unify species
        species_series = group['#1 Species'].fillna(group.get('#1 Duck Species', pd.Series([pd.NA] * len(group))))
        top_species = species_series.mode().iloc[0] if not species_series.mode().empty else None
        
        # for prob successful hunt
        yearly_avg = group.groupby(group['Date'].dt.year)['Average Ducks'].mean()
        overall_mean = yearly_avg.mean()
        prob_success = (yearly_avg > overall_mean).mean()
        
        results.append({
            'Area Name': area,
            'Season_Week': week,
            '15yr_Avg_Ducks': avg_ducks_15yr,
            'Top_Species': top_species,
            'Prob_Successful_Hunt': prob_success
        })
    return pd.DataFrame(results)

# load the processed csv
df = pd.read_csv('/Users/kylesilva/Desktop/F.F CSV Years/Processed_Combined_Harvest_F.F.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Season_Week'])
df['Season_Week'] = df['Season_Week'].astype(int)
df['Average Ducks'] = pd.to_numeric(df['Average Ducks'], errors='coerce')
df = df.dropna(subset=['Average Ducks', 'Area Name'])

baseline_df = calculate_historical_baseline(df)
baseline_df.to_csv('/Users/kylesilva/Desktop/F.F CSV Years/Historical_Baseline.csv', index=False)