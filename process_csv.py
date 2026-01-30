import pandas as pd
import os
import re

directory = '/Users/kylesilva/Desktop/F.F CSV Years'
files = [f for f in os.listdir(directory) if f.endswith('.csv') and 'Combined' not in f]
combined_df = pd.DataFrame()

for file in files:
    filepath = os.path.join(directory, file)
    # extract years
    match = re.match(r'(\d{2})-(\d{2})', file)
    if match:
        start_year = 2000 + int(match.group(1))
        end_year = 2000 + int(match.group(2))
    else:
        continue
    # read csv
    # try to detect header
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if 'Area Name' in first_line:
            header = 0
        else:
            header = 1
    df = pd.read_csv(filepath, header=header, on_bad_lines='skip')
    # standardize date
    def parse_date(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return pd.NaT
        try:
            month, day = map(int, date_str.split('/'))
            year = start_year if month >= 10 else end_year
            return f"{year}-{month:02d}-{day:02d}"
        except:
            return pd.NaT
    df['Date'] = df['Date'].apply(parse_date)
    df['Date'] = pd.to_datetime(df['Date'])
    # season week
    season_start = pd.to_datetime(f"{start_year}-10-01")
    df['Season_Week'] = ((df['Date'] - season_start).dt.days // 7) + 1
    # standardize area name
    df['Area Name'] = df['Area Name'].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    # standardize species columns
    species_cols = [col for col in df.columns if 'Species' in col]
    for col in species_cols:
        df[col] = df[col].str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    # append to combined
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# save combined
combined_df.to_csv(os.path.join(directory, 'Processed_Combined_Harvest_F.F.csv'), index=False)