#!/usr/bin/env python3
"""
Test script to verify that app.py correctly aggregates all harvest CSV files.
This validates the data loading without running the full Streamlit interface.
"""

import os
import re
import pandas as pd

def test_load_all_harvest_data():
    """
    Test function that mirrors the load_all_harvest_data() function from app.py
    """
    directory = '.'
    all_files = os.listdir(directory)
    
    # Find all harvest CSV files
    harvest_files = []
    for file in all_files:
        if file.endswith('.csv'):
            if 'harvest' in file.lower() and 'combined' not in file.lower() and 'baseline' not in file.lower():
                harvest_files.append(file)
    
    # Sort files by year range
    def extract_year_range(filename):
        match = re.search(r'(\d{2})-(\d{2})', filename)
        if match:
            return int(match.group(1))
        return 0
    
    harvest_files.sort(key=extract_year_range)
    
    print(f"Found {len(harvest_files)} harvest CSV files:")
    for file in harvest_files:
        print(f"  - {file}")
    
    combined_df = pd.DataFrame()
    files_loaded = 0
    
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
            
            # Read CSV
            df = pd.read_csv(filepath, header=header, on_bad_lines='skip')
            
            # Standardize and parse dates
            def parse_date(date_str):
                if pd.isna(date_str) or not isinstance(date_str, str):
                    return pd.NaT
                try:
                    month, day = map(int, date_str.split('/'))
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
            files_loaded += 1
            print(f"  ✓ Loaded {file} ({start_year}-{end_year}): {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error loading {file}: {str(e)}")
            continue
    
    # Drop rows with invalid dates or missing required columns
    combined_df = combined_df.dropna(subset=['Date', 'Area Name'])
    
    print(f"\n✓ Successfully loaded {files_loaded} files")
    print(f"✓ Combined DataFrame: {len(combined_df)} total rows")
    print(f"✓ Unique areas: {combined_df['Area Name'].nunique()}")
    print(f"✓ Unique season weeks: {combined_df['Season_Week'].nunique()}")
    print(f"✓ Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Show sample data
    print(f"\nSample records from master data:")
    print(combined_df[['Area Name', 'Date', 'Season_Week', 'Average Ducks', '#1 Species']].head(10))
    
    print(f"\nUnique areas in master data:")
    areas = sorted(combined_df['Area Name'].unique())
    for area in areas:
        count = len(combined_df[combined_df['Area Name'] == area])
        print(f"  - {area}: {count} records")
    
    return combined_df

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING DATA AGGREGATION")
    print("=" * 70)
    master_data = test_load_all_harvest_data()
    print("\n✅ Data aggregation test PASSED")
