import pandas as pd
from pathlib import Path

def extract_precipitation_data(combine, precep_path=None):
    '''Add precipitation values to train/test df (updates in place)'''
    
    # If no path provided, try to find the precipitation.csv file
    if precep_path is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  
        precep_path = project_root / 'data' / 'external' / 'precipitation.csv'
        
        if not precep_path.exists():
            raise FileNotFoundError(f"Could not find precipitation.csv at {precep_path}")

    # Load and format precipitation data
    precipitate = pd.read_csv(precep_path)
    precipitate['date'] = pd.to_datetime(precipitate['date'], dayfirst=True).dt.date

    for i in range(len(combine)):
        df = combine[i]

        # Ensure date is in proper format
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Merge precipitation
        df = df.merge(
            precipitate[['date', 'precipitation']],
            on='date',
            how='left',
            suffixes=("", "_prec")
        )

        # Replace NaNs with 0
        df['precipitation'] = df['precipitation'].fillna(0.0)

        # Drop the temporary date column
        df.drop("date", axis=1, inplace=True)

        # Write back into the same list
        combine[i] = df

    print("Added precipitation data successfully!")
