import dis
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import googlemaps
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil


def extract_gmaps_data(df,api_key,test=False):
    '''Extract Google Maps distance and duration data'''
    gmaps = googlemaps.Client(key=api_key)

    #store distances and durations from google maps
    distances = []
    durations = []

    #create pickup and dropoff coordinate strings
    df['pickup'] = df.start_lat.astype(str) + ',' + df.start_lng.astype(str)
    df['dropoff'] = df.end_lat.astype(str) + ',' + df.end_lng.astype(str)

    #determine the index range for batching requests
    firstindex = df.index[0]
    finalindex = df.index[-1]
    interval = finalindex - firstindex

    #loop over data in batches of 9 rows
    for i in tqdm(range(interval // 9 + 1)):
        lastindex = firstindex + 8
        df_9 = df.loc[firstindex:lastindex]
        
        # Make distance matrix API call for this batch
        result = gmaps.distance_matrix(df_9.pickup.values,df_9.dropoff.values,units='metric')

        #extract distance and duration from API result
        for i in range(len(df_9)):
            try:
                distances.append(result['rows'][i]['elements'][i]['distance']['value'])
                durations.append(result['rows'][i]['elements'][i]['duration']['value'])
            except:
                distances.append(0)
                durations.append(0)
        
        firstindex = lastindex + 1

    #add the results as new column to the DataFrame
    df['gmaps_distance'] = np.array(distances)
    df['gmaps_duration'] = np.array(durations)

    #save data depending on whether its train or test
    if test:
        save_gmaps_test_data(df)
    else:
        save_gmaps_train_data(df)

    return df

def _ensure_gmaps_output_dir(subfolder: str = "") -> Path:
    """Return the output directory path and create it if it doesn't exist.
    Args:
        subfolder: Optional subfolder name (e.g., 'train', 'test')
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed" / "gmapsdata"
    
    if subfolder:
        output_dir = output_dir / subfolder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Save preprocessed google maps data for train set
def save_gmaps_train_data(df):
    output_dir = _ensure_gmaps_output_dir("train")
    file_path = output_dir / f"{df.index[0]}-{df.index[-1]}.csv"
    df[["gmaps_distance","gmaps_duration"]].to_csv(file_path, index=False)

# Save preprocessed google maps data for test set
def save_gmaps_test_data(df):
    output_dir = _ensure_gmaps_output_dir("test")
    file_path = output_dir / f"{df.index[0]}-{df.index[-1]}.csv"
    df[["gmaps_distance","gmaps_duration"]].to_csv(file_path, index=False)


#Merging gmaps data 
def merge_gmaps_data():
    """Merge all batched CSV files from train and test subdirectories into single files.
    
    This function:
    1. Reads all CSV files from train/ and test/ subdirectories
    2. Extracts row_id information from filenames (format: start-end.csv)
    3. Merges them into single gmaps_train_data.csv and gmaps_test_data.csv files with row_id
    4. Deletes the train/ and test/ subdirectories
    """
    
    # Get the base gmapsdata directory
    project_root = Path(__file__).resolve().parents[2]
    gmaps_dir = project_root / "data" / "processed" / "gmapsdata"
    
    # Process train data
    train_dir = gmaps_dir / "train"
    if train_dir.exists():
        train_files = list(train_dir.glob("*.csv"))
        if train_files:
            print(f"Merging {len(train_files)} train CSV files...")
            train_dfs = []
            for file in sorted(train_files):
                df = pd.read_csv(file)
                
                # Extract row_id range from filename (e.g., "0-9999.csv")
                filename = file.stem  # Remove .csv extension
                start_id, end_id = map(int, filename.split('-'))
                
                # Create row_id column with the range
                df['row_id'] = range(start_id, start_id + len(df))
                train_dfs.append(df)
            
            # Concatenate all train dataframes
            merged_train = pd.concat(train_dfs, ignore_index=True)
            
            # Save merged train data with row_id as index
            train_output_path = gmaps_dir / "gmaps_train_data.csv"
            merged_train.to_csv(train_output_path, index=False)
            print(f"Saved merged train data: {train_output_path}")
            
            # Remove train directory
            shutil.rmtree(train_dir)
            print("Removed train/ subdirectory")
    
    # Process test data
    test_dir = gmaps_dir / "test"
    if test_dir.exists():
        test_files = list(test_dir.glob("*.csv"))
        if test_files:
            print(f"Merging {len(test_files)} test CSV files...")
            test_dfs = []
            for file in sorted(test_files):
                df = pd.read_csv(file)
                
                # Extract row_id range from filename (e.g., "0-9999.csv")
                filename = file.stem  # Remove .csv extension
                start_id, end_id = map(int, filename.split('-'))
                
                # Create row_id column with the range
                df['row_id'] = range(start_id, start_id + len(df))
                test_dfs.append(df)
            
            # Concatenate all test dataframes
            merged_test = pd.concat(test_dfs, ignore_index=True)
            
            # Save merged test data with row_id as index
            test_output_path = gmaps_dir / "gmaps_test_data.csv"
            merged_test.to_csv(test_output_path, index=False)
            print(f"Saved merged test data: {test_output_path}")
            
            # Remove test directory
            shutil.rmtree(test_dir)
            print("Removed test/ subdirectory")
    
    print("Merge operation completed successfully!")