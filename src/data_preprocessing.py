import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import logging
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_data(train_path, test_path):
    '''Load train and test CSV files'''
    train_df = pd.read_csv(train_path, index_col='row_id')
    test_df = pd.read_csv(test_path, index_col='row_id')
    return train_df, test_df


def convert_dtype(df):
    '''Convert data type of datetime'''  
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def clean_data(df):
    '''Dropping null values'''
    # Drop rows with null values
    df = df.dropna()
    return df


def filter_duration(df):
    '''Dropping trips out of duration thresholds & 
    remarking the wrongly marked trips '''
    
    # Dropping trips with more than 2 hours of duration (7200 seconds)
    max_sec = 7200
    duration_2hrs = df[df['duration'] > max_sec]
    df = df.drop(duration_2hrs.index)

    # Dropping zero duration trips where start and end are different
    zero_durations = (df.duration == 0) & ((df.start_lng != df.end_lng)|(df.start_lat != df.end_lat))
    df = df.drop(df.loc[zero_durations].index, axis=0)

    return df


def invalid_routes(df):
    '''Dropping invalid routes'''
    invalid_indices = [90810, 81553, 83473, 87892, 96380] 
    df = df.drop(invalid_indices, axis=0, errors='ignore')
    return df
    

def fix_longitudes(df):
    '''Fix misclassified longitudes and drop out of bounds'''
    
    # Fix start longitude negative values
    percent25 = df['start_lng'].quantile(0.25)
    percent75 = df['start_lng'].quantile(0.75)
    IQR = percent75 - percent25
    upper_limit = percent75 + 1.5*IQR
    start_lng_outliers = df[df['start_lng']>upper_limit]
    df.loc[start_lng_outliers.index,"start_lng"] = -df.loc[start_lng_outliers.index].start_lng 

    # Fix end longitude (drop trips ending in ocean)
    threshold = -73.5  # Ocean threshold for end longitude
    end_lng_outliers = df[df['end_lng']>threshold]
    df = df.drop(end_lng_outliers.index, axis=0)
        
    return df


def base_process(df):
    '''Run base preprocessing pipeline'''
    df = convert_dtype(df)
    df = clean_data(df)
    return df


def preprocess(train_df, test_df):
    '''Run full preprocessing pipeline''' 
    train_df = base_process(train_df)
    test_df = base_process(test_df)
    logging.info("Base Processing complete...")

    if train_df is not None:
        train_df = filter_duration(train_df)
        train_df = invalid_routes(train_df)
        train_df = fix_longitudes(train_df)
    logging.info("Final Processing complete...")
    
    return train_df, test_df


def generate_data_summary(df, dataset_name, output_path):
    '''Generate and save a summary report of the dataset'''
    summary = []
    summary.append(f"Data Summary for {dataset_name}")
    summary.append("=" * 50)
    summary.append(f"Shape: {df.shape}")
    summary.append(f"Columns: {list(df.columns)}")
    summary.append("")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary.append("Missing Values:")
        for col, count in missing[missing > 0].items():
            summary.append(f"  {col}: {count}")
    else:
        summary.append("No missing values")
    summary.append("")
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary.append("Numeric Columns Statistics:")
        stats = df[numeric_cols].describe()
        summary.append(stats.to_string())
        summary.append("")
    
    # Specific to trip duration if present
    if 'duration' in df.columns:
        summary.append("Trip Duration Statistics:")
        summary.append(f"  Mean: {df['duration'].mean():.2f} seconds")
        summary.append(f"  Median: {df['duration'].median():.2f} seconds")
        summary.append(f"  Min: {df['duration'].min():.2f} seconds")
        summary.append(f"  Max: {df['duration'].max():.2f} seconds")
        summary.append("")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    
    logging.info(f"Data summary saved to {output_path}")
    


if __name__ == '__main__':
    # Load raw data
    train_df, test_df = load_data("data/raw/train.csv", "data/raw/test.csv")

    # Preprocess data
    train_df, test_df = preprocess(train_df, test_df)

    # Save processed data
    train_output_path = "data/processed/eda_processed_train.csv"
    test_output_path = "data/processed/eda_processed_test.csv"
    
    print("Starting Feature Engineering pipeline...")
    save_data(train_df, test_df, train_output_path, test_output_path)
    
    logging.info(". ")
    print("[OK]Preprocessing completed successfully!!")
