import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import numpy as np
import pandas as pd
from features.distance import calc_distance
from features.time import extract_time_features
from features.geolocation import clustering
from features.precipitation import extract_precipitation_data


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_eda_data(train_path, test_path):
    '''Load EDA Processed train and test CSV files'''
    train_df = pd.read_csv(train_path, index_col='row_id')
    test_df = pd.read_csv(test_path, index_col='row_id')

    return train_df,test_df


def calc_manhattan_euclidean_dist(combine_df):
    '''Calculating Manhattan and Euclidean (Havesine) distances'''

    for df in combine_df:
        df['manhattan'] = calc_distance(df,method='manhattan')
        df['euclidean'] = calc_distance(df,method='euclidean')

    return combine_df


def add_gmaps_features(combine_df, train_gmaps_path, test_gmaps_path):
    '''Add Google Maps distance and duration features from pre-generated data'''
    
    # Load pre-generated gmaps data
    gmaps_train = pd.read_csv(train_gmaps_path, index_col='row_id')
    gmaps_test = pd.read_csv(test_gmaps_path, index_col='row_id')
    
    # Add gmaps features to train data
    train_df = combine_df[0]
    train_df['gmaps_distance'] = gmaps_train['gmaps_distance']
    train_df['gmaps_duration'] = gmaps_train['gmaps_duration']
    
    # Add gmaps features to test data  
    test_df = combine_df[1]
    test_df['gmaps_distance'] = gmaps_test['gmaps_distance']
    test_df['gmaps_duration'] = gmaps_test['gmaps_duration']
    
    # Handle Treasure Island routing errors (0 distance cases)
    for df in combine_df:
        TI_df = df[df['gmaps_distance']==0].loc[df.manhattan>2000]
        # Replace with manhattan distance
        df.loc[TI_df.index,"gmaps_distance"] = TI_df.manhattan
        # Approximate gmaps_duration 
        df.loc[TI_df.index,"gmaps_duration"] = TI_df.manhattan/11.0
    
    return combine_df


def add_time_features(combine_df):
    '''Extract weekdays,hour and date columns and drop datetime
    column. Add holidays column'''
    extract_time_features(combine_df)
    return combine_df


def add_cluster_features(combine_df):
    ''' Run DBSCAN clustering on coordinates to group locations 
    into clusters of airports,city centers etc '''
    train_df = combine_df[0]
    test_df = combine_df[1]

    clustering(train_df,test_df)
    return combine_df


def add_precipitation_data(combine_df):
    '''Add precipitation values to train/test df'''
    extract_precipitation_data(combine_df)
    return combine_df


def marking_outliers(combine_df):
    '''marking routing errors and short trips'''
    for df in combine_df: 
        df['routing_error'] = np.zeros(df.index.shape)
        df['short_trip'] = np.zeros(df.index.shape)

        df.loc[(df.gmaps_distance > 500) & (df.manhattan < 50),"routing_error"] = 1
        df.loc[(df.gmaps_distance < 500) & (df.manhattan < 50),"short_trip"] = 1

    return combine_df
    
def save_feature_eng_data(train_df, test_df, train_output_path, test_output_path):
    '''Save Feature engineered train and test data to specified paths.'''
    
    # Save feature engineered data
    train_df.to_csv(train_output_path)
    test_df.to_csv(test_output_path)
    
    logging.info(f"Feature engineered train data saved to: {train_output_path}")
    logging.info(f"Feature engineered test data saved to: {test_output_path}")
    
    return train_df, test_df


def cleanup_intermediate_files(eda_train_path, eda_test_path):
    '''Delete intermediate EDA processed files after feature engineering is complete'''
    
    try:
        if eda_train_path.exists():
            eda_train_path.unlink()
            logging.info(f"Deleted intermediate file: {eda_train_path}")
        
        if eda_test_path.exists():
            eda_test_path.unlink()
            logging.info(f"Deleted intermediate file: {eda_test_path}")
            
    except Exception as e:
        logging.warning(f"Could not delete intermediate files: {e}")


if __name__ == '__main__':

    # Input paths (EDA processed data)
    train_path = PROJECT_ROOT / "data" / "processed" / "eda_processed_train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "eda_processed_test.csv"
    
    # Google Maps data paths
    train_gmaps_path = PROJECT_ROOT / "data" / "processed" / "gmapsdata" / "gmaps_train_data.csv"
    test_gmaps_path = PROJECT_ROOT / "data" / "processed" / "gmapsdata" / "gmaps_test_data.csv"
    
    # Output paths (Feature engineered data)
    train_output_path = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
    test_output_path = PROJECT_ROOT / "data" / "processed" / "feature_engineered_test.csv"
    
    # Load data
    train_df, test_df = load_eda_data(train_path, test_path)
    combine = [train_df, test_df]

    print("Starting Feature Engineering pipeline...")
    
    # Add distance features
    logging.info("Adding distance features...")
    combine = calc_manhattan_euclidean_dist(combine)
    logging.info("Distance features added!")
    
    # Add gmaps features from pre-generated data
    logging.info("Adding gmaps features...")
    combine = add_gmaps_features(combine, train_gmaps_path, test_gmaps_path)
    logging.info("Gmaps features added!")

    #Add Time features 
    logging.info("Adding time features...")
    combine = add_time_features(combine)
    logging.info("Time features added!")

    #Add Cluster features   
    logging.info("Adding cluster features...")
    combine = add_cluster_features(combine)
    logging.info("Cluster features added!")

    #Add Precipitation data
    logging.info("Adding precipitation data...")
    combine = add_precipitation_data(combine)
    logging.info("Precipitation data added!")

    #Marking outliers
    logging.info("Marking outliers...")
    combine = marking_outliers(combine)
    logging.info("Outliers marked!")

    # Save feature engineered data
    logging.info("Saving feature engineered data...")
    train_df, test_df = save_feature_eng_data(combine[0], combine[1], train_output_path, test_output_path)
    
    # Clean up intermediate files
    logging.info("Cleaning up intermediate files...")
    cleanup_intermediate_files(train_path, test_path)

    print("[OK] Feature pipeline completed successfully!!")
    print(f"Final train data shape: {train_df.shape}")
    print(f"Final test data shape: {test_df.shape}")