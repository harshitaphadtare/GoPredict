import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import yaml
import logging
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
CONFIG_PATH = PROJECT_ROOT / "config" / "preprocess_config.yaml"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_data(train_path,test_path):
    '''Load train and test CSV files'''
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def convert_dtype(df,config):
    '''Convert data type of datetime'''  
    for col,_ in config.get("convert_dtypes",{}).items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def clean_data(df,config):
    '''Dropping null values'''
    cleaning_config = config.get('cleaning',{})

    if cleaning_config.get("drop_nulls",False):
        df = df.dropna()
    return df


def filter_duration(df,config):
    '''Dropping trips out of duration thresholds & 
    remarking the wrongly marked trips '''
    
    duration_config = config.get('duration',{})

    #dropping trips with more than 2hrs of duration
    max_sec = duration_config.get('max_seconds',None)
    if max_sec: 
        duration_2hrs = df[df['duration'] > max_sec]
        df = df.drop(duration_2hrs.index)

    #dropping zero duration
    if duration_config.get('drop_zero_durations',False):
        zero_durations = (df.duration == 0) & ((df.start_lng != df.end_lng)|(df.start_lat != df.end_lat))
        df = df.drop(df.loc[zero_durations].index,axis=0)

    return df


def invalid_routes(df,config):
    '''Dropping invalid routes'''
    invalid_config = config.get('invalid_routes',{})
    df = df.drop(invalid_config,axis=0,errors='ignore')
    return df
    

def fix_longitudes(df,config):
    '''Fix misclassified longitudes and drop out of bounds'''
    # Config key is singular 'longitude' in YAML
    fix_config = config.get('longitude',{})

    if fix_config.get('start_lng_negative_fix',False):
        percent25 = df['start_lng'].quantile(0.25)
        percent75 = df['start_lng'].quantile(0.75)

        IQR = percent75 - percent25

        upper_limit = percent75 + 1.5*IQR
        start_lng_outliers = df[df['start_lng']>upper_limit]
        df.loc[start_lng_outliers.index,"start_lng"] = -df.loc[start_lng_outliers.index].start_lng 

    if fix_config.get('end_lng_fix',False):
        threshold = fix_config.get('end_lng_ocean_threshold',None)
        if threshold is not None:
            end_lng_outliers = df[df['end_lng']>threshold]
            df = df.drop(end_lng_outliers.index,axis=0)
        
    return df


def base_process(df,config):
    '''Run base preprocessing pipeline'''
    df = convert_dtype(df,config)
    df = clean_data(df,config)
    return df


def preprocess(train_df,test_df,config):
    '''Run full preprocessing pipeline''' 
    train_df = base_process(train_df,config)
    test_df = base_process(test_df,config)
    logging.info("Base Processing complete...")

    if train_df is not None:
        train_df = filter_duration(train_df,config)
        train_df = invalid_routes(train_df,config)
        train_df = fix_longitudes(train_df,config)
    logging.info("Final Processing complete...")
    
    return train_df,test_df


def save_data(train_df,test_df,config):
    '''Save processed train and test data to specified paths.'''
    os.makedirs(os.path.dirname(config["saved_paths"]["processed_train"]), exist_ok=True)
    train_df.to_csv(config["saved_paths"]["processed_train"],index=False)
    test_df.to_csv(config["saved_paths"]["processed_test"],index=False)
    


if __name__ == '__main__':
    #laoding config
    config = load_config()

    #load raw data
    train_df,test_df = load_data("data/raw/train.csv","data/raw/test.csv")

    #preprocess data
    train_df,test_df = preprocess(train_df,test_df,config)

    #save processed data
    save_data(train_df,test_df,config)
    
    logging.info("Preprocessing complete. Processed files saved!")

