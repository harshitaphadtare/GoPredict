import sys
from pathlib import Path

# Allow running this file directly (python tests/test_preprocessing.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import pandas as pd
import pytest
from src.data_preprocessing import(
    convert_dtype,
    clean_data,
    filter_duration,
    invalid_routes,
    fix_longitudes,
    preprocess
)

@pytest.fixture
def sample_config():
    return {
        "convert_dtypes": {"pickup_datetime": "datetime"},
        "cleaning": {"drop_nulls": True},
        "duration": {"max_seconds": 7200, "drop_zero_durations": True},
        "invalid_routes": [],
        "longitudes": {"start_lng_negative_fix": True, "end_lng_fix": True, "end_lng_ocean_threshold": -60}
    }

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "pickup_datetime": ["2021-01-01 08:00:00", None],
        "duration": [100, 0],
        "start_lng": [80, -73],
        "end_lng": [85, 200],
        "start_lat": [19.1, 19.2],
        "end_lat": [19.1, 20.0]
    })

def test_convert_dtype(sample_df, sample_config):
    df = convert_dtype(sample_df.copy(), sample_config)
    assert pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])

def test_clean_data(sample_df, sample_config):
    df = clean_data(sample_df.copy(), sample_config)
    assert df.isnull().sum().sum() == 0

def test_filter_duration(sample_df, sample_config):
    df = filter_duration(sample_df.copy(), sample_config)
    assert all(df["duration"] <= 7200)

def test_invalid_routes(sample_df, sample_config):
    df = invalid_routes(sample_df.copy(), sample_config)
    assert isinstance(df, pd.DataFrame)

def test_fix_longitudes(sample_df, sample_config):
    df = fix_longitudes(sample_df.copy(), sample_config)
    assert all(df["start_lng"].notnull())

def test_preprocess_pipeline(sample_df, sample_config):
    train_df, test_df = preprocess(sample_df.copy(), sample_df.copy(), sample_config)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

if __name__ == "__main__":
    pytest.main()