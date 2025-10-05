"""
GoPredict Configuration File

This file contains all configuration settings for the GoPredict project.
Modify these settings to customize the pipeline behavior.
"""

from pathlib import Path

# Data paths
DATA_PATHS = {
    'raw_train': 'data/raw/train.csv',
    'raw_test': 'data/raw/test.csv',
    'processed_train': 'data/processed/feature_engineered_train.csv',
    'processed_test': 'data/processed/feature_engineered_test.csv',
    'precipitation': 'data/external/precipitation.csv',
    'gmaps_train': 'data/processed/gmapsdata/gmaps_train_data.csv',
    'gmaps_test': 'data/processed/gmapsdata/gmaps_test_data.csv'
}

# Output paths
OUTPUT_PATHS = {
    'models': 'saved_models',
    'logs': 'logs',
    'submissions': 'output',
    'figures': 'notebooks/figures'
}

# Cache Configuration
CACHE_CONFIG = {
    'enabled': True,
    'cache_dir': 'cache/predictions',
    'ttl_hours': 24,  # Cache expiration time in hours
    'max_cache_size_mb': 100,  # Maximum cache size in megabytes
    'clear_on_startup': False  # Whether to clear cache on program start
}

# Model configurations
MODEL_CONFIGS = {
    'linear_regression': {
        'fit_intercept': True
    },
    'ridge_regression': {
        'alpha': 0.5
    },
    'lasso_regression': {
        'alpha': 0.1,
        'max_iter': 5000
    },
    'svr': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'xgboost': {
        'n_estimators': 500,
        'learning_rate': 0.045,
        'max_depth': 9,
        'reg_lambda': 0.5
    },
    'random_forest': {
        'n_estimators': 500,
        'random_state': 42
    },
    'neural_network': {
        'epochs': 150,
        'batch_size': 50,
        'validation_split': 0.2
    }
}

# Hyperparameter tuning configurations
HYPERPARAMETER_TUNING = {
    'xgboost': {
        'max_depths': [7, 8, 9, 10, 11],
        'learning_rates': [0.04, 0.042, 0.044, 0.046, 0.048, 0.05],
        'n_estimators': 500,
        'reg_lambda': 0.5
    }
}

# Data preprocessing settings
PREPROCESSING = {
    'test_size': 0.2,
    'random_state': 1,
    'normalization_ranges': {
        'coordinates': (-1, 1),
        'distances': (0, 10),
        'precipitation': (0, 1),
        'time_features': (0, 5)
    }
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s [%(levelname)s] %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Available models
AVAILABLE_MODELS = [
    'LINREG',    # Linear Regression
    'RIDGE',     # Ridge Regression
    'LASSO',     # Lasso Regression
    'SVR',       # Support Vector Regression
    'XGB',       # XGBoost
    'RF',        # Random Forest
    'NN'         # Neural Network
]

# Feature columns
FEATURE_COLUMNS = {
    'coordinates': ['start_lng', 'start_lat', 'end_lng', 'end_lat'],
    'distances': ['manhattan', 'euclidean', 'gmaps_distance', 'gmaps_duration'],
    'precipitation': ['precipitation'],
    'time_features': ['weekday', 'hour'],
    'flags': ['holiday', 'airport', 'citycenter', 'standalone', 'routing_error', 'short_trip']
}

# Target column
TARGET_COLUMN = 'duration'

# Create cache directory
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)