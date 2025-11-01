# GoPredict - Machine Learning Pipeline for Trip Duration Prediction

A comprehensive machine learning pipeline for predicting trip durations using various regression models, feature engineering, and hyperparameter optimization.

**Medium post**: https://medium.com/@hphadtare02/how-machine-learning-predicts-trip-duration-just-like-uber-zomato-91f7db6e9ce9

## Demo

https://github.com/user-attachments/assets/f3e1078a-28d6-44e3-a69a-db3a73587c80

## Project Structure

```
GoPredict/
├── main.py                          # Main runner script
├── start_api.py                     # API server startup script
├── test_api.py                      # API testing script
├── config.py                        # Project configuration
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
├── CONTRIBUTING.md                  # Development and integration guide
├── CODE_OF_CONDUCT.md               # Code of conduct and security
│
├── api/                            # FastAPI backend
│   └── main.py                     # FastAPI application
│
├── frontend/                       # React frontend
│   └── src/
│       └── lib/
│           └── api.ts              # API client library
│
├── data/                            # Data directory
│   ├── raw/                         # Raw data files
│   │   ├── train.csv               # Training data
│   │   └── test.csv                # Test data
│   ├── processed/                   # Processed data files
│   │   ├── feature_engineered_train.csv
│   │   ├── feature_engineered_test.csv
│   │   └── gmapsdata/              # Google Maps data
│   └── external/                    # External data sources
│       └── precipitation.csv       # Weather data
│
├── src/                            # Source code
│   ├── model/                      # Model-related modules
│   │   ├── models.py              # All ML models and pipeline
│   │   ├── evaluation.py          # Model evaluation functions
│   │   └── save_models.py         # Model persistence
│   ├── features/                   # Feature engineering modules
│   │   ├── distance.py            # Distance calculations
│   │   ├── geolocation.py         # Geographic features
│   │   ├── gmaps.py               # Google Maps integration
│   │   ├── precipitation.py       # Weather features
│   │   ├── time.py                # Time-based features
│   │   └── weather_api.py         # Weather API integration
│   ├── feature_pipe.py            # Feature engineering pipeline
│   ├── data_preprocessing.py      # Data preprocessing
│   └── complete_pipeline.py       # Complete ML pipeline
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Feature engineering
│   ├── 03_Model_Training.ipynb    # Model training
│   ├── figures/                   # Generated plots
│   └── gmaps/                     # Interactive maps
│
├── saved_models/                   # Trained models (auto-created)
├── output/                         # Predictions and submissions (auto-created)
└── logs/                          # Log files (auto-created)
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd GoPredict

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs output saved_models
```

### 2. API Server

Start the FastAPI server to connect your frontend with ML models:

```bash
# Start the API server
python start_api.py

# Test the API
python test_api.py

# View API documentation
# Visit http://localhost:8000/docs
```

### 3. Frontend Development

```bash
# Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev
```

## API Documentation

The GoPredict API provides REST endpoints for machine learning-based trip duration prediction using FastAPI.

### Quick API Start

```bash
# Start the API server
python start_api.py

# Or with custom options
python start_api.py --host 0.0.0.0 --port 8000 --reload
```

### API Access Points

- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Core API Endpoints

#### Weather API

**`GET /weather`** - Get weather data for a specific location and time

**Parameters:**

- `latitude` (float): Latitude coordinate
- `longitude` (float): Longitude coordinate
- `timestamp` (str): ISO format timestamp (e.g., "2016-01-01T17:00:00")

**Example:**

```bash
curl "http://localhost:8000/weather?latitude=40.767937&longitude=-73.982155&timestamp=2016-01-01T17:00:00"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "temp": 5.0,
    "humidity": 53.0,
    "pressure": 1013.25
  },
  "location": { "latitude": 40.767937, "longitude": -73.982155 },
  "timestamp": "2016-01-01T17:00:00"
}
```

#### Distance Calculation API

**`POST /distance`** - Calculate Manhattan and/or Euclidean distances

**Parameters:**

- `start_lat` (float): Starting latitude
- `start_lng` (float): Starting longitude
- `end_lat` (float): Ending latitude
- `end_lng` (float): Ending longitude
- `method` (str): "manhattan", "euclidean", or "both" (default: "both")

**Example:**

```bash
curl -X POST "http://localhost:8000/distance" \
  -H "Content-Type: application/json" \
  -d '{
    "start_lat": 40.767937,
    "start_lng": -73.982155,
    "end_lat": 40.748817,
    "end_lng": -73.985428,
    "method": "both"
  }'
```

#### Time Features API

**`POST /time-features`** - Extract time-based features from datetime

**Parameters:**

- `datetime_str` (str): ISO format datetime string

**Example:**

```bash
curl -X POST "http://localhost:8000/time-features" \
  -H "Content-Type: application/json" \
  -d '{"datetime_str": "2016-01-01T17:00:00"}'
```

#### Prediction API

**`POST /predict`** - Predict trip duration using ML models

**Parameters (JSON Body):**

```json
{
  "from": {
    "lat": 40.767937,
    "lon": -73.982155
  },
  "to": {
    "lat": 40.748817,
    "lon": -73.985428
  },
  "startTime": "2016-01-01T17:00:00",
  "city": "new_york",
  "model_name": "XGBoost"
}
```

**Response:**

```json
{
  "minutes": 5.2,
  "confidence": 0.75,
  "model_version": "XGBoost",
  "distance_km": 2.1,
  "city": "new_york"
}
```

#### Model Management API

**`GET /models`** - List available trained models
**`GET /models/{model_name}`** - Get specific model information
**`POST /models/train`** - Train models in background

**Example:**

```bash
# List models
curl "http://localhost:8000/models"

# Train models
curl -X POST "http://localhost:8000/models/train" \
  -H "Content-Type: application/json" \
  -d '{"models_to_run": ["XGBoost", "Random Forest"]}'
```

#### Health & Status API

**`GET /health`** - Health check endpoint
**`GET /status`** - Detailed API status

### Frontend Integration

The frontend uses the API client in `frontend/src/lib/api.ts`:

```typescript
import { predictTravelTime } from "@/lib/api";

// Example usage
const prediction = await predictTravelTime({
  from: { lat: 40.767937, lon: -73.982155 },
  to: { lat: 40.748817, lon: -73.985428 },
  startTime: "2016-01-01T17:00:00",
  city: "new_york",
});
```

## ML Pipeline Usage

### Simple Pipeline (Default)

```bash
python main.py
```

Runs the complete end-to-end pipeline:

- **Data preprocessing** - Loads and cleans raw data
- **Feature engineering** - Adds distance, time, cluster, and weather features
- **Model training** - Trains all specified models
- **Model evaluation** - Compares model performance
- **Prediction generation** - Creates submission files

### Custom Models

```bash
python main.py --models XGB,RF
```

Train only specific models.

### With Hyperparameter Tuning

```bash
python main.py --tune-xgb
```

Enable XGBoost hyperparameter tuning.

## 📈 Output Files

### Predictions

- `output/[model_name]/test_prediction_YYYYMMDD_HHMMSS.csv`
- Ready-to-submit prediction files with timestamps

### Models

- `saved_models/[model_name]_YYYYMMDD_HHMMSS.pkl`
- Trained models with metadata

### Logs

- `logs/main.log` - Complete pipeline execution log
- Detailed progress tracking and metrics

### Visualizations

- `output/prediction_comparison_YYYYMMDD_HHMMSS.png`
- Model comparison plots
- Feature importance plots

## 🔧 Configuration

Edit `config.py` to customize:

- Model parameters
- Data paths
- Output directories
- Hyperparameter tuning ranges
- Logging settings

## Usage Examples

### Basic Usage

```python
from src.model.models import run_complete_pipeline
import pandas as pd

# Load data
train_df = pd.read_csv('data/processed/feature_engineered_train.csv')
test_df = pd.read_csv('data/processed/feature_engineered_test.csv')

# Run complete pipeline
results = run_complete_pipeline(
    train_df=train_df,
    test_df=test_df,
    models_to_run=['LINREG', 'RIDGE', 'XGB'],
    tune_xgb=True,
    create_submission=True
)
```

### Individual Components

```python
from src.model.models import run_regression_models, predict_duration, to_submission

# Train models
models = run_regression_models(train_df, ['XGB', 'RF'])

# Make predictions
predictions = predict_duration(models['XGBoost'], test_df)

# Create submission
submission = to_submission(predictions, test_df)
submission.to_csv('my_submission.csv', index=False)
```

## Testing

### API Testing

```bash
# Run comprehensive API tests
python test_api.py
```

### Frontend Testing

```bash
cd frontend
npm run test
npm run test:coverage
```

## Available Models

- **LINREG** - Linear Regression
- **RIDGE** - Ridge Regression
- **LASSO** - Lasso Regression
- **SVR** - Support Vector Regression
- **XGB** - XGBoost
- **RF** - Random Forest
- **NN** - Neural Network

## ontributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and frontend integration details.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for our community guidelines and security policies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
