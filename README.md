# GoPredict - Machine Learning Pipeline for Trip Duration Prediction

A comprehensive machine learning pipeline for predicting trip durations using various regression models, feature engineering, and hyperparameter optimization.

Medium post - https://medium.com/@hphadtare02/how-machine-learning-predicts-trip-duration-just-like-uber-zomato-91f7db6e9ce9

## 📁 Project Structure

```
GoPredict/
├── main.py                          # Main runner script
├── config.py                        # Project configuration
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
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
# Clone the repository
git clone <your-repo-url>
cd GoPredict

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs output saved_models
```

### 2. Data Preparation

Ensure you have the following data files in place:

- `data/raw/train.csv` - Training data
- `data/raw/test.csv` - Test data
- `data/external/precipitation.csv` - Weather data

### 3. Run the Pipeline

```bash
# Run COMPLETE end-to-end pipeline (RECOMMENDED)
python main.py --mode complete

# Run complete pipeline with all models (assumes feature engineering is done)
python main.py --mode full

# Train specific models only (assumes feature engineering is done)
python main.py --mode train --models LINREG,RIDGE,XGB

# Make predictions only (assumes feature engineering is done)
python main.py --mode predict --models XGB

# Hyperparameter tuning only (assumes feature engineering is done)
python main.py --mode tune

# Enable XGBoost hyperparameter tuning
python main.py --mode complete --tune-xgb
```

## 📊 Available Models

| Model                     | Code     | Description                        |
| ------------------------- | -------- | ---------------------------------- |
| Linear Regression         | `LINREG` | Baseline linear model              |
| Ridge Regression          | `RIDGE`  | Linear with L2 regularization      |
| Lasso Regression          | `LASSO`  | Linear with L1 regularization      |
| Support Vector Regression | `SVR`    | Support vector machine             |
| XGBoost                   | `XGB`    | Gradient boosting (best performer) |
| Random Forest             | `RF`     | Ensemble of decision trees         |
| Neural Network            | `NN`     | Deep learning model                |

## 🎯 Usage

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

## 📝 Usage Examples

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
submission_file = to_submission(predictions)
```

### Hyperparameter Tuning

```python
from src.model.models import hyperparameter_tuning_xgb

# Tune XGBoost
best_model, best_params, best_rmse = hyperparameter_tuning_xgb(train_df)
print(f"Best RMSE: {best_rmse}")
print(f"Best parameters: {best_params}")
```

## 🎨 Features

### Data Processing

- **Feature Engineering**: Distance calculations, time features, weather data
- **Normalization**: Custom normalization for different feature types
- **Data Validation**: Automatic data quality checks

### Model Training

- **Multiple Algorithms**: 7 different regression models
- **Hyperparameter Tuning**: Automated XGBoost optimization
- **Cross-Validation**: Built-in validation splits
- **Progress Tracking**: Detailed logging with sandwich format

### Evaluation

- **Comprehensive Metrics**: RMSE, MAE, R², MAPE
- **Visual Comparisons**: Histogram comparisons, feature importance
- **Model Persistence**: Save and load trained models

### Output

- **Submission Files**: Ready-to-submit CSV files
- **Visualizations**: Plots and charts for analysis
- **Logging**: Complete audit trail

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**

   ```
   FileNotFoundError: Data file not found
   ```

   Solution: Ensure all required data files are in the correct directories

2. **Import Errors**

   ```
   ModuleNotFoundError: No module named 'xgboost'
   ```

   Solution: Install missing dependencies: `pip install -r requirements.txt`

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Reduce batch size or use fewer models

### Getting Help

- Check logs in `logs/main.log` for detailed error messages
- Verify data files are in correct format and location
- Ensure all dependencies are installed correctly

## 📊 Performance

Typical model performance on validation set:

- **XGBoost**: ~400-450 RMSE (best performer)
- **Random Forest**: ~420-470 RMSE
- **Linear Models**: ~450-500 RMSE
- **Neural Network**: ~430-480 RMSE

## 🔮 Future Enhancements

- [ ] Automated feature selection
- [ ] Real-time prediction API
- [ ] Model monitoring dashboard
- [ ] A/B testing framework

  ## 📄 License

  This project is licensed under the MIT License - see the LICENSE file for details.

  ## 🤝 Contributing

  Please read [CONTRIBUTING.md](CONTRIBUTING.md). By participating, you agree to abide by our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and report vulnerabilities per [SECURITY.md](SECURITY.md).

  1. Fork the repository
  2. Create a feature branch
  3. Make your changes
  4. Add tests if applicable
  5. Submit a pull request

  ## 📞 Support

For questions or issues, please:

1. Check the logs first
2. Review this documentation
