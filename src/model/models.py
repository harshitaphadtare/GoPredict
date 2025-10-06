import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

# Safe tqdm import; provide no-op fallback if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, desc=None, total=None, unit=None, leave=True):
        # Minimal no-op wrapper to avoid breaking behavior when tqdm is absent
        return iterable if iterable is not None else range(total or 0)

# ==========================
# Utilities and normalizers
# ==========================
def normalize_features(X):
    """Normalize features into different ranges for training"""
    features = []

    coords = X[['start_lng', 'start_lat', 'end_lng', 'end_lat']]
    coordsnorm = pd.DataFrame(
        MinMaxScaler((-1, 1)).fit_transform(coords),
        index=coords.index, columns=coords.columns
    )
    features.append(coordsnorm)

    dist = X[['manhattan', 'euclidean', 'gmaps_distance', 'gmaps_duration']]
    distnorm = pd.DataFrame(
        MinMaxScaler((0, 10)).fit_transform(dist),
        index=dist.index, columns=dist.columns
    )
    features.append(distnorm)

    precipitation = X[['precipitation']]
    prenorm = pd.DataFrame(
        MinMaxScaler((0, 1)).fit_transform(precipitation),
        index=precipitation.index, columns=precipitation.columns
    )
    features.append(prenorm)

    times = X[['weekday', 'hour']]
    timesnorm = pd.DataFrame(
        MinMaxScaler((0, 5)).fit_transform(times),
        index=times.index, columns=times.columns
    )
    features.append(timesnorm)

    flags = X[['holiday', 'airport', 'citycenter', 'standalone', 'routing_error', 'short_trip']]
    features.append(flags)

    return pd.concat(features, axis=1)

def plot_feature_importance(model, X):
    """Plot feature importance for tree based models"""
    imp = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['Importance']
    ).sort_values('Importance', ascending=False)
    imp.plot(kind='barh')
    plt.show()

def plot_loss_curve(history):
    """Plot training vs validation loss for neural networks."""
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# =========================================
# PREDICTION AND SUBMISSION (MOVED UP)
# =========================================
def predict_duration(model, test_df, model_name="Model"):
    """
    Make predictions on test data
    """
    logging.info(f"Making predictions with {model_name}...")

    # Remove duration column if present
    if 'duration' in test_df.columns:
        X_test = test_df.drop('duration', axis=1)
    else:
        X_test = test_df

    # Align test features to model's expected feature set/order where available
    try:
        expected = None
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                expected = list(booster.feature_names)

        if expected is not None:
            missing = [c for c in expected if c not in X_test.columns]
            if missing:
                logging.warning(f"Adding missing features with zeros: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                for c in missing:
                    X_test[c] = 0

            extra = [c for c in X_test.columns if c not in expected]
            if extra:
                logging.warning(f"Dropping unexpected features: {extra[:10]}{'...' if len(extra) > 10 else ''}")
                X_test = X_test.drop(columns=extra)

            X_test = X_test[expected]
    except Exception as align_err:
        logging.warning(f"Feature alignment skipped due to: {align_err}")

    predictions = model.predict(X_test)

    logging.info(f"{model_name} predictions completed!")
    logging.info("-----")

    return predictions

def compare_predictions(pred_1, pred_2, title="Prediction 1 vs Prediction 2", save_plot=True):
    """Compare two sets of predictions using histograms"""
    bins = np.histogram(np.hstack((pred_1, pred_2)), bins=100)[1]  # get the bin edges

    plt.figure(figsize=(10, 6))
    plt.hist(pred_1, bins=bins, alpha=1, label="Prediction 1")
    plt.hist(pred_2, bins=bins, alpha=0.7, label="Prediction 2")
    plt.title(title)
    plt.xlabel("Duration [s]")
    plt.ylabel("Number of instances")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_plot:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"output/prediction_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')

    plt.show()

def to_submission(prediction, output_dir="output"):
    """Create submission file from predictions"""
    os.makedirs(output_dir, exist_ok=True)
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_string = f"{output_dir}/test_prediction_{date_string}.csv"

    try:
        df = pd.DataFrame(prediction, columns=["duration"])
    except Exception:
        df = pd.DataFrame(np.asarray(prediction).flatten(), columns=["duration"])

    df.index.name = "row_id"
    df.to_csv(file_string)

    logging.info(f"Submission file saved: {file_string}")
    return file_string

# ===========================
# Individual model trainers
# ===========================
def train_linear_regression(Xn_train, Yn_train, Xn_val, Yn_val):
    start_time = time.time()
    model = LinearRegression()
    model.fit(Xn_train, Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val, preds))
    end_time = time.time()

    logging.info(f'LINEAR REGRESSION\nRMSE: {rmse} \nTime: {end_time - start_time}')
    logging.info("Linear Regression Done!")
    logging.info("-----")
    return model

def train_ridge_regression(Xn_train, Yn_train, Xn_val, Yn_val, alpha=0.5):
    start_time = time.time()
    model = Ridge(alpha=alpha)
    model.fit(Xn_train, Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val, preds))
    end_time = time.time()

    logging.info(f'RIDGE REGRESSION\nRMSE: {rmse} \nTime: {end_time - start_time}')
    logging.info("Ridge Regression Done!")
    logging.info("-----")
    return model

def train_lasso_regression(Xn_train, Yn_train, Xn_val, Yn_val, alpha=0.1):
    start_time = time.time()
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(Xn_train, Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val, preds))
    end_time = time.time()

    logging.info(f'LASSO REGRESSION\nRMSE: {rmse} \nTime: {end_time - start_time}')
    logging.info("Lasso Regression Done!")
    logging.info("-----")
    return model

def train_svr(X_train, Y_train, X_val, Y_val):
    start_time = time.time()
    model = SVR()
    model.fit(X_train, Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    end_time = time.time()

    logging.info(f'SVR\nRMSE: {rmse} \nTime: {end_time - start_time}')
    logging.info("Support Vector Regression Done!")
    logging.info("-----")
    return model

def train_xgb(X_train, Y_train, X_val, Y_val):
    start_time = time.time()
    model = XGBRegressor(n_estimators=500, learning_rate=0.045, max_depth=9, reg_lambda=0.5, verbosity=0)
    model.fit(X_train, Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    end_time = time.time()

    logging.info(f'XGBOOST\nRMSE: {rmse} \nTime: {end_time - start_time}')
    plot_feature_importance(model, X_train)
    logging.info("XGBoost Done!")
    logging.info("-----")
    return model

def train_random_forest(X_train, Y_train, X_val, Y_val):
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=500)
    model.fit(X_train, Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    end_time = time.time()

    logging.info(f'RANDOM FOREST\nRMSE: {rmse} \nTime: {end_time - start_time}')
    plot_feature_importance(model, X_train)
    logging.info("Random Forest Done!")
    logging.info("-----")
    return model

def train_neural_network(Xn_train, Yn_train, Xn_val, Yn_val):
    start_time = time.time()
    model = Sequential()
    model.add(Dense(20, kernel_initializer='normal', input_dim=Xn_train.shape[1], activation='relu'))
    model.add(Dense(150, activation='relu', activity_regularizer=l2(0.2)))
    model.add(Dense(60, activation='relu', activity_regularizer=l2(0.2)))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(Xn_train, Yn_train, epochs=150, batch_size=50, verbose=2, validation_split=0.2)
    plot_loss_curve(history)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val, preds))
    end_time = time.time()

    logging.info(f'NEURAL NETWORK\nRMSE: {rmse} \nTime: {end_time - start_time}')
    logging.info("Neural Network Done!")
    logging.info("-----")
    return model

# ===========================
# Multi-model training (tqdm)
# ===========================
def run_regression_models(train_df, models_to_run=None):
    """Train multiple models on train_df and return them as a dictionary"""
    if models_to_run is None:
        models_to_run = ['XGB']

    X = train_df.drop(columns=['duration'], axis=1)
    Y = train_df['duration']
    Xn = normalize_features(X)
    logging.info("Normalized train and test dataset!")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)
    Xn_train, Xn_val, Yn_train, Yn_val = train_test_split(Xn, Y, test_size=0.2, random_state=1)

    results = {}

    # tqdm progress over requested models
    for model_key in tqdm(models_to_run, desc="Training Models", unit="model"):
        if model_key == 'LINREG':
            logging.info("Running Linear Regression...")
            results['Linear Regression'] = train_linear_regression(Xn_train, Yn_train, Xn_val, Yn_val)
        elif model_key == 'RIDGE':
            logging.info("Running Ridge Regression...")
            results['Ridge Regression'] = train_ridge_regression(Xn_train, Yn_train, Xn_val, Yn_val)
        elif model_key == 'LASSO':
            logging.info("Running Lasso Regression...")
            results['Lasso Regression'] = train_lasso_regression(Xn_train, Yn_train, Xn_val, Yn_val)
        elif model_key == 'SVR':
            logging.info("Running Support Vector Regression...")
            results['Support Vector Regression'] = train_svr(X_train, Y_train, X_val, Y_val)
        elif model_key == 'XGB':
            logging.info("Running XGBoost...")
            results['XGBoost'] = train_xgb(X_train, Y_train, X_val, Y_val)
        elif model_key == 'RF':
            logging.info("Running Random Forest...")
            results['Random Forest'] = train_random_forest(X_train, Y_train, X_val, Y_val)
        elif model_key == 'NN':
            logging.info("Running Neural Network...")
            results['Neural Network'] = train_neural_network(Xn_train, Yn_train, Xn_val, Yn_val)

    return results

# =========================================
# Hyperparameter tuning (tqdm as requested)
# =========================================
def hyperparameter_tuning_xgb(train_df, test_size=0.2, random_state=1):
    """
    Perform hyperparameter tuning for XGBoost
    """
    logging.info("Starting XGBoost hyperparameter tuning...")
    logging.info("=" * 50)

    # Prepare training data
    X = train_df.drop("duration", axis=1)
    Y = train_df.duration
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Ranges as given in the issue
    max_depths = [7, 8, 9, 10, 11]
    learning_rates = [0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
    optimum = np.ones((3, 3)) * float('inf')

    total = len(max_depths) * len(learning_rates)

    with tqdm(total=total, desc="XGBoost Tuning", unit="combo") as pbar:
        for max_depth in max_depths:
            for learning_rate in learning_rates:
                xgb = XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=500,
                    reg_lambda=0.5,
                    verbosity=0
                )
                xgb.fit(X_train, Y_train)
                pred_xgb = xgb.predict(X_val)
                error = np.sqrt(mean_squared_error(pred_xgb, Y_val))

                # Maintain existing top-3 tracking logic
                if error < optimum[0, 0]:
                    optimum[2, :], optimum[1, :] = optimum[1, :], optimum[0, :]
                    optimum[0, :] = np.array([error, max_depth, learning_rate])
                elif error < optimum[1, 0]:
                    optimum[2, :] = optimum[1, :]
                    optimum[1, :] = np.array([error, max_depth, learning_rate])
                elif error < optimum[2, 0]:
                    optimum[2, :] = np.array([error, max_depth, learning_rate])

                # Postfix exactly as in the issue description
                pbar.set_postfix({
                    'depth': max_depth,
                    'lr': learning_rate,
                    'rmse': f'{error:.2f}'
                })
                pbar.update(1)

    logging.info("=== HYPERPARAMETER TUNING RESULTS ===")
    logging.info('Top 3 optimal hyperparameters:')
    for i in range(3):
        logging.info(f'{i+1}. RMSE: {optimum[i][0]:.4f}, max_depth: {int(optimum[i][1])}, learning_rate: {optimum[i][2]:.3f}')

    best_max_depth = int(optimum[0][1])
    best_learning_rate = optimum[0][2]

    xgb_final = XGBRegressor(
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=500,
        reg_lambda=0.5,
        verbosity=0
    )
    xgb_final.fit(X_train, Y_train)
    pred_xgb_final = xgb_final.predict(X_val)
    final_rmse = np.sqrt(mean_squared_error(pred_xgb_final, Y_val))

    plot_feature_importance(xgb_final, X_train)

    best_params = {
        'max_depth': best_max_depth,
        'learning_rate': best_learning_rate,
        'n_estimators': 500,
        'reg_lambda': 0.5
    }

    logging.info("Hyperparameter tuning completed!")
    logging.info("=" * 50)
    return xgb_final, best_params, final_rmse

# =========================================
# Complete pipeline (uses predict_duration)
# =========================================
def run_complete_pipeline(train_df, test_df, models_to_run=None,
                          tune_xgb=False, create_submission=True):
    """
    Run the complete ML pipeline including training and prediction
    """
    logging.info("Starting Complete ML Pipeline...")
    logging.info("=" * 60)

    # Step 1: Train models (optionally tune XGB)
    if tune_xgb and 'XGB' in (models_to_run or ['XGB']):
        logging.info("Performing XGBoost hyperparameter tuning...")
        best_xgb, best_params, best_rmse = hyperparameter_tuning_xgb(train_df)
        models = run_regression_models(train_df, models_to_run)
        models['XGBoost_Tuned'] = best_xgb
        logging.info(f"Best XGBoost parameters: {best_params}")
        logging.info(f"Best XGBoost RMSE: {best_rmse:.4f}")
    else:
        models = run_regression_models(train_df, models_to_run)

    # Step 2: Predictions for all models (progress over models)
    predictions = {}
    for model_name, model in tqdm(models.items(), desc="Predicting with models", unit="model"):
        pred = predict_duration(model, test_df, model_name)
        predictions[model_name] = pred

    # (Downstream evaluation/compare/submission is performed elsewhere in the project)

    logging.info("Complete ML Pipeline finished successfully!")
    logging.info("=" * 60)

    return {
        'models': models,
        'predictions': predictions
    }


