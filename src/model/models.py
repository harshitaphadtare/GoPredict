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
    """
    Normalize features into different ranges for model training.
    
    This function applies different normalization ranges to different feature groups:
    coordinates, distances, precipitation, time features, and binary flags.
    
    Args:
        X (pd.DataFrame): Input features containing coordinates (start_lng, start_lat, 
                         end_lng, end_lat), distances (manhattan, euclidean, gmaps_distance, 
                         gmaps_duration), precipitation, time features (weekday, hour), 
                         and flags (holiday, airport, citycenter, standalone, routing_error, 
                         short_trip).
    
    Returns:
        pd.DataFrame: Normalized features with:
            - Coordinates scaled to (-1, 1)
            - Distances scaled to (0, 10)
            - Precipitation scaled to (0, 1)
            - Time features scaled to (0, 5)
            - Flags left unchanged
    
    Examples:
        >>> X_normalized = normalize_features(train_df.drop('duration', axis=1))
        >>> print(X_normalized.shape)
        (10000, 18)
    """
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
    """
    Plot feature importance for tree-based models.
    
    Creates a horizontal bar chart showing the importance of each feature
    as determined by the model's feature_importances_ attribute.
    
    Args:
        model: A trained tree-based model (e.g., RandomForestRegressor, XGBRegressor)
               that has a feature_importances_ attribute.
        X (pd.DataFrame): Input features DataFrame used to get column names.
    
    Returns:
        None: Displays the plot directly using matplotlib.
    
    Examples:
        >>> model = XGBRegressor()
        >>> model.fit(X_train, y_train)
        >>> plot_feature_importance(model, X_train)
    """
    imp = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['Importance']
    ).sort_values('Importance', ascending=False)
    imp.plot(kind='barh')
    plt.show()

def plot_loss_curve(history):
    """
    Plot training vs validation loss curves for neural networks.
    
    Creates a line plot showing how training and validation loss
    evolved across epochs during model training.
    
    Args:
        history: A Keras History object returned by model.fit() containing
                loss and val_loss in its history dictionary.
    
    Returns:
        None: Displays the plot directly using matplotlib.
    
    Examples:
        >>> model = Sequential([...])
        >>> history = model.fit(X_train, y_train, validation_split=0.2, epochs=100)
        >>> plot_loss_curve(history)
    """
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
    Make duration predictions on test data with automatic feature alignment.
    
    This function handles feature alignment by ensuring the test data has
    the same features in the same order as expected by the model.
    
    Args:
        model: A trained sklearn-compatible model with a predict() method.
        test_df (pd.DataFrame): Test dataset, optionally containing 'duration' column
                               which will be dropped if present.
        model_name (str, optional): Name of the model for logging purposes. 
                                   Defaults to "Model".
    
    Returns:
        np.ndarray: Array of predicted duration values.
    
    Examples:
        >>> model = XGBRegressor()
        >>> model.fit(X_train, y_train)
        >>> predictions = predict_duration(model, test_df, "XGBoost")
        >>> print(predictions[:5])
        [450.2, 523.1, 380.5, 612.3, 295.8]
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
    """
    Compare two sets of predictions using overlaid histograms.
    
    Creates a histogram visualization comparing the distribution of two
    prediction sets, useful for analyzing model agreement or differences.
    
    Args:
        pred_1 (np.ndarray or list): First set of predictions.
        pred_2 (np.ndarray or list): Second set of predictions.
        title (str, optional): Plot title. Defaults to "Prediction 1 vs Prediction 2".
        save_plot (bool, optional): Whether to save the plot to the output directory.
                                   Defaults to True.
    
    Returns:
        None: Displays the plot and optionally saves it to output/prediction_comparison_TIMESTAMP.png.
    
    Examples:
        >>> pred_xgb = model_xgb.predict(X_test)
        >>> pred_rf = model_rf.predict(X_test)
        >>> compare_predictions(pred_xgb, pred_rf, "XGBoost vs Random Forest")
    """
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
    """
    Create a CSV submission file from model predictions.
    
    Generates a timestamped CSV file with predictions formatted for
    competition submission, with row_id as index and duration as the column.
    
    Args:
        prediction (np.ndarray or list): Array of predicted duration values.
        output_dir (str, optional): Directory to save the submission file.
                                   Defaults to "output".
    
    Returns:
        str: Path to the saved submission file.
    
    Examples:
        >>> predictions = model.predict(test_df)
        >>> file_path = to_submission(predictions)
        >>> print(file_path)
        'output/test_prediction_20250930_103835.csv'
    """
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
    """
    Train a Linear Regression model and evaluate on validation set.
    
    Fits a standard Linear Regression model using Ordinary Least Squares (OLS)
    and logs the RMSE performance on the validation set along with training time.
    
    Args:
        Xn_train (pd.DataFrame or np.ndarray): Normalized training features.
        Yn_train (pd.Series or np.ndarray): Training target values (durations).
        Xn_val (pd.DataFrame or np.ndarray): Normalized validation features.
        Yn_val (pd.Series or np.ndarray): Validation target values (durations).
    
    Returns:
        LinearRegression: Trained Linear Regression model.
    
    Examples:
        >>> model = train_linear_regression(Xn_train, y_train, Xn_val, y_val)
        >>> predictions = model.predict(Xn_test)
    """
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
    """
    Train a Ridge Regression model with L2 regularization.
    
    Fits a Ridge Regression model with L2 penalty to prevent overfitting
    and logs the RMSE performance on the validation set along with training time.
    
    Args:
        Xn_train (pd.DataFrame or np.ndarray): Normalized training features.
        Yn_train (pd.Series or np.ndarray): Training target values (durations).
        Xn_val (pd.DataFrame or np.ndarray): Normalized validation features.
        Yn_val (pd.Series or np.ndarray): Validation target values (durations).
        alpha (float, optional): Regularization strength. Higher values mean
                                stronger regularization. Defaults to 0.5.
    
    Returns:
        Ridge: Trained Ridge Regression model.
    
    Examples:
        >>> model = train_ridge_regression(Xn_train, y_train, Xn_val, y_val, alpha=1.0)
        >>> predictions = model.predict(Xn_test)
    """
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
    """
    Train a Lasso Regression model with L1 regularization.
    
    Fits a Lasso Regression model with L1 penalty that can perform feature
    selection by driving some coefficients to zero, and logs the RMSE performance
    on the validation set along with training time.
    
    Args:
        Xn_train (pd.DataFrame or np.ndarray): Normalized training features.
        Yn_train (pd.Series or np.ndarray): Training target values (durations).
        Xn_val (pd.DataFrame or np.ndarray): Normalized validation features.
        Yn_val (pd.Series or np.ndarray): Validation target values (durations).
        alpha (float, optional): Regularization strength. Higher values mean
                                stronger regularization. Defaults to 0.1.
    
    Returns:
        Lasso: Trained Lasso Regression model.
    
    Examples:
        >>> model = train_lasso_regression(Xn_train, y_train, Xn_val, y_val, alpha=0.05)
        >>> predictions = model.predict(Xn_test)
    """
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
    """
    Train a Support Vector Regression (SVR) model.
    
    Fits a Support Vector Regression model using default RBF kernel
    and logs the RMSE performance on the validation set along with training time.
    Note: SVR can be computationally expensive on large datasets.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features (non-normalized).
        Y_train (pd.Series or np.ndarray): Training target values (durations).
        X_val (pd.DataFrame or np.ndarray): Validation features (non-normalized).
        Y_val (pd.Series or np.ndarray): Validation target values (durations).
    
    Returns:
        SVR: Trained Support Vector Regression model.
    
    Examples:
        >>> model = train_svr(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
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
    """
    Train an XGBoost Regression model with predefined hyperparameters.
    
    Fits an XGBoost gradient boosting model with specific hyperparameters
    optimized for this problem, logs RMSE performance on validation set,
    and displays feature importance plot.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features (non-normalized).
        Y_train (pd.Series or np.ndarray): Training target values (durations).
        X_val (pd.DataFrame or np.ndarray): Validation features (non-normalized).
        Y_val (pd.Series or np.ndarray): Validation target values (durations).
    
    Returns:
        XGBRegressor: Trained XGBoost model with n_estimators=500, 
                     learning_rate=0.045, max_depth=9, reg_lambda=0.5.
    
    Examples:
        >>> model = train_xgb(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
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
    """
    Train a Random Forest Regression model.
    
    Fits a Random Forest ensemble model with 500 trees, logs RMSE performance
    on validation set, and displays feature importance plot.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features (non-normalized).
        Y_train (pd.Series or np.ndarray): Training target values (durations).
        X_val (pd.DataFrame or np.ndarray): Validation features (non-normalized).
        Y_val (pd.Series or np.ndarray): Validation target values (durations).
    
    Returns:
        RandomForestRegressor: Trained Random Forest model with 500 estimators.
    
    Examples:
        >>> model = train_random_forest(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
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
    """
    Train a Deep Neural Network for regression using Keras.
    
    Builds and trains a fully connected neural network with 3 hidden layers
    and L2 regularization. Displays loss curves and logs RMSE performance
    on validation set.
    
    Architecture:
        - Input layer: 20 neurons (ReLU)
        - Hidden layer 1: 150 neurons (ReLU, L2=0.2)
        - Hidden layer 2: 60 neurons (ReLU, L2=0.2)
        - Output layer: 1 neuron (Linear)
    
    Args:
        Xn_train (pd.DataFrame or np.ndarray): Normalized training features.
        Yn_train (pd.Series or np.ndarray): Training target values (durations).
        Xn_val (pd.DataFrame or np.ndarray): Normalized validation features.
        Yn_val (pd.Series or np.ndarray): Validation target values (durations).
    
    Returns:
        Sequential: Trained Keras Sequential model with MSE loss and Adam optimizer.
    
    Examples:
        >>> model = train_neural_network(Xn_train, y_train, Xn_val, y_val)
        >>> predictions = model.predict(Xn_test)
    """
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
    """
    Train multiple regression models and return them as a dictionary.
    
    This function orchestrates training of multiple models with progress tracking,
    automatically handling data splitting and normalization where needed.
    
    Args:
        train_df (pd.DataFrame): Training dataset containing features and 'duration' column.
        models_to_run (list of str, optional): List of model identifiers to train.
                                               Available options: 'LINREG', 'RIDGE', 'LASSO',
                                               'SVR', 'XGB', 'RF', 'NN'.
                                               Defaults to ['XGB'].
    
    Returns:
        dict: Dictionary mapping model names to trained model objects.
              Keys are descriptive names (e.g., 'XGBoost', 'Random Forest').
    
    Examples:
        >>> models = run_regression_models(train_df, ['XGB', 'RF', 'LINREG'])
        >>> xgb_model = models['XGBoost']
        >>> predictions = xgb_model.predict(test_features)
    """
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
    Perform grid search hyperparameter tuning for XGBoost model.
    
    Searches over max_depth and learning_rate parameters to find the optimal
    combination that minimizes RMSE on validation set. Displays progress and
    tracks top 3 parameter combinations.
    
    Args:
        train_df (pd.DataFrame): Training dataset containing features and 'duration' column.
        test_size (float, optional): Proportion of data to use for validation.
                                    Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 1.
    
    Returns:
        tuple: A tuple containing:
            - XGBRegressor: Best tuned model trained with optimal parameters
            - dict: Dictionary of best hyperparameters
            - float: RMSE of the best model on validation set
    
    Examples:
        >>> best_model, best_params, rmse = hyperparameter_tuning_xgb(train_df)
        >>> print(f"Best parameters: {best_params}")
        >>> print(f"Best RMSE: {rmse:.4f}")
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
    Run the complete machine learning pipeline from training to prediction.
    
    This end-to-end function orchestrates model training, optional hyperparameter
    tuning for XGBoost, and predictions on test data for all specified models.
    
    Args:
        train_df (pd.DataFrame): Training dataset containing features and 'duration' column.
        test_df (pd.DataFrame): Test dataset for making predictions.
        models_to_run (list of str, optional): List of model identifiers to train.
                                               Available: 'LINREG', 'RIDGE', 'LASSO',
                                               'SVR', 'XGB', 'RF', 'NN'.
                                               Defaults to None (uses default from run_regression_models).
        tune_xgb (bool, optional): Whether to perform hyperparameter tuning for XGBoost.
                                  If True, creates an additional 'XGBoost_Tuned' model.
                                  Defaults to False.
        create_submission (bool, optional): Parameter for future submission file creation.
                                           Currently not used. Defaults to True.
    
    Returns:
        dict: Dictionary containing:
            - 'models' (dict): Trained model objects keyed by model name
            - 'predictions' (dict): Prediction arrays keyed by model name
    
    Examples:
        >>> results = run_complete_pipeline(
        ...     train_df, test_df, 
        ...     models_to_run=['XGB', 'RF'],
        ...     tune_xgb=True
        ... )
        >>> xgb_predictions = results['predictions']['XGBoost']
        >>> tuned_xgb_model = results['models']['XGBoost_Tuned']
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


