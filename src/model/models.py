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
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

# Import caching system
try:
    from src.cache.prediction_cache import PredictionCache, predict_with_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("Cache module not available. Running without caching.")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        return iterable

# Initialize global cache instance
if CACHE_AVAILABLE:
    prediction_cache = PredictionCache(
        cache_dir="cache/predictions",
        ttl_hours=24,  # Cache expires after 24 hours
        max_cache_size_mb=100  # Maximum 100MB cache size
    )
else:
    prediction_cache = None


#Normalising X using MinMaxScaler 
def normalize_features(X):
    '''Normalize features into different ranges for training'''
    features = []

    #normalizing geographic coordinates
    coords = X[['start_lng','start_lat','end_lng','end_lat']]
    coordsnorm = pd.DataFrame(MinMaxScaler((-1,1)).fit_transform(coords),
    index=coords.index, columns=coords.columns)
    features.append(coordsnorm)

    #normalizing distance & duration
    dist = X[['manhattan','euclidean','gmaps_distance','gmaps_duration']]
    distnorm = pd.DataFrame(MinMaxScaler((0,10)).fit_transform(dist),
    index=dist.index,columns=dist.columns)
    features.append(distnorm)

    #normalizing precipitation
    precipitation = X[['precipitation']]
    prenorm = pd.DataFrame(MinMaxScaler((0,1)).fit_transform(precipitation),
    index=precipitation.index,columns=precipitation.columns)
    features.append(prenorm)

    #normalizing time features
    times = X[['weekday','hour']]
    timesnorm = pd.DataFrame(MinMaxScaler((0,5)).fit_transform(times),
    index=times.index,columns=times.columns)
    features.append(timesnorm)

    flags = X[['holiday','airport','citycenter','standalone','routing_error','short_trip']]
    features.append(flags)

    return pd.concat(features,axis=1)

#Visualization for tree based models
def plot_feature_importance(model,X):
    '''Plot feature importance for tree based models'''
    imp = pd.DataFrame(model.feature_importances_,
        index=X.columns,
        columns=['Importance']).sort_values('Importance',ascending=False)
    imp.plot(kind='barh')
    plt.show()

#Visualization for neural network
def plot_loss_curve(history):
    '''Plot training vs validation loss for neural networks.'''
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

#Linear regression model
def train_linear_regression(Xn_train,Yn_train,Xn_val,Yn_val):
    start_time = time.time()

    model = LinearRegression()
    model.fit(Xn_train,Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val,preds))

    end_time = time.time()
    
    logging.info(f'LINEAR REGRESSION\nRMSE: {rmse} \nTime: {end_time-start_time}')
    logging.info("Linear Regression Done!")
    logging.info("-----")
    return model

#Ridge regression model
def train_ridge_regression(Xn_train,Yn_train,Xn_val,Yn_val,alpha=0.5):
    start_time = time.time()

    model = Ridge(alpha=alpha)
    model.fit(Xn_train,Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val,preds))

    end_time = time.time()
    
    logging.info(f'RIDGE REGRESSION\nRMSE: {rmse} \nTime: {end_time-start_time}')
    logging.info("Ridge Regression Done!")
    logging.info("-----")
    return model

#Lasso regression model
def train_lasso_regression(Xn_train,Yn_train,Xn_val,Yn_val,alpha=0.1):
    start_time = time.time()

    model = Lasso(alpha=alpha,max_iter=5000)
    model.fit(Xn_train,Yn_train)
    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val,preds))

    end_time = time.time()
    
    logging.info(f'LASSO REGRESSION\nRMSE: {rmse} \nTime: {end_time-start_time}')
    logging.info("Lasso Regression Done!")
    logging.info("-----")
    return model

#SVR model
def train_svr(X_train,Y_train,X_val,Y_val):
    start_time = time.time()
    model = SVR()
    model.fit(X_train,Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val,preds))
    end_time = time.time()
    
    logging.info(f'SVR\nRMSE: {rmse} \nTime: {end_time-start_time}')
    logging.info("Support Vector Regression Done!")
    logging.info("-----")
    return model

#XGB model
def train_xgb(X_train,Y_train,X_val,Y_val):
    start_time = time.time()    

    model = XGBRegressor(n_estimators=500, learning_rate=0.045,max_depth=9,reg_lambda=0.5)
    model.fit(X_train,Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val,preds))

    end_time = time.time()
    
    logging.info(f'XGBOOST\nRMSE: {rmse} \nTime: {end_time-start_time}')
    plot_feature_importance(model, X_train)
    logging.info("XGBoost Done!")
    logging.info("-----")
    return model

#Random Forest model
def train_random_forest(X_train,Y_train,X_val,Y_val):
    start_time = time.time()

    model = RandomForestRegressor(n_estimators=500)
    model.fit(X_train,Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val,preds))

    end_time = time.time()
    
    logging.info(f'RANDOM FOREST\nRMSE: {rmse} \nTime: {end_time-start_time}')
    plot_feature_importance(model, X_train)
    logging.info("Random Forest Done!")
    logging.info("-----")
    return model

#neural network
def train_neural_network(Xn_train,Yn_train,Xn_val,Yn_val):
    start_time = time.time()

    model = Sequential()
    model.add(Dense(20,kernel_initializer='normal',input_dim=Xn_train.shape[1],activation='relu'))
    model.add(Dense(150,activation='relu',activity_regularizer=l2(0.2)))
    model.add(Dense(60,activation='relu',activity_regularizer=l2(0.2)))
    model.add(Dense(1,kernel_initializer='normal',activation='linear'))
    model.compile(loss='mse',optimizer='adam')

    history = model.fit(Xn_train,Yn_train,epochs=150,batch_size=50,verbose=2,validation_split=0.2)
    plot_loss_curve(history)

    preds = model.predict(Xn_val)
    rmse = np.sqrt(mean_squared_error(Yn_val,preds))

    end_time = time.time()
    
    logging.info(f'NEURAL NETWORK\nRMSE: {rmse} \nTime: {end_time-start_time}')
    logging.info("Neural Network Done!")
    logging.info("-----")
    return model
 
#Running all the models
def run_regression_models(train_df,models_to_run=None):
    '''Train multiple models on train_df and return them as a dictionary'''
    if models_to_run is None:
        models_to_run = ['XGB']

    X = train_df.drop(columns=['duration'],axis=1)
    Y = train_df['duration']
    Xn = normalize_features(X)
    logging.info("Normalized train and test dataset!")
    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=1)
    Xn_train,Xn_val,Yn_train,Yn_val = train_test_split(Xn,Y,test_size=0.2,random_state=1)

    results = {}
    if 'LINREG' in models_to_run:
        logging.info("Running Linear Regression...")
        results['Linear Regression'] = train_linear_regression(Xn_train,Yn_train,Xn_val,Yn_val)

    if 'RIDGE' in models_to_run:
        logging.info("Running Ridge Regression...")
        results['Ridge Regression'] = train_ridge_regression(Xn_train,Yn_train,Xn_val,Yn_val)

    if 'LASSO' in models_to_run:
        logging.info("Running Lasso Regression...")
        results['Lasso Regression'] = train_lasso_regression(Xn_train,Yn_train,Xn_val,Yn_val)

    if 'SVR' in models_to_run:
        logging.info("Running Support Vector Regression...")
        results['Support Vector Regression'] = train_svr(X_train,Y_train,X_val,Y_val)

    if 'XGB' in models_to_run:
        logging.info("Running XGBoost...")
        results['XGBoost'] = train_xgb(X_train,Y_train,X_val,Y_val)

    if 'RF' in models_to_run:
        logging.info("Running Random Forest...")
        results['Random Forest'] = train_random_forest(X_train,Y_train,X_val,Y_val)

    if 'NN' in models_to_run:
        logging.info("Running Neural Network...")
        results['Neural Network'] = train_neural_network(Xn_train,Yn_train,Xn_val,Yn_val)

    return results


# ============================================================================
# PREDICTION AND EVALUATION FUNCTIONS WITH CACHING
# ============================================================================

def predict_duration(model, test_df, model_name="Model", use_cache=False):
    """
    Make predictions on test data with optional caching
    
    Args:
        model: Trained model
        test_df: Test dataframe (without duration column)
        model_name: Name of the model for logging
        use_cache: Whether to use caching (default: False for compatibility)
    
    Returns:
        numpy array: Predictions
    """
    logging.info(f"Making predictions with {model_name}...")
    
    # Remove duration column if it exists
    if 'duration' in test_df.columns:
        X_test = test_df.drop('duration', axis=1)
    else:
        X_test = test_df

    # Align test features with the model's expected feature set/order
    try:
        expected = None
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                expected = list(booster.feature_names)

        if expected is not None:
            # Add any missing columns as zeros
            missing = [c for c in expected if c not in X_test.columns]
            if missing:
                logging.warning(f"Adding missing features with zeros: {missing[:10]}{'...' if len(missing)>10 else ''}")
                for c in missing:
                    X_test[c] = 0

            # Drop unexpected extra columns
            extra = [c for c in X_test.columns if c not in expected]
            if extra:
                logging.warning(f"Dropping unexpected features: {extra[:10]}{'...' if len(extra)>10 else ''}")
                X_test = X_test.drop(columns=extra)

            # Reorder to match training order exactly
            X_test = X_test[expected]
    except Exception as align_err:
        logging.warning(f"Feature alignment skipped due to: {align_err}")

    # Use caching if enabled and available
    if use_cache and CACHE_AVAILABLE and prediction_cache is not None:
        return predict_duration_with_cache(model, X_test, model_name)
    else:
        predictions = model.predict(X_test)
        logging.info(f"{model_name} predictions completed!")
        logging.info("-----")
        return predictions


def predict_duration_with_cache(model, test_df, model_name="Model"):
    """
    Make predictions with caching support.
    
    Args:
        model: Trained model
        test_df: Test DataFrame with features
        model_name: Name of the model for logging
        
    Returns:
        Array of predictions
    """
    predictions = []
    cache_hits = 0
    cache_misses = 0
    
    for idx, row in test_df.iterrows():
        # Create route parameter dict from row
        route_params = row.to_dict()
        
        # Check cache first
        cached_pred = prediction_cache.get(route_params)
        
        if cached_pred is not None:
            predictions.append(cached_pred)
            cache_hits += 1
        else:
            # Make prediction
            pred = model.predict(pd.DataFrame([row]))[0]
            predictions.append(float(pred))
            
            # Store in cache
            prediction_cache.set(route_params, float(pred))
            cache_misses += 1
    
    # Log cache statistics
    stats = prediction_cache.get_stats()
    logging.info(f"{model_name} predictions completed with caching!")
    logging.info(f"Cache Performance - Hits: {cache_hits}, Misses: {cache_misses}")
    logging.info(f"Overall Cache Stats - Hit Rate: {stats['hit_rate']}, "
                f"Entries: {stats['cache_entries']}, "
                f"Size: {stats['cache_size_mb']} MB")
    logging.info("-----")
    
    return np.array(predictions)


def compare_predictions(pred_1, pred_2, title="Prediction 1 vs Prediction 2", save_plot=True):
    """
    Compare two sets of predictions using histograms
    
    Args:
        pred_1: First set of predictions
        pred_2: Second set of predictions
        title: Title for the plot
        save_plot: Whether to save the plot
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
    Create submission file from predictions
    
    Args:
        prediction: Array of predictions
        output_dir: Directory to save submission file
    
    Returns:
        str: Path to saved submission file
    """
    os.makedirs(output_dir, exist_ok=True)
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_string = f"{output_dir}/test_prediction_{date_string}.csv"
    
    try:
        df = pd.DataFrame(prediction, columns=["duration"])
    except:
        df = pd.DataFrame(prediction.flatten(), columns=["duration"])
        
    df.index.name = "row_id"
    df.to_csv(file_string)
    
    logging.info(f"Submission file saved: {file_string}")
    return file_string


# ============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# ============================================================================

def hyperparameter_tuning_xgb(train_df, test_size=0.2, random_state=1):
    """
    Perform hyperparameter tuning for XGBoost
    
    Args:
        train_df: Training dataframe
        test_size: Test size for train-validation split
        random_state: Random state for reproducibility
    
    Returns:
        tuple: (best_model, best_params, best_rmse)
    """
    logging.info("Starting XGBoost hyperparameter tuning...")
    logging.info("=" * 50)
    
    # Prepare training data
    X = train_df.drop("duration", axis=1)
    Y = train_df.duration
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    # XGBoost hyperparameter tuning
    max_depths = [7, 8, 9, 10, 11]
    learning_rates = [0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
    optimum = np.ones((3,3)) * float('inf')
    
    total_combinations = len(max_depths) * len(learning_rates)
    current_combination = 0
    
    for max_depth in max_depths:
        for learning_rate in learning_rates:
            current_combination += 1
            logging.info(f"Progress: {current_combination}/{total_combinations}")
            
            xgb = XGBRegressor(
                max_depth=max_depth, 
                learning_rate=learning_rate, 
                n_estimators=500, 
                reg_lambda=0.5
            )
            
            logging.info(f"Training: max_depth={max_depth}, learning_rate={learning_rate}")
            xgb.fit(X_train, Y_train)
            pred_xgb = xgb.predict(X_val)
            error = np.sqrt(mean_squared_error(pred_xgb, Y_val))
            
            logging.info(f"RMSE: {error:.4f}")
            
            # Update optimum parameters
            if error < optimum[0,0]:
                optimum[2,:], optimum[1,:] = optimum[1,:], optimum[0,:]
                optimum[0,:] = np.array([error, max_depth, learning_rate])
            elif error < optimum[1,0]:
                optimum[2,:] = optimum[1,:]
                optimum[1,:] = np.array([error, max_depth, learning_rate])
            elif error < optimum[2,0]:
                optimum[2,:] = np.array([error, max_depth, learning_rate])
            
            logging.info("-----")
    
    # Display optimal hyperparameters and fit final model
    logging.info("=== HYPERPARAMETER TUNING RESULTS ===")
    logging.info(f'Top 3 optimal hyperparameters:')
    for i in range(3):
        logging.info(f'{i+1}. RMSE: {optimum[i][0]:.4f}, max_depth: {int(optimum[i][1])}, learning_rate: {optimum[i][2]:.3f}')
    
    # Train final model with best parameters
    best_max_depth = int(optimum[0][1])
    best_learning_rate = optimum[0][2]
    
    logging.info(f"Training final model with best parameters...")
    xgb_final = XGBRegressor(
        max_depth=best_max_depth,  
        learning_rate=best_learning_rate, 
        n_estimators=500, 
        reg_lambda=0.5
    )
    xgb_final.fit(X_train, Y_train)
    pred_xgb_final = xgb_final.predict(X_val)
    
    final_rmse = np.sqrt(mean_squared_error(pred_xgb_final, Y_val))
    logging.info(f'Final XGBoost RMSE with {len(X_train)} training data points: {final_rmse:.4f}')
    
    # Plot feature importance
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


# ============================================================================
# COMPLETE PIPELINE FUNCTION WITH CACHING
# ============================================================================

def run_complete_pipeline(train_df, test_df, models_to_run=None, 
                         tune_xgb=False, create_submission=True, use_cache=True):
    """
    Run the complete ML pipeline including training, evaluation, and submission
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        models_to_run: List of models to train
        tune_xgb: Whether to perform hyperparameter tuning for XGBoost
        create_submission: Whether to create submission file
        use_cache: Whether to use prediction caching (default: True)
    
    Returns:
        dict: Complete pipeline results
    """
    logging.info("Starting Complete ML Pipeline...")
    logging.info("=" * 60)
    
    if use_cache and CACHE_AVAILABLE:
        logging.info("Caching is ENABLED for predictions")
    else:
        logging.info("Caching is DISABLED for predictions")
    
    # Step 1: Train models
    if tune_xgb and 'XGB' in (models_to_run or ['XGB']):
        logging.info("Performing XGBoost hyperparameter tuning...")
        best_xgb, best_params, best_rmse = hyperparameter_tuning_xgb(train_df)
        
        # Train all models including tuned XGBoost
        models = run_regression_models(train_df, models_to_run)
        models['XGBoost_Tuned'] = best_xgb
        
        logging.info(f"Best XGBoost parameters: {best_params}")
        logging.info(f"Best XGBoost RMSE: {best_rmse:.4f}")
        
    else:
        models = run_regression_models(train_df, models_to_run)
    
    # Step 2: Make predictions on test data with optional caching
    predictions = {}
    for model_name, model in models.items():
        pred = predict_duration(model, test_df, model_name, use_cache=use_cache)
        predictions[model_name] = pred
    
    # Step 3: Compare predictions (if multiple models)
    if len(predictions) > 1:
        model_names = list(predictions.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                compare_predictions(
                    predictions[model_names[i]], 
                    predictions[model_names[j]],
                    title=f"{model_names[i]} vs {model_names[j]}"
                )
    
    # Step 4: Create submission files
    submission_files = {}
    if create_submission:
        for model_name, pred in predictions.items():
            submission_file = to_submission(pred, f"output/{model_name.lower().replace(' ', '_')}")
            submission_files[model_name] = submission_file
    
    # Step 5: Display final cache statistics
    if use_cache and CACHE_AVAILABLE and prediction_cache is not None:
        stats = prediction_cache.get_stats()
        logging.info(f"\n{'='*60}")
        logging.info(f"FINAL CACHE STATISTICS:")
        logging.info(f"  Total Requests: {stats['total_requests']}")
        logging.info(f"  Cache Hits: {stats['hits']}")
        logging.info(f"  Cache Misses: {stats['misses']}")
        logging.info(f"  Hit Rate: {stats['hit_rate']}")
        logging.info(f"  Cache Entries: {stats['cache_entries']}")
        logging.info(f"  Cache Size: {stats['cache_size_mb']} MB")
        logging.info(f"{'='*60}\n")
    
    # Step 6: Prepare results
    results = {
        'models': models,
        'predictions': predictions,
        'submission_files': submission_files,
        'cache_stats': prediction_cache.get_stats() if (use_cache and CACHE_AVAILABLE and prediction_cache) else None
    }
    
    logging.info("Complete ML Pipeline finished successfully!")
    logging.info("=" * 60)
    
    return results