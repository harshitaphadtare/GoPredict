import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model for logging
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Ensure 1D numpy arrays (avoid NxN broadcasting when subtracting DataFrames)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Some models/loader pipelines return shape (n, 1). Flatten safely
    if y_true.ndim > 1:
        y_true = np.ravel(y_true)
    if y_pred.ndim > 1:
        y_pred = np.ravel(y_pred)

    # Guard: ensure same length
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length. Got {y_true.shape[0]} and {y_pred.shape[0]}"
        )

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage error (avoid divide-by-zero; use nonzero mask)
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = float("nan")
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape,
        'MSE': mse
    }
    
    # Log metrics
    logging.info(f"=== {model_name.upper()} EVALUATION ===")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    logging.info("-----")
    
    return metrics

def compare_models(model_results, save_plots=True, output_dir="saved_models"):
    """
    Compare multiple models and create visualizations
    
    Args:
        model_results: List of dictionaries containing model metrics
        save_plots: Whether to save comparison plots
        output_dir: Directory to save plots and results
    """
    # Create results DataFrame
    results_df = pd.DataFrame(model_results)
    
    # Log comparison
    logging.info("=== MODEL COMPARISON ===")
    logging.info(results_df.to_string(index=False))
    logging.info("-----")
    
    # Create visualizations
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # RMSE comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y='RMSE')
        plt.title('Model Comparison - RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_rmse_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # R² Score comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y='R2_Score')
        plt.title('Model Comparison - R² Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_r2_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Multiple metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['RMSE', 'MAE', 'R2_Score', 'MAPE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            sns.barplot(data=results_df, x='Model', y=metric, ax=ax)
            ax.set_title(f'Model Comparison - {metric}')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comprehensive_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{output_dir}/model_comparison_{timestamp}.csv", index=False)
    
    return results_df

def save_evaluation_logs(model_results, output_dir="saved_models"):
    """
    Save detailed evaluation logs to file
    
    Args:
        model_results: List of dictionaries containing model metrics
        output_dir: Directory to save logs
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = f"{output_dir}/evaluation_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Model Evaluation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for result in model_results:
            f.write(f"Model: {result['Model']}\n")
            f.write(f"RMSE: {result['RMSE']:.4f}\n")
            f.write(f"MAE: {result['MAE']:.4f}\n")
            f.write(f"R² Score: {result['R2_Score']:.4f}\n")
            f.write(f"MAPE: {result['MAPE']:.2f}%\n")
            f.write("-" * 30 + "\n")
    
    logging.info(f"Evaluation logs saved to: {log_file}")

#visualization function for ML evaluation code that generates and saves prediction-vs-actual scatter plots and residual (error) histograms

def plot_prediction_analysis(model, X_test, y_test, model_name="Model", output_dir="output"):
    """
    Generate and save prediction-vs-actual scatter plots and residual histograms.
    
    Args:
        model: Trained model
        X_test: Test features 
        y_test: Test targets
        model_name: Name for plot titles and filename
        output_dir: Directory to save plots (default: "output")
    
    Returns:
        str: Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Make predictions 
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # for Plot 1: Prediction vs Actual scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # this is for Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Trip Duration (seconds)')
    plt.ylabel('Predicted Trip Duration (seconds)')
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #this is for Plot 2: Residual (Error) Histogram
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
   #check for output directory and create if not exists
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{model_name.replace(' ', '_')}_prediction_analysis.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 {model_name} prediction analysis saved: {filename}")
    
    return filename