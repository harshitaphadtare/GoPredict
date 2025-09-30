"""
Complete End-to-End ML Pipeline for GoPredict

This pipeline includes:
1. Data preprocessing (data_preprocessing.py)
2. Feature engineering (feature_pipe.py)
3. Model training and evaluation (models.py)
4. Prediction and submission generation

Usage:
    python src/complete_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import preprocessing modules
from data_preprocessing import load_data, preprocess, save_data as save_preprocessed_data
from feature_pipe import (
    load_eda_data, calc_manhattan_euclidean_dist, add_gmaps_features,
    add_time_features, add_cluster_features, add_precipitation_data,
    marking_outliers, save_feature_eng_data, cleanup_intermediate_files
)

# Import model modules
from model.models import run_complete_pipeline, run_regression_models
from model.save_models import save_model, save_model_results
from model.evaluation import evaluate_model, compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class CompleteMLPipeline:
    """
    Complete End-to-End ML Pipeline
    
    This pipeline handles the entire workflow from raw data to final predictions:
    1. Data preprocessing and cleaning
    2. Feature engineering
    3. Model training and evaluation
    4. Prediction generation and submission
    """
    
    def __init__(self, project_root=None):
        """
        Initialize the complete pipeline
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)
        
        # Define all paths
        self.setup_paths()
        
        # Create necessary directories
        self.create_directories()
        
        logging.info(f"Complete ML Pipeline initialized")
        logging.info(f"Project root: {self.project_root}")
    
    def setup_paths(self):
        """Setup all file paths"""
        self.paths = {
            # Raw data paths
            'raw_train': self.project_root / "data" / "raw" / "train.csv",
            'raw_test': self.project_root / "data" / "raw" / "test.csv",
            
            # Intermediate processed data paths
            'eda_train': self.project_root / "data" / "processed" / "eda_processed_train.csv",
            'eda_test': self.project_root / "data" / "processed" / "eda_processed_test.csv",
            
            # Google Maps data paths
            'gmaps_train': self.project_root / "data" / "processed" / "gmapsdata" / "gmaps_train_data.csv",
            'gmaps_test': self.project_root / "data" / "processed" / "gmapsdata" / "gmaps_test_data.csv",
            
            # Final feature engineered data paths
            'feature_train': self.project_root / "data" / "processed" / "feature_engineered_train.csv",
            'feature_test': self.project_root / "data" / "processed" / "feature_engineered_test.csv",
            
            # External data paths
            'precipitation': self.project_root / "data" / "external" / "precipitation.csv",
            
            # Output paths
            'models': self.project_root / "saved_models",
            'output': self.project_root / "output",
            'logs': self.project_root / "logs"
        }
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.project_root / "data" / "processed",
            self.project_root / "data" / "processed" / "gmapsdata",
            self.paths['models'],
            self.paths['output'],
            self.paths['logs']
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def step1_data_preprocessing(self):
        """
        Step 1: Data Preprocessing
        - Load raw data
        - Clean and preprocess data
        - Save intermediate processed data
        """
        logging.info("=" * 60)
        logging.info("STEP 1: DATA PREPROCESSING")
        logging.info("=" * 60)
        
        # Load raw data
        logging.info("Loading raw data...")
        train_df, test_df = load_data(
            str(self.paths['raw_train']), 
            str(self.paths['raw_test'])
        )
        
        logging.info(f"Raw train data shape: {train_df.shape}")
        logging.info(f"Raw test data shape: {test_df.shape}")
        
        # Preprocess data
        logging.info("Preprocessing data...")
        train_df, test_df = preprocess(train_df, test_df)
        
        logging.info(f"Processed train data shape: {train_df.shape}")
        logging.info(f"Processed test data shape: {test_df.shape}")
        
        # Save preprocessed data
        logging.info("Saving preprocessed data...")
        save_preprocessed_data(
            train_df, test_df,
            str(self.paths['eda_train']),
            str(self.paths['eda_test'])
        )
        
        logging.info("✅ Data preprocessing completed!")
        return train_df, test_df
    
    def step2_feature_engineering(self):
        """
        Step 2: Feature Engineering
        - Load preprocessed data
        - Add distance features
        - Add Google Maps features
        - Add time features
        - Add cluster features
        - Add precipitation data
        - Mark outliers
        """
        logging.info("=" * 60)
        logging.info("STEP 2: FEATURE ENGINEERING")
        logging.info("=" * 60)
        
        # Load EDA processed data
        logging.info("Loading preprocessed data...")
        train_df, test_df = load_eda_data(
            str(self.paths['eda_train']),
            str(self.paths['eda_test'])
        )
        combine = [train_df, test_df]
        
        logging.info(f"Preprocessed train data shape: {train_df.shape}")
        logging.info(f"Preprocessed test data shape: {test_df.shape}")
        
        # Add distance features
        logging.info("Adding distance features...")
        combine = calc_manhattan_euclidean_dist(combine)
        logging.info("✅ Distance features added!")
        
        # Add Google Maps features
        logging.info("Adding Google Maps features...")
        combine = add_gmaps_features(
            combine,
            str(self.paths['gmaps_train']),
            str(self.paths['gmaps_test'])
        )
        logging.info("✅ Google Maps features added!")
        
        # Add time features
        logging.info("Adding time features...")
        combine = add_time_features(combine)
        logging.info("✅ Time features added!")
        
        # Add cluster features
        logging.info("Adding cluster features...")
        combine = add_cluster_features(combine)
        logging.info("✅ Cluster features added!")
        
        # Add precipitation data
        logging.info("Adding precipitation data...")
        combine = add_precipitation_data(combine)
        logging.info("✅ Precipitation data added!")
        
        # Mark outliers
        logging.info("Marking outliers...")
        combine = marking_outliers(combine)
        logging.info("✅ Outliers marked!")
        
        # Save feature engineered data
        logging.info("Saving feature engineered data...")
        train_df, test_df = save_feature_eng_data(
            combine[0], combine[1],
            str(self.paths['feature_train']),
            str(self.paths['feature_test'])
        )
        
        logging.info(f"Final train data shape: {train_df.shape}")
        logging.info(f"Final test data shape: {test_df.shape}")
        
        # Clean up intermediate files
        logging.info("Cleaning up intermediate files...")
        cleanup_intermediate_files(
            Path(self.paths['eda_train']),
            Path(self.paths['eda_test'])
        )
        
        logging.info("✅ Feature engineering completed!")
        return train_df, test_df
    
    def step3_model_training(self, train_df, models_to_run=None, tune_xgb=False):
        """
        Step 3: Model Training
        - Train multiple ML models
        - Perform hyperparameter tuning (optional)
        - Evaluate model performance
        """
        logging.info("=" * 60)
        logging.info("STEP 3: MODEL TRAINING")
        logging.info("=" * 60)
        
        if models_to_run is None:
            models_to_run = ['LINREG', 'RIDGE', 'LASSO', 'SVR', 'XGB', 'RF', 'NN']
        
        logging.info(f"Training models: {models_to_run}")
        
        # Train models
        models = run_regression_models(train_df, models_to_run)
        
        # Save trained models
        logging.info("Saving trained models...")
        saved_models = {}
        for model_name, model in models.items():
            model_path = save_model(
                model=model,
                model_name=model_name.replace(' ', '_').lower(),
                output_dir=str(self.paths['models'])
            )
            saved_models[model_name] = model_path
            logging.info(f"✅ Saved {model_name}")
        
        logging.info("✅ Model training completed!")
        return models, saved_models
    
    def step4_model_evaluation(self, models, train_df):
        """
        Step 4: Model Evaluation
        - Evaluate all trained models
        - Compare model performance
        - Identify best performing model
        """
        logging.info("=" * 60)
        logging.info("STEP 4: MODEL EVALUATION")
        logging.info("=" * 60)
        
        # Prepare validation data
        X = train_df.drop(columns=['duration'], axis=1)
        Y = train_df['duration']
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)
        
        # Evaluate models
        results = []
        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}...")
            
            # Make predictions
            if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Neural Network']:
                # Use normalized data for these models
                from model.models import normalize_features
                Xn = normalize_features(X)
                Xn_train, Xn_val, Yn_train, Yn_val = train_test_split(Xn, Y, test_size=0.2, random_state=1)
                metrics = evaluate_model(model, Xn_val, Yn_val, model_name)
            else:
                # Use regular data for tree-based models
                metrics = evaluate_model(model, X_val, Y_val, model_name)
            
            results.append(metrics)
        
        # Compare models
        logging.info("Comparing model performance...")
        comparison_df = compare_models(results, save_plots=True, output_dir=str(self.paths['output']))
        
        # Save evaluation results
        evaluation_results = {
            'model_performance': results,
            'comparison_dataframe': comparison_df.to_dict('records'),
            'best_model': comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model'],
            'best_rmse': comparison_df['RMSE'].min()
        }
        
        save_model_results(evaluation_results, str(self.paths['models']))
        
        logging.info(f"🏆 Best performing model: {evaluation_results['best_model']}")
        logging.info(f"   Best RMSE: {evaluation_results['best_rmse']:.4f}")
        
        logging.info("✅ Model evaluation completed!")
        return results, comparison_df
    
    def step5_prediction_generation(self, models, test_df, comparison_df):
        """
        Step 5: Prediction Generation
        - Make predictions on test data
        - Generate submission files
        """
        logging.info("=" * 60)
        logging.info("STEP 5: PREDICTION GENERATION")
        logging.info("=" * 60)
        
        # Get best model
        best_model_name = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        best_model = models[best_model_name]
        
        logging.info(f"Using best model for predictions: {best_model_name}")
        
        # Make predictions with best model
        from model.models import predict_duration, to_submission
        test_predictions = predict_duration(best_model, test_df, best_model_name)
        
        # Create submission file
        submission_file = to_submission(test_predictions, str(self.paths['output']))
        logging.info(f"📄 Submission file created: {submission_file}")
        
        # Generate predictions for all models
        logging.info("Generating predictions for all models...")
        for model_name, model in models.items():
            predictions = predict_duration(model, test_df, model_name)
            model_submission = to_submission(
                predictions, 
                str(self.paths['output'])
            )
            logging.info(f"📄 {model_name} submission: {model_submission}")
        
        logging.info("✅ Prediction generation completed!")
        return test_predictions, submission_file
    
    def run_complete_pipeline(self, models_to_run=None, tune_xgb=False):
        """
        Run the complete end-to-end pipeline
        
        Args:
            models_to_run: List of models to train
            tune_xgb: Whether to perform XGBoost hyperparameter tuning
        
        Returns:
            dict: Complete pipeline results
        """
        logging.info("🚀 Starting Complete End-to-End ML Pipeline")
        logging.info("=" * 80)
        
        try:
            # Step 1: Data Preprocessing
            train_df_preprocessed, test_df_preprocessed = self.step1_data_preprocessing()
            
            # Step 2: Feature Engineering
            train_df_features, test_df_features = self.step2_feature_engineering()
            
            # Step 3: Model Training
            models, saved_models = self.step3_model_training(train_df_features, models_to_run, tune_xgb)
            
            # Step 4: Model Evaluation
            evaluation_results, comparison_df = self.step4_model_evaluation(models, train_df_features)
            
            # Step 5: Prediction Generation
            test_predictions, submission_file = self.step5_prediction_generation(
                models, test_df_features, comparison_df
            )
            
            # Final summary
            logging.info("=" * 80)
            logging.info("🎉 COMPLETE PIPELINE SUCCESSFULLY COMPLETED!")
            logging.info("=" * 80)
            
            results = {
                'preprocessed_data': {
                    'train_shape': train_df_features.shape,
                    'test_shape': test_df_features.shape
                },
                'models_trained': len(models),
                'best_model': comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model'],
                'best_rmse': comparison_df['RMSE'].min(),
                'submission_file': submission_file,
                'saved_models': saved_models,
                'evaluation_results': evaluation_results
            }
            
            logging.info(f"📊 Final Results:")
            logging.info(f"   Models trained: {results['models_trained']}")
            logging.info(f"   Best model: {results['best_model']}")
            logging.info(f"   Best RMSE: {results['best_rmse']:.4f}")
            logging.info(f"   Submission file: {results['submission_file']}")
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Pipeline failed: {e}")
            raise


def main():
    """Main function to run the complete pipeline"""
    # Initialize pipeline
    pipeline = CompleteMLPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        models_to_run=['LINREG', 'RIDGE', 'LASSO', 'SVR', 'XGB', 'RF', 'NN'],
        tune_xgb=False  # Set to True for hyperparameter tuning
    )
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best RMSE: {results['best_rmse']:.4f}")
    print(f"Submission File: {results['submission_file']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
