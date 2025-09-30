#!/usr/bin/env python3
"""
GoPredict - Simplified ML Pipeline

Simple entry point for the GoPredict project.
Runs complete pipeline from raw data to predictions.

Usage:
    python main.py                           # Run complete pipeline
    python main.py --models XGB,RF           # Train specific models
    python main.py --tune-xgb                # Enable hyperparameter tuning
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.complete_pipeline import CompleteMLPipeline

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/main.log")
        ]
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GoPredict ML Pipeline")
    parser.add_argument("--models", type=str, 
                       default="LINREG,RIDGE,LASSO,SVR,XGB,RF,NN",
                       help="Comma-separated list of models to train")
    parser.add_argument("--tune-xgb", action="store_true",
                       help="Enable XGBoost hyperparameter tuning")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info("Starting GoPredict Pipeline...")
    logging.info(f"Models: {args.models}")
    logging.info("=" * 50)
    
    # Parse models
    models_to_run = [model.strip() for model in args.models.split(",")]
    
    try:
        # Run complete pipeline
        pipeline = CompleteMLPipeline()
        results = pipeline.run_complete_pipeline(
            models_to_run=models_to_run,
            tune_xgb=args.tune_xgb
        )
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
