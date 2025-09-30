#!/bin/bash
# GoPredict Quick Start Script for Unix/Linux/Mac

echo "========================================"
echo "GoPredict - Machine Learning Pipeline"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

# Create necessary directories
mkdir -p logs output saved_models

echo "Starting GoPredict Pipeline..."
echo

# Run the complete pipeline with default parameters
python3 main.py --models LINREG,RIDGE,LASSO,SVR,XGB,RF,NN

echo
echo "Pipeline completed!"
echo "Check the 'output' folder for results"
echo "Check the 'logs' folder for detailed logs"
echo
