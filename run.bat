@echo off
REM GoPredict Quick Start Script for Windows

echo ========================================
echo GoPredict - Machine Learning Pipeline
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if required directories exist
if not exist "logs" mkdir logs
if not exist "output" mkdir output
if not exist "saved_models" mkdir saved_models

echo Starting GoPredict Pipeline...
echo.

REM Run the complete pipeline with default parameters
python main.py --models LINREG,RIDGE,LASSO,SVR,XGB,RF,NN

echo.
echo Pipeline completed!
echo Check the 'output' folder for results
echo Check the 'logs' folder for detailed logs
echo.
pause
