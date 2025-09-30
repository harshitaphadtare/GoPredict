#!/usr/bin/env python3
"""
GoPredict Setup Script

This script sets up the GoPredict project environment and creates
necessary directories and configuration files.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'output',
        'saved_models',
        'data/raw',
        'data/processed',
        'data/external',
        'data/processed/gmapsdata',
        'notebooks/figures',
        'notebooks/gmaps'
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data files (uncomment if you don't want to track data)
# data/raw/*.csv
# data/processed/*.csv

# Output files
output/*.csv
output/*.png
logs/*.log
saved_models/*.pkl
saved_models/*.h5

# Environment variables
.env
.env.local

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✓ Created: .gitignore")

def check_requirements():
    """Check if requirements.txt exists"""
    if not Path('requirements.txt').exists():
        print("⚠ Warning: requirements.txt not found!")
        print("Please create requirements.txt with your dependencies.")
    else:
        print("✓ Found: requirements.txt")

def create_sample_env():
    """Create sample .env file"""
    env_content = """
# Google Maps API Key (optional)
# Get your API key from: https://developers.google.com/maps/documentation/javascript/get-api-key
GOOGLE_MAPS_API_KEY=your_api_key_here

# Logging Level
LOG_LEVEL=INFO

# Model Configuration
DEFAULT_MODELS=LINREG,RIDGE,LASSO,SVR,XGB,RF,NN
ENABLE_TUNING=false
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_content.strip())
        print("✓ Created: .env (sample)")
    else:
        print("✓ Found: .env")

def main():
    """Main setup function"""
    print("🚀 Setting up GoPredict project...")
    print("=" * 50)
    
    try:
        create_directories()
        create_gitignore()
        create_sample_env()
        check_requirements()
        
        print("\n" + "=" * 50)
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add your data files to data/raw/")
        print("3. Run the pipeline: python main.py --mode full")
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
