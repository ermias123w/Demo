#!/usr/bin/env python3
"""
Installation script for Crypto Prediction System
"""

import os
import sys
import subprocess
import platform

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"📦 STEP {step_num}: {title}")
    print(f"{'='*60}")

def run_command(command, description=""):
    """Run a system command"""
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Error: {result.stderr}")
        return False
    
    return True

def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing required packages...")
    
    # Core dependencies
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "yfinance>=0.1.70",
        "streamlit>=1.10.0",
        "plotly>=5.0.0",
        "ta>=0.10.0",
        "vaderSentiment>=3.3.2",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "tensorflow>=2.9.0",
        "nltk>=3.7",
        "beautifulsoup4>=4.11.0",
        "python-telegram-bot>=20.0",
        "discord.py>=2.0.0",
        "python-dotenv>=0.19.0",
        "schedule>=1.1.0",
        "psutil>=5.8.0",
        "colorama>=0.4.4",
        "tqdm>=4.64.0",
        "ccxt>=1.90.0",
        "praw>=7.6.0",
        "tweepy>=4.10.0",
        "newsapi-python>=0.2.6",
        "cryptography>=3.4.8",
        "joblib>=1.1.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.8.0",
        "httpx>=0.23.0",
        "aiohttp>=3.8.0",
        "websockets>=10.3",
        "python-multipart>=0.0.5"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True

def setup_directories():
    """Set up necessary directories"""
    directories = [
        "data",
        "models",
        "logs",
        "backups",
        "reports",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_env_file():
    """Create environment file template"""
    env_content = """# Crypto Prediction System Environment Variables

# API Keys (replace with your actual keys)
COINGECKO_API_KEY=your_coingecko_key_here
NEWS_API_KEY=your_news_api_key_here
CRYPTOPANIC_API_KEY=your_cryptopanic_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=CryptoPredictionBot/1.0

# Notification Settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=recipient@gmail.com

# Database Settings
DATABASE_URL=sqlite:///crypto_predictor.db

# Trading Settings (Paper Trading Mode)
PAPER_TRADING=true
INITIAL_BALANCE=10000
MAX_POSITION_SIZE=0.1
RISK_LEVEL=medium

# Platform Settings
PLATFORM=replit
MEMORY_LIMIT=512
DEBUG=false
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file template")
    print("📝 Please edit .env file with your API keys")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️ Error downloading NLTK data: {e}")
    
    return True

def create_run_script():
    """Create run script for easy execution"""
    
    if platform.system() == "Windows":
        script_content = '''@echo off
echo Starting Crypto Prediction System...
python step_by_step_guide.py
pause
'''
        with open('run.bat', 'w') as f:
            f.write(script_content)
        print("✅ Created run.bat script")
    else:
        script_content = '''#!/bin/bash
echo "Starting Crypto Prediction System..."
python3 step_by_step_guide.py
read -p "Press Enter to exit..."
'''
        with open('run.sh', 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod('run.sh', 0o755)
        print("✅ Created run.sh script")

def main():
    """Main installation function"""
    print("""
🚀 CRYPTO PREDICTION SYSTEM INSTALLER
=====================================

This installer will set up everything you need to run the
Crypto Prediction System on your machine.

Requirements:
- Python 3.8 or higher
- Internet connection
- At least 2GB free disk space
- 512MB+ RAM available

Let's get started!
    """)
    
    input("Press Enter to begin installation...")
    
    # Step 1: Check Python version
    print_step(1, "CHECKING PYTHON VERSION")
    if not check_python_version():
        return
    
    # Step 2: Install dependencies
    print_step(2, "INSTALLING DEPENDENCIES")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Step 3: Setup directories
    print_step(3, "SETTING UP DIRECTORIES")
    if not setup_directories():
        print("❌ Failed to setup directories")
        return
    
    # Step 4: Create environment file
    print_step(4, "CREATING ENVIRONMENT FILE")
    if not create_env_file():
        print("❌ Failed to create environment file")
        return
    
    # Step 5: Download NLTK data
    print_step(5, "DOWNLOADING NLTK DATA")
    if not download_nltk_data():
        print("❌ Failed to download NLTK data")
        return
    
    # Step 6: Create run script
    print_step(6, "CREATING RUN SCRIPT")
    create_run_script()
    
    # Final message
    print(f"\n{'='*60}")
    print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("""
✅ Everything is set up and ready to go!

📋 NEXT STEPS:
1. Edit the .env file with your API keys
2. Run the system with: python step_by_step_guide.py
3. Or use the run script: ./run.sh (Linux/Mac) or run.bat (Windows)

🔧 CONFIGURATION:
- Edit crypto_predictor/config/config.py for system settings
- Check .env file for API keys and credentials
- Logs will be saved in the logs/ directory

📚 DOCUMENTATION:
- Read README.md for detailed instructions
- Check QUICKSTART.md for a 5-minute guide
- Use --help flag for command-line options

🎯 GETTING STARTED:
python step_by_step_guide.py

Happy trading! 🚀
    """)

if __name__ == "__main__":
    main()