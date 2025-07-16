#!/usr/bin/env python3
"""
Quick setup script for Crypto Prediction System
"""

import os
import sys
import subprocess

def setup_basic_directories():
    """Set up basic directories"""
    directories = [
        "data",
        "models", 
        "logs",
        "backups"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_basic_packages():
    """Install only essential packages"""
    basic_packages = [
        "numpy",
        "pandas", 
        "requests",
        "yfinance",
        "matplotlib",
        "scikit-learn",
        "vaderSentiment",
        "python-dotenv"
    ]
    
    print("Installing basic packages...")
    for package in basic_packages:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Installed {package}")
            else:
                print(f"⚠️ Failed to install {package}")
        except Exception as e:
            print(f"⚠️ Error installing {package}: {e}")

def create_basic_env():
    """Create basic environment file"""
    env_content = """# Basic configuration for demo
PAPER_TRADING=true
INITIAL_BALANCE=10000
DEBUG=true
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")

def main():
    """Main setup function"""
    print("🚀 Quick Setup for Crypto Prediction System")
    print("==========================================")
    
    print("\n📦 Setting up directories...")
    setup_basic_directories()
    
    print("\n📦 Installing basic packages...")
    install_basic_packages()
    
    print("\n📦 Creating environment file...")
    create_basic_env()
    
    print("\n✅ Basic setup complete!")
    print("Now run: python3 step_by_step_guide.py")

if __name__ == "__main__":
    main()