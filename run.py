#!/usr/bin/env python3
"""
Crypto Prediction System - Startup Script
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crypto_predictor.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'torch', 'transformers', 
        'requests', 'scikit-learn', 'ta'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Crypto Prediction System')
    parser.add_argument('--predict', help='Generate prediction for symbol (e.g., BTC)', type=str)
    parser.add_argument('--backtest', help='Run backtest for symbol', type=str)
    parser.add_argument('--days', help='Number of days for backtest', type=int, default=30)
    parser.add_argument('--status', help='Show system status', action='store_true')
    parser.add_argument('--test-notifications', help='Test notification services', action='store_true')
    parser.add_argument('--dashboard', help='Launch dashboard', action='store_true')
    parser.add_argument('--debug', help='Enable debug logging', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import after dependency check
    try:
        from crypto_predictor.main import orchestrator
        from crypto_predictor.alerts.notification_manager import notification_manager
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
    
    # Handle command line arguments
    if args.predict:
        print(f"Generating prediction for {args.predict}...")
        prediction = orchestrator.generate_manual_prediction(args.predict)
        
        if prediction:
            print(f"\n🎯 Prediction for {prediction.symbol}")
            print(f"Signal: {prediction.signal}")
            print(f"Confidence: {prediction.confidence:.1%}")
            print(f"Entry Price: ${prediction.entry_price:.2f}")
            print(f"Stop Loss: ${prediction.stop_loss:.2f}")
            print(f"Take Profit: ${prediction.take_profit:.2f}")
            print(f"Leverage: {prediction.leverage:.1f}x")
            print(f"Rationale: {prediction.rationale}")
        else:
            print("Failed to generate prediction")
    
    elif args.backtest:
        print(f"Running backtest for {args.backtest} over {args.days} days...")
        results = orchestrator.run_backtest(args.backtest, args.days)
        
        if results:
            print(f"\n📊 Backtest Results for {args.backtest}")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"Total Return: {results.get('total_return', 0):.2%}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        else:
            print("Failed to run backtest")
    
    elif args.status:
        status = orchestrator.get_system_status()
        print("\n📊 System Status")
        print(f"Running: {status['is_running']}")
        print(f"Models Loaded: {status['models_loaded']}")
        print(f"Trading Pairs: {', '.join(status['trading_pairs'])}")
        print(f"Last Prediction: {status.get('last_prediction_time', 'Never')}")
        print(f"Win Rate: {status['performance_metrics']['win_rate']:.1%}")
        print(f"Total Predictions: {status['performance_metrics']['total_predictions']}")
    
    elif args.test_notifications:
        print("Testing notification services...")
        results = notification_manager.test_notifications()
        
        print("\n🔔 Notification Test Results")
        for service, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{service.capitalize()}: {status}")
    
    elif args.dashboard:
        print("Launching dashboard...")
        try:
            import streamlit as st
            import subprocess
            subprocess.run(['streamlit', 'run', 'crypto_predictor/dashboard/app.py'])
        except ImportError:
            print("Streamlit not installed. Install with: pip install streamlit")
        except Exception as e:
            print(f"Error launching dashboard: {e}")
    
    else:
        # Default: Start the main system
        print("🚀 Starting Crypto Prediction System...")
        print("Press Ctrl+C to stop")
        
        try:
            orchestrator.start_system()
            
            # Keep running until interrupted
            while orchestrator.is_running:
                import time
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
            orchestrator.stop_system()
            print("System stopped")

if __name__ == "__main__":
    main()