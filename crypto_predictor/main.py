#!/usr/bin/env python3
"""
Crypto Prediction System - Main Orchestrator
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np
import schedule
import signal
import sys
import os

# Import all modules
from .config.config import system_config, api_config, platform_config
from .data_collection.market_data import market_collector
from .data_collection.sentiment_data import news_aggregator
from .data_collection.technical_indicators import technical_analyzer
from .models.hybrid_models import prediction_system, PredictionResult
from .storage.database import DatabaseManager
from .risk_management.risk_manager import RiskManager
from .backtesting.backtester import Backtester
from .alerts.notification_manager import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_predictor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CryptoPredictionOrchestrator:
    """Main orchestrator for the crypto prediction system"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.risk_manager = RiskManager()
        self.backtester = Backtester()
        self.notification_manager = NotificationManager()
        
        self.is_running = False
        self.prediction_thread = None
        self.training_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'last_updated': datetime.now()
        }
        
        # Initialize system
        self._initialize_system()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_system(self):
        """Initialize the prediction system"""
        logger.info("Initializing Crypto Prediction System...")
        
        # Initialize database
        self.db_manager.initialize_database()
        
        # Load existing models if available
        self._load_existing_models()
        
        # Set up scheduled tasks
        self._setup_scheduler()
        
        logger.info("System initialization complete")
    
    def _load_existing_models(self):
        """Load existing trained models"""
        models_dir = "models"
        if os.path.exists(models_dir):
            for symbol in system_config.TRADING_PAIRS:
                model_path = os.path.join(models_dir, f"{symbol}_model.pth")
                if os.path.exists(model_path):
                    try:
                        prediction_system.load_model(symbol, model_path)
                        logger.info(f"Loaded existing model for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading model for {symbol}: {e}")
    
    def _setup_scheduler(self):
        """Set up scheduled tasks"""
        # Data collection every 5 minutes
        schedule.every(system_config.DATA_COLLECTION_INTERVAL).minutes.do(
            self._collect_and_predict
        )
        
        # Model retraining every week
        schedule.every().week.do(self._retrain_models)
        
        # Performance evaluation every day
        schedule.every().day.at("23:59").do(self._evaluate_performance)
        
        # Backup every 6 hours
        schedule.every(6).hours.do(self._backup_system)
        
        logger.info("Scheduled tasks configured")
    
    def start_system(self):
        """Start the prediction system"""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        self.is_running = True
        logger.info("Starting Crypto Prediction System...")
        
        # Start main prediction loop
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop, 
            daemon=True
        )
        self.prediction_thread.start()
        
        # Start training loop
        self.training_thread = threading.Thread(
            target=self._training_loop, 
            daemon=True
        )
        self.training_thread.start()
        
        logger.info("System started successfully")
    
    def stop_system(self):
        """Stop the prediction system"""
        if not self.is_running:
            logger.warning("System is not running")
            return
        
        logger.info("Stopping Crypto Prediction System...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.prediction_thread:
            self.prediction_thread.join(timeout=30)
        
        if self.training_thread:
            self.training_thread.join(timeout=30)
        
        logger.info("System stopped")
    
    def _prediction_loop(self):
        """Main prediction loop"""
        logger.info("Starting prediction loop...")
        
        while self.is_running:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(60)
    
    def _training_loop(self):
        """Background training loop"""
        logger.info("Starting training loop...")
        
        while self.is_running:
            try:
                # Check if retraining is needed
                if self._should_retrain():
                    self._retrain_models()
                
                # Sleep for an hour before checking again
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                time.sleep(3600)
    
    def _collect_and_predict(self):
        """Collect data and generate predictions"""
        logger.info("Collecting data and generating predictions...")
        
        try:
            # Collect market data
            market_data = market_collector.collect_market_data(
                system_config.TRADING_PAIRS, 
                detailed=True
            )
            
            if not market_data:
                logger.warning("No market data collected")
                return
            
            # Collect news data
            news_articles = news_aggregator.collect_news(
                system_config.TRADING_PAIRS,
                limit_per_source=10
            )
            
            # Generate predictions for each symbol
            predictions = []
            for symbol in system_config.TRADING_PAIRS:
                if symbol in market_data:
                    prediction = self._generate_prediction(
                        symbol, 
                        market_data[symbol], 
                        news_articles
                    )
                    if prediction:
                        predictions.append(prediction)
            
            # Process predictions
            self._process_predictions(predictions)
            
            logger.info(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error in data collection and prediction: {e}")
    
    def _generate_prediction(self, symbol: str, market_data, news_articles) -> Optional[PredictionResult]:
        """Generate prediction for a specific symbol"""
        try:
            # Get historical data
            historical_data = market_collector.get_historical_data(
                symbol, 
                days=system_config.LSTM_LOOKBACK_DAYS + 10
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            # Calculate technical indicators
            technical_indicators = technical_analyzer.calculate_indicators(
                historical_data, 
                symbol
            )
            
            if not technical_indicators:
                logger.warning(f"No technical indicators for {symbol}")
                return None
            
            # Prepare data for prediction
            tech_features, prices, sentiment_scores, targets = prediction_system.prepare_data(
                historical_data,
                technical_indicators,
                news_articles,
                symbol
            )
            
            # Generate prediction
            prediction = prediction_system.predict(
                symbol,
                tech_features,
                prices,
                sentiment_scores,
                market_data.price
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    def _process_predictions(self, predictions: List[PredictionResult]):
        """Process and store predictions"""
        for prediction in predictions:
            try:
                # Apply risk management
                adjusted_prediction = self.risk_manager.evaluate_prediction(prediction)
                
                # Store prediction
                self.db_manager.store_prediction(adjusted_prediction)
                
                # Send notifications if enabled
                if adjusted_prediction.signal != 'HOLD':
                    self.notification_manager.send_prediction_alert(adjusted_prediction)
                
                # Update performance metrics
                self._update_performance_metrics(adjusted_prediction)
                
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
    
    def _retrain_models(self):
        """Retrain all models"""
        logger.info("Starting model retraining...")
        
        try:
            for symbol in system_config.TRADING_PAIRS:
                self._retrain_model(symbol)
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    def _retrain_model(self, symbol: str):
        """Retrain model for a specific symbol"""
        try:
            # Get training data
            historical_data = market_collector.get_historical_data(
                symbol, 
                days=system_config.BACKTEST_DAYS
            )
            
            if len(historical_data) < 100:
                logger.warning(f"Insufficient training data for {symbol}")
                return
            
            # Calculate technical indicators
            technical_indicators = technical_analyzer.calculate_indicators(
                historical_data, 
                symbol
            )
            
            # Get news data (recent)
            news_articles = news_aggregator.collect_news([symbol], limit_per_source=100)
            
            # Prepare training data
            tech_features, prices, sentiment_scores, targets = prediction_system.prepare_data(
                historical_data,
                technical_indicators,
                news_articles,
                symbol
            )
            
            # Train model
            prediction_system.train_model(
                symbol,
                tech_features,
                prices,
                sentiment_scores,
                targets,
                epochs=50,
                batch_size=platform_config.get('batch_size', 32)
            )
            
            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{symbol}_model.pth")
            prediction_system.save_model(symbol, model_path)
            
            logger.info(f"Model retrained and saved for {symbol}")
            
        except Exception as e:
            logger.error(f"Error retraining model for {symbol}: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if model retraining is needed"""
        # Check performance degradation
        if self.performance_metrics['win_rate'] < system_config.TARGET_WIN_RATE * 0.8:
            return True
        
        # Check time since last training
        last_training = self.db_manager.get_last_training_time()
        if last_training and (datetime.now() - last_training).days >= 7:
            return True
        
        return False
    
    def _evaluate_performance(self):
        """Evaluate system performance"""
        logger.info("Evaluating system performance...")
        
        try:
            # Get recent predictions
            predictions = self.db_manager.get_recent_predictions(days=1)
            
            if not predictions:
                logger.warning("No recent predictions to evaluate")
                return
            
            # Evaluate predictions
            evaluation_results = self.backtester.evaluate_predictions(predictions)
            
            # Update performance metrics
            self.performance_metrics.update(evaluation_results)
            self.performance_metrics['last_updated'] = datetime.now()
            
            # Store performance metrics
            self.db_manager.store_performance_metrics(self.performance_metrics)
            
            # Check if performance is below target
            if self.performance_metrics['win_rate'] < system_config.TARGET_WIN_RATE:
                logger.warning(f"Performance below target: {self.performance_metrics['win_rate']:.2%}")
                
                # Send alert
                self.notification_manager.send_performance_alert(self.performance_metrics)
            
            logger.info(f"Performance evaluation completed: {evaluation_results}")
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
    
    def _update_performance_metrics(self, prediction: PredictionResult):
        """Update performance metrics"""
        self.performance_metrics['total_predictions'] += 1
        # Other metrics will be updated during performance evaluation
    
    def _backup_system(self):
        """Backup system data"""
        logger.info("Backing up system data...")
        
        try:
            # Backup database
            self.db_manager.backup_database()
            
            # Backup models
            self._backup_models()
            
            logger.info("System backup completed")
            
        except Exception as e:
            logger.error(f"Error during system backup: {e}")
    
    def _backup_models(self):
        """Backup trained models"""
        backup_dir = f"backups/models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        models_dir = "models"
        if os.path.exists(models_dir):
            import shutil
            shutil.copytree(models_dir, os.path.join(backup_dir, "models"))
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'performance_metrics': self.performance_metrics,
            'last_prediction_time': self.db_manager.get_last_prediction_time(),
            'models_loaded': len(prediction_system.models),
            'trading_pairs': system_config.TRADING_PAIRS,
            'platform': platform_config
        }
    
    def generate_manual_prediction(self, symbol: str) -> Optional[PredictionResult]:
        """Generate a manual prediction for a symbol"""
        logger.info(f"Generating manual prediction for {symbol}")
        
        try:
            # Get current market data
            market_data = market_collector.collect_market_data([symbol], detailed=True)
            
            if symbol not in market_data:
                logger.error(f"No market data available for {symbol}")
                return None
            
            # Get news data
            news_articles = news_aggregator.collect_news([symbol], limit_per_source=5)
            
            # Generate prediction
            prediction = self._generate_prediction(symbol, market_data[symbol], news_articles)
            
            if prediction:
                # Apply risk management
                adjusted_prediction = self.risk_manager.evaluate_prediction(prediction)
                return adjusted_prediction
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating manual prediction: {e}")
            return None
    
    def run_backtest(self, symbol: str, days: int = 30) -> Dict:
        """Run backtest for a symbol"""
        logger.info(f"Running backtest for {symbol} over {days} days")
        
        try:
            # Get historical data
            historical_data = market_collector.get_historical_data(symbol, days=days)
            
            if historical_data.empty:
                logger.error(f"No historical data for {symbol}")
                return {}
            
            # Run backtest
            results = self.backtester.run_backtest(symbol, historical_data)
            
            logger.info(f"Backtest completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}

# Global orchestrator instance
orchestrator = CryptoPredictionOrchestrator()

def main():
    """Main entry point"""
    logger.info("Starting Crypto Prediction System...")
    
    try:
        # Start the system
        orchestrator.start_system()
        
        # Keep the main thread alive
        while orchestrator.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
    finally:
        orchestrator.stop_system()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()