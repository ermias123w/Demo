#!/usr/bin/env python3
"""
Crypto Prediction System
A comprehensive AI-powered cryptocurrency prediction system that generates daily trading signals
for Bitcoin and Ethereum using LSTM, Transformer, and sentiment analysis.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import schedule
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from config.config import config
from src.data_collection.database import DatabaseManager
from src.data_collection.market_data_collector import MarketDataCollector
from src.data_collection.sentiment_analyzer import SentimentAnalyzer
from src.technical_analysis.indicators import TechnicalAnalyzer
from src.models.prediction_models import CryptoPredictionSystem
from src.risk_management.risk_manager import RiskManager
from src.backtesting.backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crypto_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoPredictionApp:
    """Main application class that orchestrates the entire system"""
    
    def __init__(self):
        # Initialize components
        self.db_manager = DatabaseManager(config.data.DB_PATH)
        self.market_collector = MarketDataCollector(
            config.COINGECKO_API_KEY,
            config.NEWSAPI_KEY,
            config.CRYPTOPANIC_API_KEY
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager(config.risk.__dict__)
        self.backtester = Backtester(config.backtest.__dict__)
        
        # Initialize prediction systems for each symbol
        self.prediction_systems = {}
        for symbol in config.data.SYMBOLS:
            self.prediction_systems[symbol] = CryptoPredictionSystem(config.model.__dict__)
        
        # System state
        self.last_update = None
        self.system_running = False
        self.paper_trading_mode = True
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'profitable_trades': 0,
            'total_trades': 0,
            'current_accuracy': 0.0,
            'current_win_rate': 0.0
        }
        
        logger.info("CryptoPredictionApp initialized successfully")
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect all required data: market data, news, and sentiment"""
        try:
            logger.info("Starting data collection...")
            
            # Collect market data and news in parallel
            market_data = self.market_collector.collect_current_market_data()
            news_data = self.market_collector.collect_news_data()
            market_overview = self.market_collector.get_market_overview()
            
            # Store market data in database
            for symbol, data in market_data.items():
                self.db_manager.store_market_data({
                    'symbol': symbol,
                    'timestamp': data.timestamp,
                    'price': data.price,
                    'volume': data.volume,
                    'market_cap': data.market_cap,
                    'price_change_24h': data.price_change_24h,
                    'volume_change_24h': data.volume_change_24h
                })
            
            # Analyze news sentiment
            all_sentiment_results = []
            for symbol in config.data.SYMBOLS:
                sentiment_results = self.sentiment_analyzer.analyze_news_batch(news_data, symbol)
                all_sentiment_results.extend(sentiment_results)
                
                # Store sentiment data
                for result in sentiment_results:
                    self.db_manager.store_news_sentiment(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        headline=result['title'],
                        source=result['source'],
                        sentiment_score=result['sentiment_score'],
                        sentiment_label=result['sentiment_label'],
                        confidence=result['confidence']
                    )
            
            logger.info(f"Data collection completed. Market data: {len(market_data)} symbols, "
                       f"News articles: {len(news_data)}, Sentiment results: {len(all_sentiment_results)}")
            
            return {
                'market_data': market_data,
                'news_data': news_data,
                'sentiment_results': all_sentiment_results,
                'market_overview': market_overview
            }
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return {}
    
    def prepare_prediction_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare data for prediction models"""
        try:
            # Get historical data from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Get 90 days of data
            
            # Get market data
            market_df = self.db_manager.get_market_data(symbol, start_date, end_date)
            if market_df.empty:
                logger.warning(f"No market data found for {symbol}")
                return None
            
            # Get technical indicators
            tech_df = self.db_manager.get_technical_indicators(symbol, start_date, end_date)
            
            # Calculate technical indicators if not in database
            if tech_df.empty:
                logger.info(f"Calculating technical indicators for {symbol}")
                market_df = self.technical_analyzer.calculate_all_indicators(market_df)
                
                # Store calculated indicators
                for _, row in market_df.iterrows():
                    if pd.notna(row.get('rsi')):  # Only store if indicators are calculated
                        indicators = self.technical_analyzer.get_latest_indicators(
                            market_df[market_df.index <= row.name]
                        )
                        self.db_manager.store_technical_indicators(
                            symbol, row['timestamp'], indicators
                        )
            else:
                # Merge technical indicators with market data
                market_df = market_df.merge(tech_df, on=['symbol', 'timestamp'], how='left')
            
            # Get sentiment data
            sentiment_df = self.db_manager.get_market_data(symbol, start_date, end_date)  # This needs to be updated
            # For now, add a simple sentiment score
            market_df['sentiment_score'] = 0.0  # Placeholder
            
            # Calculate aggregated sentiment for recent periods
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Get recent sentiment from news
            try:
                recent_news = [
                    result for result in self.db_manager.get_recent_predictions(symbol, 24)
                    if datetime.fromisoformat(result.get('timestamp', '')) >= cutoff_time
                ]
                
                if recent_news:
                    aggregated_sentiment = self.sentiment_analyzer.calculate_aggregated_sentiment(recent_news)
                    market_df['sentiment_score'] = aggregated_sentiment.get('weighted_sentiment', 0.0)
            except Exception as e:
                logger.warning(f"Error calculating sentiment for {symbol}: {e}")
                market_df['sentiment_score'] = 0.0
            
            # Fill missing values
            market_df = market_df.fillna(method='ffill').fillna(method='bfill')
            
            return market_df
            
        except Exception as e:
            logger.error(f"Error preparing prediction data for {symbol}: {e}")
            return None
    
    def generate_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate prediction for a symbol"""
        try:
            # Get prediction system for symbol
            prediction_system = self.prediction_systems.get(symbol)
            if not prediction_system:
                logger.error(f"No prediction system found for {symbol}")
                return None
            
            # Prepare data
            df = self.prepare_prediction_data(symbol)
            if df is None or df.empty:
                logger.warning(f"No data available for prediction for {symbol}")
                return None
            
            # Check if we have enough data
            if len(df) < config.model.SEQUENCE_LENGTH:
                logger.warning(f"Not enough data for prediction for {symbol}. "
                              f"Required: {config.model.SEQUENCE_LENGTH}, Available: {len(df)}")
                return None
            
            # Prepare data for models
            sequences, sentiment_data, targets = prediction_system.prepare_data(df)
            
            if sequences is None:
                logger.error(f"Failed to prepare data for prediction for {symbol}")
                return None
            
            # Check if models are trained
            if prediction_system.ensemble_model is None:
                logger.info(f"Models not trained for {symbol}. Training now...")
                
                # Build models
                input_size = sequences.shape[2]
                prediction_system.build_models(input_size)
                
                # Train models
                training_results = prediction_system.train_models(sequences, sentiment_data, targets)
                
                if not training_results:
                    logger.error(f"Failed to train models for {symbol}")
                    return None
                
                logger.info(f"Models trained for {symbol}. Validation accuracy: "
                           f"{training_results.get('final_val_accuracy', 0):.2f}%")
            
            # Make prediction on latest data
            latest_sequence = sequences[-1:]  # Get the last sequence
            latest_sentiment = sentiment_data[-1:]  # Get the last sentiment
            
            predictions = prediction_system.predict(latest_sequence, latest_sentiment)
            
            if not predictions:
                logger.error(f"Failed to generate prediction for {symbol}")
                return None
            
            # Get current market data
            current_price = df.iloc[-1]['price']
            
            # Get technical signals
            technical_signals = self.technical_analyzer.generate_signals(df.tail(100))
            
            # Combine predictions
            ensemble_prediction = predictions['ensemble_prediction'][0]
            ensemble_confidence = predictions['ensemble_confidence'][0]
            
            # Apply confidence threshold
            if ensemble_confidence < config.model.MIN_CONFIDENCE:
                ensemble_prediction = 'HOLD'
                logger.info(f"Low confidence prediction for {symbol}. Defaulting to HOLD.")
            
            # Calculate entry price range and risk parameters
            risk_params = self.risk_manager.calculate_risk_parameters(
                current_price, ensemble_confidence, df.tail(20)
            )
            
            # Generate reasoning
            reasoning_parts = []
            reasoning_parts.append(f"Ensemble: {ensemble_prediction}")
            reasoning_parts.append(f"Confidence: {ensemble_confidence:.2f}")
            reasoning_parts.append(f"Technical: {technical_signals.get('signal', 'HOLD')}")
            
            if technical_signals.get('reason'):
                reasoning_parts.append(technical_signals['reason'])
            
            reasoning = ' | '.join(reasoning_parts)
            
            # Create prediction result
            prediction_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'prediction_type': ensemble_prediction,
                'entry_price': current_price,
                'entry_range_min': risk_params['entry_range_min'],
                'entry_range_max': risk_params['entry_range_max'],
                'stop_loss': risk_params['stop_loss'],
                'take_profit': risk_params['take_profit'],
                'leverage': risk_params['leverage'],
                'confidence': ensemble_confidence,
                'reasoning': reasoning,
                'model_version': 'v1.0',
                'lstm_prediction': predictions['lstm_prediction'][0],
                'transformer_prediction': predictions['transformer_prediction'][0],
                'sentiment_prediction': predictions['sentiment_prediction'][0],
                'ensemble_prediction': ensemble_prediction
            }
            
            # Store prediction in database
            prediction_id = self.db_manager.store_prediction(prediction_result)
            prediction_result['prediction_id'] = prediction_id
            
            logger.info(f"Generated prediction for {symbol}: {ensemble_prediction} "
                       f"(Confidence: {ensemble_confidence:.2f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    def update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            for symbol in config.data.SYMBOLS:
                metrics = self.db_manager.get_model_performance(symbol)
                
                if metrics:
                    self.performance_metrics['current_accuracy'] = metrics.get('accuracy', 0.0)
                    self.performance_metrics['current_win_rate'] = metrics.get('win_rate', 0.0)
                    self.performance_metrics['total_trades'] = metrics.get('total_trades', 0)
                    
                    # Calculate other metrics
                    if self.performance_metrics['total_trades'] > 0:
                        self.performance_metrics['profitable_trades'] = int(
                            self.performance_metrics['current_win_rate'] * 
                            self.performance_metrics['total_trades']
                        )
            
            logger.info(f"Performance metrics updated. Accuracy: "
                       f"{self.performance_metrics['current_accuracy']:.2f}, "
                       f"Win rate: {self.performance_metrics['current_win_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def run_prediction_cycle(self):
        """Run a complete prediction cycle"""
        try:
            logger.info("Starting prediction cycle...")
            
            # Collect all data
            data = await self.collect_all_data()
            
            if not data:
                logger.error("Failed to collect data. Skipping prediction cycle.")
                return
            
            # Generate predictions for each symbol
            predictions = {}
            for symbol in config.data.SYMBOLS:
                prediction = self.generate_prediction(symbol)
                if prediction:
                    predictions[symbol] = prediction
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Log results
            logger.info(f"Prediction cycle completed. Generated {len(predictions)} predictions.")
            
            for symbol, prediction in predictions.items():
                logger.info(f"{symbol}: {prediction['prediction_type']} "
                           f"(Confidence: {prediction['confidence']:.2f}) - "
                           f"{prediction['reasoning']}")
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
    
    def run_backtesting(self, symbol: str, days: int = 30):
        """Run backtesting for a symbol"""
        try:
            logger.info(f"Starting backtesting for {symbol} ({days} days)")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.db_manager.get_market_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data available for backtesting {symbol}")
                return
            
            # Get predictions from database
            predictions_df = self.db_manager.get_recent_predictions(symbol, days * 24)
            
            if predictions_df.empty:
                logger.warning(f"No predictions available for backtesting {symbol}")
                return
            
            # Run backtesting
            backtest_results = self.backtester.run_backtest(df, predictions_df)
            
            if backtest_results:
                logger.info(f"Backtesting completed for {symbol}. "
                           f"Total return: {backtest_results.get('total_return', 0):.2f}%, "
                           f"Win rate: {backtest_results.get('win_rate', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"Error in backtesting for {symbol}: {e}")
    
    def start_system(self):
        """Start the prediction system"""
        try:
            logger.info("Starting Crypto Prediction System...")
            
            # Create necessary directories
            Path('logs').mkdir(exist_ok=True)
            Path('data').mkdir(exist_ok=True)
            Path('models_checkpoints').mkdir(exist_ok=True)
            
            # Initialize database
            self.db_manager.init_database()
            
            # Schedule tasks
            schedule.every(config.data.UPDATE_INTERVAL).seconds.do(
                lambda: asyncio.run(self.run_prediction_cycle())
            )
            
            # Schedule daily backtesting
            schedule.every().day.at("00:00").do(
                lambda: [self.run_backtesting(symbol) for symbol in config.data.SYMBOLS]
            )
            
            # Schedule weekly model retraining
            schedule.every().sunday.at("02:00").do(self.retrain_models)
            
            # Schedule database cleanup
            schedule.every().week.do(self.db_manager.cleanup_old_data)
            
            self.system_running = True
            
            # Run initial prediction cycle
            asyncio.run(self.run_prediction_cycle())
            
            logger.info("Crypto Prediction System started successfully!")
            
            # Main loop
            while self.system_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal. Stopping system...")
            self.stop_system()
        except Exception as e:
            logger.error(f"Error in system startup: {e}")
            self.stop_system()
    
    def stop_system(self):
        """Stop the prediction system"""
        try:
            logger.info("Stopping Crypto Prediction System...")
            self.system_running = False
            
            # Save final state
            self.save_system_state()
            
            logger.info("System stopped successfully.")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def retrain_models(self):
        """Retrain all prediction models"""
        try:
            logger.info("Starting model retraining...")
            
            for symbol in config.data.SYMBOLS:
                try:
                    # Get training data
                    df = self.prepare_prediction_data(symbol)
                    if df is None or len(df) < config.model.SEQUENCE_LENGTH:
                        continue
                    
                    # Prepare data
                    prediction_system = self.prediction_systems[symbol]
                    sequences, sentiment_data, targets = prediction_system.prepare_data(df)
                    
                    if sequences is None:
                        continue
                    
                    # Retrain models
                    input_size = sequences.shape[2]
                    prediction_system.build_models(input_size)
                    
                    training_results = prediction_system.train_models(sequences, sentiment_data, targets)
                    
                    if training_results:
                        logger.info(f"Retrained models for {symbol}. "
                                   f"Validation accuracy: {training_results.get('final_val_accuracy', 0):.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error retraining models for {symbol}: {e}")
                    continue
            
            logger.info("Model retraining completed.")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
    
    def save_system_state(self):
        """Save current system state"""
        try:
            state = {
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'performance_metrics': self.performance_metrics,
                'system_running': self.system_running,
                'paper_trading_mode': self.paper_trading_mode
            }
            
            with open('data/system_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info("System state saved.")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self):
        """Load saved system state"""
        try:
            if os.path.exists('data/system_state.json'):
                with open('data/system_state.json', 'r') as f:
                    state = json.load(f)
                
                self.last_update = datetime.fromisoformat(state['last_update']) if state['last_update'] else None
                self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
                self.paper_trading_mode = state.get('paper_trading_mode', True)
                
                logger.info("System state loaded.")
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            db_stats = self.db_manager.get_database_stats()
            
            status = {
                'system_running': self.system_running,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'performance_metrics': self.performance_metrics,
                'database_stats': db_stats,
                'paper_trading_mode': self.paper_trading_mode,
                'supported_symbols': config.data.SYMBOLS,
                'update_interval': config.data.UPDATE_INTERVAL
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}

def main():
    """Main entry point"""
    try:
        print("=" * 60)
        print("🚀 Crypto Prediction System v1.0")
        print("=" * 60)
        print("AI-Powered Bitcoin & Ethereum Trading Signals")
        print("Using LSTM + Transformer + Sentiment Analysis")
        print("=" * 60)
        
        # Initialize and start the application
        app = CryptoPredictionApp()
        app.load_system_state()
        app.start_system()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()