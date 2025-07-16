import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for the crypto prediction system"""
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    market_cap REAL,
                    price_change_24h REAL,
                    volume_change_24h REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Technical indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    ema_10 REAL,
                    ema_50 REAL,
                    ema_200 REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    atr REAL,
                    volatility REAL,
                    volume_spike REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # News sentiment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT,
                    sentiment_score REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    prediction_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_range_min REAL NOT NULL,
                    entry_range_max REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    leverage REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    model_version TEXT,
                    lstm_prediction REAL,
                    transformer_prediction REAL,
                    sentiment_prediction REAL,
                    ensemble_prediction REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_timestamp DATETIME NOT NULL,
                    exit_timestamp DATETIME,
                    actual_entry_price REAL,
                    actual_exit_price REAL,
                    profit_loss REAL,
                    profit_loss_percentage REAL,
                    hit_stop_loss BOOLEAN DEFAULT FALSE,
                    hit_take_profit BOOLEAN DEFAULT FALSE,
                    outcome TEXT,
                    duration_hours REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            ''')
            
            # Model performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    profitable_trades INTEGER NOT NULL,
                    average_profit REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_name, date)
                )
            ''')
            
            # RL training history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rl_training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    training_date DATE NOT NULL,
                    episode INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    loss REAL,
                    epsilon REAL,
                    learning_rate REAL,
                    model_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_details TEXT
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def store_market_data(self, data: Dict[str, any]) -> bool:
        """Store market data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, price, volume, market_cap, price_change_24h, volume_change_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['timestamp'],
                    data['price'],
                    data['volume'],
                    data.get('market_cap'),
                    data.get('price_change_24h'),
                    data.get('volume_change_24h')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    def store_technical_indicators(self, symbol: str, timestamp: datetime, indicators: Dict[str, float]) -> bool:
        """Store technical indicators"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, timestamp, rsi, macd, macd_signal, macd_histogram, 
                     ema_10, ema_50, ema_200, bb_upper, bb_middle, bb_lower, 
                     atr, volatility, volume_spike)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, timestamp,
                    indicators.get('rsi'),
                    indicators.get('macd'),
                    indicators.get('macd_signal'),
                    indicators.get('macd_histogram'),
                    indicators.get('ema_10'),
                    indicators.get('ema_50'),
                    indicators.get('ema_200'),
                    indicators.get('bb_upper'),
                    indicators.get('bb_middle'),
                    indicators.get('bb_lower'),
                    indicators.get('atr'),
                    indicators.get('volatility'),
                    indicators.get('volume_spike')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing technical indicators: {e}")
            return False
    
    def store_news_sentiment(self, symbol: str, timestamp: datetime, headline: str, 
                           source: str, sentiment_score: float, sentiment_label: str, 
                           confidence: float) -> bool:
        """Store news sentiment analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO news_sentiment 
                    (symbol, timestamp, headline, source, sentiment_score, sentiment_label, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timestamp, headline, source, sentiment_score, sentiment_label, confidence))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing news sentiment: {e}")
            return False
    
    def store_prediction(self, prediction_data: Dict[str, any]) -> Optional[int]:
        """Store prediction and return prediction ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions 
                    (symbol, timestamp, prediction_type, entry_price, entry_range_min, entry_range_max,
                     stop_loss, take_profit, leverage, confidence, reasoning, model_version,
                     lstm_prediction, transformer_prediction, sentiment_prediction, ensemble_prediction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_data['symbol'],
                    prediction_data['timestamp'],
                    prediction_data['prediction_type'],
                    prediction_data['entry_price'],
                    prediction_data['entry_range_min'],
                    prediction_data['entry_range_max'],
                    prediction_data['stop_loss'],
                    prediction_data['take_profit'],
                    prediction_data['leverage'],
                    prediction_data['confidence'],
                    prediction_data['reasoning'],
                    prediction_data.get('model_version'),
                    prediction_data.get('lstm_prediction'),
                    prediction_data.get('transformer_prediction'),
                    prediction_data.get('sentiment_prediction'),
                    prediction_data.get('ensemble_prediction')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return None
    
    def update_prediction_outcome(self, prediction_id: int, outcome_data: Dict[str, any]) -> bool:
        """Update prediction outcome after 24 hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_tracking 
                    (prediction_id, symbol, entry_timestamp, exit_timestamp, 
                     actual_entry_price, actual_exit_price, profit_loss, profit_loss_percentage,
                     hit_stop_loss, hit_take_profit, outcome, duration_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id,
                    outcome_data['symbol'],
                    outcome_data['entry_timestamp'],
                    outcome_data['exit_timestamp'],
                    outcome_data['actual_entry_price'],
                    outcome_data['actual_exit_price'],
                    outcome_data['profit_loss'],
                    outcome_data['profit_loss_percentage'],
                    outcome_data['hit_stop_loss'],
                    outcome_data['hit_take_profit'],
                    outcome_data['outcome'],
                    outcome_data['duration_hours']
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
            return False
    
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data for a symbol within date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get technical indicators for a symbol within date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM technical_indicators 
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return pd.DataFrame()
    
    def get_recent_predictions(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent predictions for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                query = '''
                    SELECT * FROM predictions 
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
    
    def get_model_performance(self, symbol: str, days: int = 30) -> Dict[str, any]:
        """Get model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Get accuracy and other metrics
                query = '''
                    SELECT AVG(accuracy) as avg_accuracy,
                           AVG(win_rate) as avg_win_rate,
                           AVG(total_trades) as avg_trades,
                           AVG(average_profit) as avg_profit,
                           AVG(max_drawdown) as avg_drawdown,
                           AVG(sharpe_ratio) as avg_sharpe
                    FROM model_metrics 
                    WHERE symbol = ? AND date >= ?
                '''
                cursor = conn.cursor()
                cursor.execute(query, (symbol, cutoff_date))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'accuracy': result[0] or 0.0,
                        'win_rate': result[1] or 0.0,
                        'total_trades': result[2] or 0,
                        'average_profit': result[3] or 0.0,
                        'max_drawdown': result[4] or 0.0,
                        'sharpe_ratio': result[5] or 0.0
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to keep database size manageable"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                cursor = conn.cursor()
                
                # Keep recent data for all tables
                tables = ['market_data', 'technical_indicators', 'news_sentiment', 'system_logs']
                for table in tables:
                    cursor.execute(f'DELETE FROM {table} WHERE created_at < ?', (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def log_system_event(self, level: str, module: str, message: str, error_details: str = None):
        """Log system events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_logs (level, module, message, error_details)
                    VALUES (?, ?, ?, ?)
                ''', (level, module, message, error_details))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                tables = ['market_data', 'technical_indicators', 'news_sentiment', 
                         'predictions', 'performance_tracking', 'model_metrics']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[table] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}