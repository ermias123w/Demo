import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import asdict
import os
import shutil

from ..config.config import system_config
from ..models.hybrid_models import PredictionResult

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for storing predictions and performance data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or system_config.DB_PATH
        self.connection = None
        self.create_tables()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def initialize_database(self):
        """Initialize database with required tables"""
        logger.info("Initializing database...")
        self.create_tables()
        logger.info("Database initialized successfully")
    
    def create_tables(self):
        """Create database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                leverage REAL NOT NULL,
                rationale TEXT,
                model_outputs TEXT,
                technical_features TEXT,
                sentiment_score REAL,
                actual_outcome TEXT,
                profit_loss REAL,
                is_evaluated BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE NOT NULL,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_profit REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                symbol TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                market_cap REAL NOT NULL,
                price_change_24h REAL,
                volume_change_24h REAL,
                high_24h REAL,
                low_24h REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT NOT NULL,
                url TEXT,
                published_at DATETIME NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                keywords TEXT,
                relevance_score REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Training history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                training_date DATETIME NOT NULL,
                epochs INTEGER NOT NULL,
                final_loss REAL NOT NULL,
                model_path TEXT,
                performance_before TEXT,
                performance_after TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp ON predictions(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_signal ON predictions(signal)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(metric_date)")
        
        conn.commit()
        logger.info("Database tables created successfully")
    
    def store_prediction(self, prediction: PredictionResult):
        """Store a prediction in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO predictions (
                    symbol, timestamp, signal, confidence, entry_price, 
                    stop_loss, take_profit, leverage, rationale, 
                    model_outputs, technical_features, sentiment_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.symbol,
                prediction.timestamp,
                prediction.signal,
                prediction.confidence,
                prediction.entry_price,
                prediction.stop_loss,
                prediction.take_profit,
                prediction.leverage,
                prediction.rationale,
                json.dumps(prediction.model_outputs, default=str),
                json.dumps(prediction.technical_features.tolist()),
                prediction.sentiment_score
            ))
            
            conn.commit()
            logger.info(f"Stored prediction for {prediction.symbol}: {prediction.signal}")
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            conn.rollback()
    
    def get_predictions(self, 
                       symbol: str = None, 
                       days: int = 30,
                       signal: str = None) -> List[Dict]:
        """Get predictions from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM predictions 
            WHERE timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if signal:
            query += " AND signal = ?"
            params.append(signal)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            prediction_dict = dict(row)
            
            # Parse JSON fields
            if prediction_dict['model_outputs']:
                prediction_dict['model_outputs'] = json.loads(prediction_dict['model_outputs'])
            
            if prediction_dict['technical_features']:
                prediction_dict['technical_features'] = json.loads(prediction_dict['technical_features'])
            
            predictions.append(prediction_dict)
        
        return predictions
    
    def get_recent_predictions(self, days: int = 1) -> List[Dict]:
        """Get recent predictions for evaluation"""
        return self.get_predictions(days=days)
    
    def update_prediction_outcome(self, 
                                 prediction_id: int, 
                                 actual_outcome: str, 
                                 profit_loss: float):
        """Update prediction with actual outcome"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE predictions 
                SET actual_outcome = ?, profit_loss = ?, is_evaluated = TRUE
                WHERE id = ?
            """, (actual_outcome, profit_loss, prediction_id))
            
            conn.commit()
            logger.info(f"Updated prediction {prediction_id} with outcome: {actual_outcome}")
            
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
            conn.rollback()
    
    def store_performance_metrics(self, metrics: Dict):
        """Store performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO performance_metrics (
                    metric_date, total_predictions, correct_predictions, 
                    win_rate, total_profit, max_drawdown, sharpe_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date(),
                metrics.get('total_predictions', 0),
                metrics.get('correct_predictions', 0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_profit', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('sharpe_ratio', 0.0)
            ))
            
            conn.commit()
            logger.info("Stored performance metrics")
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            conn.rollback()
    
    def get_performance_metrics(self, days: int = 30) -> List[Dict]:
        """Get performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM performance_metrics 
            WHERE metric_date >= date('now', '-{} days')
            ORDER BY metric_date DESC
        """.format(days))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def store_market_data(self, market_data_list: List[Dict]):
        """Store market data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for data in market_data_list:
                cursor.execute("""
                    INSERT INTO market_data (
                        symbol, timestamp, price, volume, market_cap, 
                        price_change_24h, volume_change_24h, high_24h, low_24h
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['symbol'],
                    data['timestamp'],
                    data['price'],
                    data['volume'],
                    data['market_cap'],
                    data.get('price_change_24h', 0),
                    data.get('volume_change_24h', 0),
                    data.get('high_24h', 0),
                    data.get('low_24h', 0)
                ))
            
            conn.commit()
            logger.info(f"Stored {len(market_data_list)} market data entries")
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            conn.rollback()
    
    def get_market_data(self, 
                       symbol: str, 
                       days: int = 30) -> pd.DataFrame:
        """Get market data as DataFrame"""
        conn = self.get_connection()
        
        query = """
            SELECT * FROM market_data 
            WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def store_news_articles(self, articles: List[Dict]):
        """Store news articles"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for article in articles:
                cursor.execute("""
                    INSERT INTO news_articles (
                        title, content, source, url, published_at, 
                        sentiment_score, sentiment_label, confidence, 
                        keywords, relevance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article['title'],
                    article['content'],
                    article['source'],
                    article['url'],
                    article['published_at'],
                    article['sentiment_score'],
                    article['sentiment_label'],
                    article['confidence'],
                    json.dumps(article.get('keywords', [])),
                    article['relevance_score']
                ))
            
            conn.commit()
            logger.info(f"Stored {len(articles)} news articles")
            
        except Exception as e:
            logger.error(f"Error storing news articles: {e}")
            conn.rollback()
    
    def get_news_articles(self, days: int = 7) -> List[Dict]:
        """Get news articles"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM news_articles 
            WHERE published_at >= datetime('now', '-{} days')
            ORDER BY published_at DESC
        """.format(days))
        
        rows = cursor.fetchall()
        
        articles = []
        for row in rows:
            article_dict = dict(row)
            
            # Parse keywords
            if article_dict['keywords']:
                article_dict['keywords'] = json.loads(article_dict['keywords'])
            
            articles.append(article_dict)
        
        return articles
    
    def store_training_record(self, 
                            symbol: str, 
                            epochs: int, 
                            final_loss: float,
                            model_path: str = None):
        """Store training record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO training_history (
                    symbol, training_date, epochs, final_loss, model_path
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now(),
                epochs,
                final_loss,
                model_path
            ))
            
            conn.commit()
            logger.info(f"Stored training record for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing training record: {e}")
            conn.rollback()
    
    def get_last_training_time(self, symbol: str = None) -> Optional[datetime]:
        """Get last training time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT MAX(training_date) FROM training_history 
                WHERE symbol = ?
            """, (symbol,))
        else:
            cursor.execute("SELECT MAX(training_date) FROM training_history")
        
        result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        
        return None
    
    def get_last_prediction_time(self, symbol: str = None) -> Optional[datetime]:
        """Get last prediction time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT MAX(timestamp) FROM predictions 
                WHERE symbol = ?
            """, (symbol,))
        else:
            cursor.execute("SELECT MAX(timestamp) FROM predictions")
        
        result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        
        return None
    
    def log_system_event(self, level: str, message: str, module: str = None):
        """Log system event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO system_logs (level, message, module) 
                VALUES (?, ?, ?)
            """, (level, message, module))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def get_system_logs(self, days: int = 7, level: str = None) -> List[Dict]:
        """Get system logs"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM system_logs 
            WHERE timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        params = []
        
        if level:
            query += " AND level = ?"
            params.append(level)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to keep database size manageable"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean up old market data
            cursor.execute("""
                DELETE FROM market_data 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            # Clean up old news articles
            cursor.execute("""
                DELETE FROM news_articles 
                WHERE published_at < datetime('now', '-{} days')
            """.format(days))
            
            # Clean up old system logs
            cursor.execute("""
                DELETE FROM system_logs 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            conn.rollback()
    
    def backup_database(self, backup_path: str = None):
        """Backup database"""
        if backup_path is None:
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(
                backup_dir, 
                f"crypto_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            )
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        tables = ['predictions', 'performance_metrics', 'market_data', 'news_articles', 'system_logs', 'training_history']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[f"{table}_count"] = count
        
        # Database size
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        stats['database_size_mb'] = (page_count * page_size) / (1024 * 1024)
        
        return stats
    
    def export_data(self, table_name: str, output_path: str):
        """Export table data to CSV"""
        conn = self.get_connection()
        
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {table_name} data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()

# Create global database manager instance
db_manager = DatabaseManager()