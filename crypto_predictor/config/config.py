import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class SystemConfig:
    """Main system configuration"""
    
    # Trading pairs
    TRADING_PAIRS = ['BTC', 'ETH']
    
    # Data collection settings
    DATA_COLLECTION_INTERVAL = 5  # minutes
    MAX_DATA_POINTS = 10000
    
    # API endpoints and keys
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/v1"
    NEWS_API_URL = "https://newsapi.org/v2"
    
    # Model settings
    CONFIDENCE_THRESHOLD = 0.60
    LSTM_LOOKBACK_DAYS = 30
    TRANSFORMER_SEQUENCE_LENGTH = 100
    
    # Risk management
    MAX_LEVERAGE = 3.0
    DEFAULT_STOP_LOSS = 0.02  # 2%
    DEFAULT_TAKE_PROFIT = 0.04  # 4%
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    
    # Performance targets
    TARGET_WIN_RATE = 0.55
    TARGET_ACCURACY = 0.65
    
    # Reinforcement learning
    RL_UPDATE_FREQUENCY = 100  # predictions
    RL_LEARNING_RATE = 0.001
    
    # Database
    DB_PATH = "crypto_predictor.db"
    BACKUP_INTERVAL = 24  # hours
    
    # Notifications
    TELEGRAM_ENABLED = False
    EMAIL_ENABLED = False
    
    # Dashboard
    DASHBOARD_PORT = 8501
    UPDATE_INTERVAL = 300  # seconds
    
    # Backtesting
    BACKTEST_DAYS = 365
    PAPER_TRADING = True

class APIConfig:
    """API configuration management"""
    
    def __init__(self, config_file: str = "api_keys.json"):
        self.config_file = config_file
        self.api_keys = self.load_api_keys()
    
    def load_api_keys(self) -> Dict[str, str]:
        """Load API keys from file or environment variables"""
        keys = {}
        
        # Try to load from file first
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    keys = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load API keys from file: {e}")
        
        # Override with environment variables if available
        env_keys = {
            'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY'),
            'CRYPTOPANIC_API_KEY': os.getenv('CRYPTOPANIC_API_KEY'),
            'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
            'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID'),
            'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
            'EMAIL_USER': os.getenv('EMAIL_USER'),
            'EMAIL_PASS': os.getenv('EMAIL_PASS'),
            'GOOGLE_DRIVE_CREDENTIALS': os.getenv('GOOGLE_DRIVE_CREDENTIALS')
        }
        
        for key, value in env_keys.items():
            if value:
                keys[key] = value
        
        return keys
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        return self.api_keys.get(service)
    
    def save_api_keys(self, keys: Dict[str, str]):
        """Save API keys to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(keys, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save API keys to file: {e}")

class TechnicalIndicatorConfig:
    """Technical indicator configuration"""
    
    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # MACD
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # EMA periods
    EMA_PERIODS = [10, 50, 200]
    
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD = 2
    
    # ATR
    ATR_PERIOD = 14
    
    # Volume
    VOLUME_SMA_PERIOD = 20
    VOLUME_SPIKE_THRESHOLD = 2.0
    
    # Volatility
    VOLATILITY_PERIOD = 30

# Global configuration instances
system_config = SystemConfig()
api_config = APIConfig()
tech_config = TechnicalIndicatorConfig()

# Colab/Replit optimization settings
class PlatformConfig:
    """Platform-specific optimizations"""
    
    @staticmethod
    def is_colab():
        """Check if running on Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_replit():
        """Check if running on Replit"""
        return os.getenv('REPLIT_DB_URL') is not None
    
    @staticmethod
    def get_platform():
        """Get current platform"""
        if PlatformConfig.is_colab():
            return "colab"
        elif PlatformConfig.is_replit():
            return "replit"
        else:
            return "local"
    
    @staticmethod
    def optimize_for_platform():
        """Apply platform-specific optimizations"""
        platform = PlatformConfig.get_platform()
        
        if platform == "colab":
            # Colab optimizations
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            return {
                'batch_size': 32,
                'max_workers': 2,
                'memory_limit': '2GB'
            }
        elif platform == "replit":
            # Replit optimizations
            return {
                'batch_size': 16,
                'max_workers': 1,
                'memory_limit': '512MB'
            }
        else:
            # Local optimizations
            return {
                'batch_size': 64,
                'max_workers': 4,
                'memory_limit': '4GB'
            }

platform_config = PlatformConfig.optimize_for_platform()