import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    """Configuration for data collection"""
    # Data Collection
    UPDATE_INTERVAL: int = 300  # 5 minutes in seconds
    SYMBOLS: List[str] = None
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    CRYPTOPANIC_API_URL: str = "https://cryptopanic.com/api/v1"
    NEWSAPI_URL: str = "https://newsapi.org/v2/everything"
    
    # Database
    DB_PATH: str = "data/crypto_data.db"
    BACKUP_PATH: str = "data/backup/"
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["bitcoin", "ethereum"]

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # Model Parameters
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    
    TRANSFORMER_D_MODEL: int = 512
    TRANSFORMER_NHEAD: int = 8
    TRANSFORMER_NUM_LAYERS: int = 6
    TRANSFORMER_DROPOUT: float = 0.1
    
    # Training Parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Sequence Length
    SEQUENCE_LENGTH: int = 60  # 60 time steps for LSTM
    
    # Model Paths
    MODEL_SAVE_PATH: str = "models_checkpoints/"
    
    # Confidence Threshold
    MIN_CONFIDENCE: float = 0.6

@dataclass
class TechnicalConfig:
    """Configuration for technical analysis"""
    # Technical Indicators
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    EMA_PERIODS: List[int] = None
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    ATR_PERIOD: int = 14
    VOLUME_SPIKE_THRESHOLD: float = 2.0
    
    def __post_init__(self):
        if self.EMA_PERIODS is None:
            self.EMA_PERIODS = [10, 50, 200]

@dataclass
class RiskConfig:
    """Configuration for risk management"""
    # Risk Parameters
    MAX_LEVERAGE: float = 5.0
    MIN_LEVERAGE: float = 1.0
    BASE_STOP_LOSS: float = 0.02  # 2%
    BASE_TAKE_PROFIT: float = 0.04  # 4%
    
    # Volatility Adjustment
    VOLATILITY_MULTIPLIER: float = 1.5
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    
    # Confidence Scaling
    CONFIDENCE_SCALING: bool = True
    MIN_CONFIDENCE_FOR_LEVERAGE: float = 0.8

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Backtesting Parameters
    INITIAL_CAPITAL: float = 10000.0
    COMMISSION: float = 0.001  # 0.1%
    SLIPPAGE: float = 0.0005  # 0.05%
    
    # Performance Metrics
    MIN_WIN_RATE: float = 0.55
    MIN_ACCURACY: float = 0.60
    DISABLE_THRESHOLD: float = 0.45
    
    # Backtesting Period
    BACKTEST_DAYS: int = 30

@dataclass
class ReinforcementConfig:
    """Configuration for reinforcement learning"""
    # RL Parameters
    ALGORITHM: str = "PPO"  # PPO or DQN
    LEARNING_RATE: float = 0.0003
    BUFFER_SIZE: int = 10000
    
    # Reward Function
    PROFIT_REWARD_MULTIPLIER: float = 1.0
    LOSS_PENALTY_MULTIPLIER: float = -1.0
    CONFIDENCE_REWARD_MULTIPLIER: float = 0.1
    
    # Training Schedule
    RETRAIN_INTERVAL_DAYS: int = 7
    MIN_SAMPLES_FOR_TRAINING: int = 100

@dataclass
class DashboardConfig:
    """Configuration for dashboard"""
    # Streamlit Settings
    PAGE_TITLE: str = "Crypto Prediction System"
    PAGE_ICON: str = "📈"
    LAYOUT: str = "wide"
    
    # Refresh Settings
    AUTO_REFRESH: bool = True
    REFRESH_INTERVAL: int = 60  # seconds

@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    # Email (optional)
    EMAIL_ENABLED: bool = False
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587

class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.technical = TechnicalConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.reinforcement = ReinforcementConfig()
        self.dashboard = DashboardConfig()
        self.notification = NotificationConfig()
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        # API Keys
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
        self.CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
        self.NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
        
        # Telegram
        self.notification.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.notification.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        
        # Google Drive (optional)
        self.GOOGLE_DRIVE_ENABLED = os.getenv("GOOGLE_DRIVE_ENABLED", "false").lower() == "true"
        self.GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "")

# Global configuration instance
config = Config()