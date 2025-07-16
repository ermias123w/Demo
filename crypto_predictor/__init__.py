"""
Crypto Prediction System

A free, mobile-optimized, self-learning cryptocurrency prediction system
that generates high-confidence trading signals using AI models and sentiment analysis.
"""

__version__ = "1.0.0"
__author__ = "Crypto Prediction Team"
__email__ = "support@cryptopredictor.com"

from .main import orchestrator
from .models.hybrid_models import prediction_system
from .data_collection.market_data import market_collector
from .data_collection.sentiment_data import news_aggregator
from .data_collection.technical_indicators import technical_analyzer
from .risk_management.risk_manager import risk_manager
from .backtesting.backtester import backtester
from .storage.database import db_manager
from .alerts.notification_manager import notification_manager

__all__ = [
    'orchestrator',
    'prediction_system',
    'market_collector',
    'news_aggregator',
    'technical_analyzer',
    'risk_manager',
    'backtester',
    'db_manager',
    'notification_manager'
]