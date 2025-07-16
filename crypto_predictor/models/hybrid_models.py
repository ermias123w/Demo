import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pickle
import os
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ..config.config import system_config, platform_config
from ..data_collection.technical_indicators import TechnicalIndicators
from ..data_collection.sentiment_data import NewsArticle

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result structure"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    rationale: str
    model_outputs: Dict[str, float]
    technical_features: np.ndarray
    sentiment_score: float

class CryptoDataset(Dataset):
    """Dataset for cryptocurrency time series prediction"""
    
    def __init__(self, 
                 price_data: np.ndarray,
                 technical_features: np.ndarray,
                 sentiment_data: np.ndarray,
                 targets: np.ndarray,
                 sequence_length: int = 30):
        
        self.price_data = price_data
        self.technical_features = technical_features
        self.sentiment_data = sentiment_data
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences for time series prediction"""
        sequences = []
        
        for i in range(len(self.price_data) - self.sequence_length):
            # Price sequence
            price_seq = self.price_data[i:i + self.sequence_length]
            
            # Technical features sequence
            tech_seq = self.technical_features[i:i + self.sequence_length]
            
            # Sentiment sequence (aggregate over sequence)
            sentiment_seq = self.sentiment_data[i:i + self.sequence_length]
            
            # Target (next period prediction)
            target = self.targets[i + self.sequence_length]
            
            sequences.append({
                'price': torch.FloatTensor(price_seq),
                'technical': torch.FloatTensor(tech_seq),
                'sentiment': torch.FloatTensor(sentiment_seq),
                'target': torch.LongTensor([target])
            })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class LSTMPredictor(nn.Module):
    """LSTM model for short-term time series prediction"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 num_classes: int = 3):
        
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output
        last_output = attn_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_output)
        
        # Confidence
        confidence = self.confidence_head(last_output)
        
        return logits, confidence

class TransformerPredictor(nn.Module):
    """Transformer model for long-range pattern detection"""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 num_classes: int = 3):
        
        super(TransformerPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Confidence
        confidence = self.confidence_head(pooled)
        
        return logits, confidence

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)].transpose(0, 1)
        return self.dropout(x)

class SentimentIntegrator(nn.Module):
    """Integrates sentiment analysis with technical indicators"""
    
    def __init__(self, 
                 technical_size: int,
                 sentiment_size: int = 1,
                 hidden_size: int = 128,
                 num_classes: int = 3):
        
        super(SentimentIntegrator, self).__init__()
        
        # Technical features processing
        self.technical_encoder = nn.Sequential(
            nn.Linear(technical_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Sentiment processing
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, technical_features, sentiment_features):
        # Encode technical features
        tech_encoded = self.technical_encoder(technical_features)
        
        # Encode sentiment features
        sentiment_encoded = self.sentiment_encoder(sentiment_features)
        
        # Concatenate features
        combined = torch.cat([tech_encoded, sentiment_encoded], dim=-1)
        
        # Final prediction
        logits = self.fusion(combined)
        confidence = self.confidence_head(combined)
        
        return logits, confidence

class EnsemblePredictor(nn.Module):
    """Ensemble model combining LSTM, Transformer, and Sentiment models"""
    
    def __init__(self,
                 technical_input_size: int,
                 sequence_length: int = 30,
                 num_classes: int = 3):
        
        super(EnsemblePredictor, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Individual models
        self.lstm_model = LSTMPredictor(
            input_size=technical_input_size,
            hidden_size=platform_config.get('batch_size', 64),
            num_classes=num_classes
        )
        
        self.transformer_model = TransformerPredictor(
            input_size=technical_input_size,
            d_model=128,
            num_classes=num_classes
        )
        
        self.sentiment_model = SentimentIntegrator(
            technical_size=technical_input_size,
            num_classes=num_classes
        )
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Meta-learner for final prediction
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 3 + 3, 128),  # 3 model outputs + 3 confidences
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Final confidence prediction
        self.final_confidence = nn.Sequential(
            nn.Linear(num_classes * 3 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, price_sequence, technical_sequence, sentiment_sequence):
        # LSTM prediction
        lstm_logits, lstm_conf = self.lstm_model(technical_sequence)
        
        # Transformer prediction
        transformer_logits, transformer_conf = self.transformer_model(technical_sequence)
        
        # Sentiment-integrated prediction (use last technical features)
        last_technical = technical_sequence[:, -1, :]
        last_sentiment = sentiment_sequence[:, -1:] if sentiment_sequence.dim() > 1 else sentiment_sequence.unsqueeze(1)
        sentiment_logits, sentiment_conf = self.sentiment_model(last_technical, last_sentiment)
        
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted ensemble
        ensemble_logits = (weights[0] * lstm_logits + 
                          weights[1] * transformer_logits + 
                          weights[2] * sentiment_logits)
        
        # Meta-learning input
        meta_input = torch.cat([
            lstm_logits, transformer_logits, sentiment_logits,
            lstm_conf, transformer_conf, sentiment_conf
        ], dim=-1)
        
        # Final prediction
        final_logits = self.meta_learner(meta_input)
        final_confidence = self.final_confidence(meta_input)
        
        return final_logits, final_confidence, {
            'lstm_logits': lstm_logits,
            'transformer_logits': transformer_logits,
            'sentiment_logits': sentiment_logits,
            'lstm_conf': lstm_conf,
            'transformer_conf': transformer_conf,
            'sentiment_conf': sentiment_conf,
            'ensemble_weights': weights
        }

class CryptoPredictionSystem:
    """Main prediction system orchestrator"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.feature_size = 29  # Based on technical indicators
        
        # Initialize models for each trading pair
        for symbol in system_config.TRADING_PAIRS:
            self.models[symbol] = EnsemblePredictor(
                technical_input_size=self.feature_size,
                sequence_length=system_config.LSTM_LOOKBACK_DAYS
            ).to(self.device)
            
            self.scalers[symbol] = {
                'technical': StandardScaler(),
                'price': MinMaxScaler(),
                'sentiment': StandardScaler()
            }
    
    def prepare_data(self, 
                     historical_data: pd.DataFrame,
                     technical_indicators: List[TechnicalIndicators],
                     news_articles: List[NewsArticle],
                     symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training/prediction"""
        
        # Extract features
        technical_features = []
        sentiment_scores = []
        prices = []
        
        for i, indicator in enumerate(technical_indicators):
            # Technical features
            feature_vector = self._extract_technical_features(indicator)
            technical_features.append(feature_vector)
            
            # Price
            prices.append(indicator.price)
            
            # Sentiment (aggregate news sentiment for this timestamp)
            sentiment_score = self._aggregate_sentiment(news_articles, indicator.timestamp)
            sentiment_scores.append(sentiment_score)
        
        # Convert to numpy arrays
        technical_features = np.array(technical_features)
        sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)
        prices = np.array(prices).reshape(-1, 1)
        
        # Create targets (price movement: 0=down, 1=stable, 2=up)
        targets = self._create_targets(prices)
        
        # Scale features
        if len(technical_features) > 0:
            technical_features = self.scalers[symbol]['technical'].fit_transform(technical_features)
            prices = self.scalers[symbol]['price'].fit_transform(prices)
            sentiment_scores = self.scalers[symbol]['sentiment'].fit_transform(sentiment_scores)
        
        return technical_features, prices, sentiment_scores, targets
    
    def _extract_technical_features(self, indicator: TechnicalIndicators) -> np.ndarray:
        """Extract technical features from indicator"""
        features = [
            indicator.price,
            indicator.sma_20,
            indicator.ema_10,
            indicator.ema_50,
            indicator.ema_200,
            indicator.rsi / 100.0,
            indicator.stoch_k / 100.0,
            indicator.stoch_d / 100.0,
            indicator.williams_r / -100.0,
            indicator.macd,
            indicator.macd_signal,
            indicator.macd_histogram,
            indicator.bb_percent,
            indicator.bb_width,
            indicator.volume_ratio,
            indicator.mfi / 100.0,
            indicator.atr,
            indicator.volatility,
            indicator.adx / 100.0,
            indicator.cci / 100.0,
            indicator.price_momentum,
            indicator.volume_momentum,
            indicator.volatility_ratio,
            indicator.ma_crossover_signal,
            indicator.rsi_signal,
            indicator.macd_signal_cross,
            indicator.bb_signal,
            indicator.volume_signal,
            indicator.support_level / indicator.price if indicator.price > 0 else 0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _aggregate_sentiment(self, news_articles: List[NewsArticle], timestamp: datetime) -> float:
        """Aggregate sentiment for a specific timestamp"""
        if not news_articles:
            return 0.0
        
        # Find articles within 1 hour of timestamp
        relevant_articles = [
            article for article in news_articles
            if abs((article.published_at - timestamp).total_seconds()) < 3600
        ]
        
        if not relevant_articles:
            return 0.0
        
        # Weighted average by relevance
        total_weight = sum(article.relevance_score for article in relevant_articles)
        
        if total_weight == 0:
            return 0.0
        
        weighted_sentiment = sum(
            article.sentiment_score * article.relevance_score 
            for article in relevant_articles
        ) / total_weight
        
        return weighted_sentiment
    
    def _create_targets(self, prices: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Create target labels based on price movement"""
        targets = []
        
        for i in range(len(prices) - 1):
            current_price = prices[i][0]
            next_price = prices[i + 1][0]
            
            price_change = (next_price - current_price) / current_price
            
            if price_change > threshold:
                targets.append(2)  # UP
            elif price_change < -threshold:
                targets.append(0)  # DOWN
            else:
                targets.append(1)  # STABLE
        
        # Last target (no future data)
        targets.append(1)
        
        return np.array(targets)
    
    def train_model(self, 
                    symbol: str,
                    technical_features: np.ndarray,
                    prices: np.ndarray,
                    sentiment_scores: np.ndarray,
                    targets: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 32):
        """Train the ensemble model"""
        
        # Create dataset
        dataset = CryptoDataset(
            price_data=prices,
            technical_features=technical_features,
            sentiment_data=sentiment_scores,
            targets=targets,
            sequence_length=system_config.LSTM_LOOKBACK_DAYS
        )
        
        # Data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues on limited resources
        )
        
        # Model and optimizer
        model = self.models[symbol]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                logits, confidence, model_outputs = model(
                    batch['price'],
                    batch['technical'],
                    batch['sentiment']
                )
                
                # Loss calculation
                loss = criterion(logits, batch['target'].squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        logger.info(f"Training completed for {symbol}")
    
    def predict(self, 
                symbol: str,
                technical_features: np.ndarray,
                prices: np.ndarray,
                sentiment_scores: np.ndarray,
                current_price: float) -> PredictionResult:
        """Make prediction for a symbol"""
        
        model = self.models[symbol]
        model.eval()
        
        # Prepare input sequences
        sequence_length = system_config.LSTM_LOOKBACK_DAYS
        
        if len(technical_features) < sequence_length:
            logger.warning(f"Insufficient data for prediction: {len(technical_features)} < {sequence_length}")
            return self._create_hold_prediction(symbol, current_price)
        
        # Get last sequence
        price_seq = torch.FloatTensor(prices[-sequence_length:]).unsqueeze(0).to(self.device)
        tech_seq = torch.FloatTensor(technical_features[-sequence_length:]).unsqueeze(0).to(self.device)
        sentiment_seq = torch.FloatTensor(sentiment_scores[-sequence_length:]).unsqueeze(0).to(self.device)
        
        # Prediction
        with torch.no_grad():
            logits, confidence, model_outputs = model(price_seq, tech_seq, sentiment_seq)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            prediction_confidence = confidence.item()
        
        # Convert to trading signal
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[predicted_class]
        
        # Apply confidence threshold
        if prediction_confidence < system_config.CONFIDENCE_THRESHOLD:
            signal = 'HOLD'
        
        # Calculate trading parameters
        entry_price = current_price
        stop_loss, take_profit, leverage = self._calculate_trading_params(
            signal, current_price, prediction_confidence
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            signal, model_outputs, technical_features[-1], sentiment_scores[-1]
        )
        
        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=signal,
            confidence=prediction_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            rationale=rationale,
            model_outputs=model_outputs,
            technical_features=technical_features[-1],
            sentiment_score=sentiment_scores[-1][0]
        )
    
    def _create_hold_prediction(self, symbol: str, current_price: float) -> PredictionResult:
        """Create a default HOLD prediction"""
        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            signal='HOLD',
            confidence=0.5,
            entry_price=current_price,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.02,
            leverage=1.0,
            rationale="Insufficient data for prediction",
            model_outputs={},
            technical_features=np.zeros(self.feature_size),
            sentiment_score=0.0
        )
    
    def _calculate_trading_params(self, signal: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Calculate stop loss, take profit, and leverage"""
        
        # Base parameters
        base_stop_loss = system_config.DEFAULT_STOP_LOSS
        base_take_profit = system_config.DEFAULT_TAKE_PROFIT
        
        # Adjust based on confidence
        confidence_multiplier = confidence * 2  # Scale confidence
        
        if signal == 'BUY':
            stop_loss = price * (1 - base_stop_loss * confidence_multiplier)
            take_profit = price * (1 + base_take_profit * confidence_multiplier)
            leverage = min(system_config.MAX_LEVERAGE, 1 + confidence)
        elif signal == 'SELL':
            stop_loss = price * (1 + base_stop_loss * confidence_multiplier)
            take_profit = price * (1 - base_take_profit * confidence_multiplier)
            leverage = min(system_config.MAX_LEVERAGE, 1 + confidence)
        else:  # HOLD
            stop_loss = price * 0.98
            take_profit = price * 1.02
            leverage = 1.0
        
        return stop_loss, take_profit, leverage
    
    def _generate_rationale(self, signal: str, model_outputs: Dict, technical_features: np.ndarray, sentiment_score: float) -> str:
        """Generate human-readable rationale for the prediction"""
        
        rationale_parts = []
        
        # Signal explanation
        if signal == 'BUY':
            rationale_parts.append("Bullish signal detected")
        elif signal == 'SELL':
            rationale_parts.append("Bearish signal detected")
        else:
            rationale_parts.append("Neutral market conditions")
        
        # Technical analysis
        if len(technical_features) > 20:
            rsi = technical_features[5] * 100  # Assuming RSI is at index 5
            if rsi < 30:
                rationale_parts.append("RSI oversold")
            elif rsi > 70:
                rationale_parts.append("RSI overbought")
        
        # Sentiment analysis
        if sentiment_score > 0.1:
            rationale_parts.append("Positive news sentiment")
        elif sentiment_score < -0.1:
            rationale_parts.append("Negative news sentiment")
        
        # Model ensemble
        if 'ensemble_weights' in model_outputs:
            weights = model_outputs['ensemble_weights']
            dominant_model = ['LSTM', 'Transformer', 'Sentiment'][torch.argmax(weights).item()]
            rationale_parts.append(f"{dominant_model} model dominant")
        
        return " + ".join(rationale_parts)
    
    def save_model(self, symbol: str, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.models[symbol].state_dict(),
            'scalers': self.scalers[symbol],
            'feature_size': self.feature_size
        }, filepath)
    
    def load_model(self, symbol: str, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.models[symbol].load_state_dict(checkpoint['model_state_dict'])
        self.scalers[symbol] = checkpoint['scalers']
        self.feature_size = checkpoint.get('feature_size', self.feature_size)

# Create global instance
prediction_system = CryptoPredictionSystem()