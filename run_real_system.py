#!/usr/bin/env python3
"""
Real-time Crypto Prediction System with Trade Logging and RL Updates
"""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeLogger:
    """Handles trade logging and database operations"""
    
    def __init__(self, db_path: str = "trade_log.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the trade database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                leverage REAL NOT NULL,
                position_size REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                rationale TEXT,
                technical_data TEXT,
                sentiment_score REAL,
                prediction_id TEXT UNIQUE,
                
                -- Outcome fields (filled after 24h)
                exit_price REAL,
                exit_time TIMESTAMP,
                outcome TEXT,  -- WIN/LOSS/NEUTRAL
                profit_loss REAL,
                profit_loss_percent REAL,
                evaluation_time TIMESTAMP,
                
                -- RL Update fields
                reward REAL,
                model_weights_before TEXT,
                model_weights_after TEXT,
                rl_update_time TIMESTAMP
            )
        """)
        
        # Create performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                neutral INTEGER,
                win_rate REAL,
                avg_profit_loss REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                confidence_accuracy REAL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Trade database initialized")
    
    def store_prediction(self, prediction: Dict) -> str:
        """Store a new prediction in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prediction_id = f"pred_{int(time.time())}_{prediction['symbol']}"
        
        cursor.execute("""
            INSERT INTO trades (
                symbol, signal, confidence, entry_price, stop_loss, take_profit,
                leverage, position_size, entry_time, rationale, technical_data,
                sentiment_score, prediction_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction['symbol'],
            prediction['signal'],
            prediction['confidence'],
            prediction['entry_price'],
            prediction['stop_loss'],
            prediction['take_profit'],
            prediction['leverage'],
            prediction['position_size'],
            prediction['timestamp'],
            prediction['rationale'],
            json.dumps(prediction['technical_data']),
            prediction['sentiment_score'],
            prediction_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction {prediction_id}: {prediction['signal']} {prediction['symbol']} @ {prediction['confidence']:.1%}")
        return prediction_id
    
    def update_outcome(self, prediction_id: str, outcome_data: Dict):
        """Update trade outcome after 24h evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                outcome = ?,
                profit_loss = ?,
                profit_loss_percent = ?,
                evaluation_time = ?
            WHERE prediction_id = ?
        """, (
            outcome_data['exit_price'],
            outcome_data['exit_time'],
            outcome_data['outcome'],
            outcome_data['profit_loss'],
            outcome_data['profit_loss_percent'],
            outcome_data['evaluation_time'],
            prediction_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated outcome for {prediction_id}: {outcome_data['outcome']} ({outcome_data['profit_loss_percent']:+.2%})")
    
    def update_rl_data(self, prediction_id: str, rl_data: Dict):
        """Update RL learning data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trades SET
                reward = ?,
                model_weights_before = ?,
                model_weights_after = ?,
                rl_update_time = ?
            WHERE prediction_id = ?
        """, (
            rl_data['reward'],
            json.dumps(rl_data['weights_before']),
            json.dumps(rl_data['weights_after']),
            rl_data['update_time'],
            prediction_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated RL data for {prediction_id}: reward={rl_data['reward']:.3f}")
    
    def get_pending_evaluations(self) -> List[Dict]:
        """Get trades pending 24h evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trades older than 24 hours without outcome
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        cursor.execute("""
            SELECT * FROM trades 
            WHERE outcome IS NULL 
            AND entry_time < ?
            ORDER BY entry_time
        """, (cutoff_time,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'NEUTRAL' THEN 1 ELSE 0 END) as neutral,
                AVG(CASE WHEN outcome IS NOT NULL THEN profit_loss_percent END) as avg_return,
                AVG(confidence) as avg_confidence
            FROM trades
            WHERE outcome IS NOT NULL
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            total, wins, losses, neutral, avg_return, avg_confidence = row
            win_rate = wins / total if total > 0 else 0
            
            return {
                'total_trades': total,
                'wins': wins,
                'losses': losses,
                'neutral': neutral,
                'win_rate': win_rate,
                'avg_return': avg_return or 0,
                'avg_confidence': avg_confidence or 0
            }
        
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'neutral': 0,
            'win_rate': 0,
            'avg_return': 0,
            'avg_confidence': 0
        }

class RealTimeDataCollector:
    """Collects real-time market data"""
    
    def __init__(self):
        self.session = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting
    
    def get_market_data(self, symbols: List[str]) -> Dict:
        """Get real-time market data for symbols"""
        import requests
        import time
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # CoinGecko API (free tier)
                coin_id = 'bitcoin' if symbol == 'BTC' else 'ethereum'
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': coin_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true',
                    'include_market_cap': 'true'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if coin_id in data:
                    market_data[symbol] = {
                        'price': data[coin_id]['usd'],
                        'change_24h': data[coin_id].get('usd_24h_change', 0) / 100,
                        'volume_24h': data[coin_id].get('usd_24h_vol', 0),
                        'market_cap': data[coin_id].get('usd_market_cap', 0),
                        'timestamp': datetime.now()
                    }
                
                time.sleep(0.5)  # Rate limiting between requests
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                # Fallback to simulated data
                market_data[symbol] = self.get_fallback_data(symbol)
        
        self.last_request_time = time.time()
        return market_data
    
    def get_fallback_data(self, symbol: str) -> Dict:
        """Fallback data when API fails"""
        import random
        
        base_prices = {'BTC': 45000, 'ETH': 2500}
        base_price = base_prices.get(symbol, 1000)
        
        return {
            'price': base_price + random.uniform(-base_price*0.05, base_price*0.05),
            'change_24h': random.uniform(-0.05, 0.05),
            'volume_24h': random.uniform(1000000, 5000000),
            'market_cap': base_price * 19000000,
            'timestamp': datetime.now()
        }
    
    def calculate_technical_indicators(self, symbol: str, current_price: float) -> Dict:
        """Calculate technical indicators (simplified for demo)"""
        import random
        
        # In a real implementation, you'd use historical data
        # For demo, we'll simulate realistic indicators
        
        return {
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-100, 100),
            'macd_signal': random.uniform(-100, 100),
            'sma_20': current_price * random.uniform(0.98, 1.02),
            'sma_50': current_price * random.uniform(0.95, 1.05),
            'bollinger_upper': current_price * 1.02,
            'bollinger_lower': current_price * 0.98,
            'volume_sma': random.uniform(1000000, 3000000),
            'atr': current_price * random.uniform(0.01, 0.03),
            'williams_r': random.uniform(-80, -20),
            'stoch_k': random.uniform(20, 80),
            'stoch_d': random.uniform(20, 80)
        }
    
    def get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score (simplified for demo)"""
        import random
        
        # In a real implementation, you'd analyze news/social media
        # For demo, we'll simulate realistic sentiment
        
        return random.uniform(-0.5, 0.5)

class PredictionEngine:
    """Generates predictions using ensemble models"""
    
    def __init__(self):
        self.model_weights = {
            'technical': 0.4,
            'sentiment': 0.3,
            'momentum': 0.3
        }
        self.confidence_threshold = 0.6
    
    def generate_prediction(self, symbol: str, market_data: Dict, technical_data: Dict, sentiment_score: float) -> Dict:
        """Generate a prediction for the given symbol"""
        
        # Technical analysis signal
        technical_signal = self.analyze_technical_indicators(technical_data)
        
        # Sentiment signal
        sentiment_signal = self.analyze_sentiment(sentiment_score)
        
        # Momentum signal
        momentum_signal = self.analyze_momentum(market_data)
        
        # Ensemble prediction
        ensemble_score = (
            technical_signal * self.model_weights['technical'] +
            sentiment_signal * self.model_weights['sentiment'] +
            momentum_signal * self.model_weights['momentum']
        )
        
        # Convert to signal
        if ensemble_score > 0.2:
            signal = 'BUY'
            confidence = 0.6 + abs(ensemble_score) * 0.3
        elif ensemble_score < -0.2:
            signal = 'SELL'
            confidence = 0.6 + abs(ensemble_score) * 0.3
        else:
            signal = 'HOLD'
            confidence = 0.5 + abs(ensemble_score) * 0.2
        
        # Cap confidence
        confidence = min(0.95, confidence)
        
        # Calculate entry parameters
        entry_price = market_data['price']
        
        if signal == 'BUY':
            stop_loss = entry_price * 0.98  # 2% stop loss
            take_profit = entry_price * 1.04  # 4% take profit
        elif signal == 'SELL':
            stop_loss = entry_price * 1.02  # 2% stop loss
            take_profit = entry_price * 0.96  # 4% take profit
        else:
            stop_loss = entry_price * 0.99
            take_profit = entry_price * 1.01
        
        # Calculate leverage and position size
        leverage = 1.0 + (confidence - 0.5) * 2  # 1x to 2x leverage
        leverage = min(3.0, leverage)
        
        position_size = confidence * 0.1  # Max 10% position size
        
        # Generate rationale
        rationale = self.generate_rationale(technical_data, sentiment_score, ensemble_score)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage,
            'position_size': position_size,
            'timestamp': datetime.now(),
            'rationale': rationale,
            'technical_data': technical_data,
            'sentiment_score': sentiment_score,
            'ensemble_score': ensemble_score
        }
    
    def analyze_technical_indicators(self, tech_data: Dict) -> float:
        """Analyze technical indicators and return signal strength"""
        signals = []
        
        # RSI analysis
        rsi = tech_data['rsi']
        if rsi < 30:
            signals.append(0.5)  # Oversold - buy signal
        elif rsi > 70:
            signals.append(-0.5)  # Overbought - sell signal
        else:
            signals.append(0)
        
        # MACD analysis
        macd = tech_data['macd']
        macd_signal = tech_data['macd_signal']
        if macd > macd_signal:
            signals.append(0.3)  # Bullish crossover
        else:
            signals.append(-0.3)  # Bearish crossover
        
        # Moving average analysis
        sma_20 = tech_data['sma_20']
        sma_50 = tech_data['sma_50']
        if sma_20 > sma_50:
            signals.append(0.2)  # Bullish trend
        else:
            signals.append(-0.2)  # Bearish trend
        
        return sum(signals) / len(signals)
    
    def analyze_sentiment(self, sentiment_score: float) -> float:
        """Analyze sentiment and return signal strength"""
        # Normalize sentiment score
        return sentiment_score  # Already normalized between -0.5 and 0.5
    
    def analyze_momentum(self, market_data: Dict) -> float:
        """Analyze momentum and return signal strength"""
        change_24h = market_data['change_24h']
        
        # Normalize momentum signal
        if change_24h > 0.02:  # Strong positive momentum
            return 0.4
        elif change_24h > 0:  # Positive momentum
            return 0.2
        elif change_24h < -0.02:  # Strong negative momentum
            return -0.4
        elif change_24h < 0:  # Negative momentum
            return -0.2
        else:
            return 0
    
    def generate_rationale(self, tech_data: Dict, sentiment: float, ensemble_score: float) -> str:
        """Generate human-readable rationale"""
        parts = []
        
        # Technical analysis
        rsi = tech_data['rsi']
        if rsi < 30:
            parts.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            parts.append(f"RSI overbought ({rsi:.1f})")
        else:
            parts.append(f"RSI neutral ({rsi:.1f})")
        
        # MACD
        macd = tech_data['macd']
        macd_signal = tech_data['macd_signal']
        if macd > macd_signal:
            parts.append("MACD bullish")
        else:
            parts.append("MACD bearish")
        
        # Sentiment
        if sentiment > 0.1:
            parts.append("positive sentiment")
        elif sentiment < -0.1:
            parts.append("negative sentiment")
        else:
            parts.append("neutral sentiment")
        
        # Ensemble score
        if ensemble_score > 0.2:
            parts.append("strong bullish signals")
        elif ensemble_score < -0.2:
            parts.append("strong bearish signals")
        else:
            parts.append("mixed signals")
        
        return f"Based on {', '.join(parts)}"

class OutcomeEvaluator:
    """Evaluates prediction outcomes after 24h"""
    
    def __init__(self, data_collector: RealTimeDataCollector):
        self.data_collector = data_collector
    
    def evaluate_prediction(self, trade_data: Dict) -> Dict:
        """Evaluate a prediction outcome"""
        symbol = trade_data['symbol']
        entry_price = trade_data['entry_price']
        stop_loss = trade_data['stop_loss']
        take_profit = trade_data['take_profit']
        signal = trade_data['signal']
        leverage = trade_data['leverage']
        
        # Get current price
        current_market_data = self.data_collector.get_market_data([symbol])
        current_price = current_market_data[symbol]['price']
        
        # Calculate price change
        price_change = (current_price - entry_price) / entry_price
        
        # Determine outcome
        outcome = self.determine_outcome(signal, price_change, current_price, stop_loss, take_profit)
        
        # Calculate profit/loss
        profit_loss = self.calculate_profit_loss(signal, price_change, leverage, stop_loss, take_profit, entry_price, current_price)
        
        return {
            'exit_price': current_price,
            'exit_time': datetime.now(),
            'outcome': outcome,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss,
            'price_change': price_change,
            'evaluation_time': datetime.now()
        }
    
    def determine_outcome(self, signal: str, price_change: float, current_price: float, 
                         stop_loss: float, take_profit: float) -> str:
        """Determine if the trade was WIN/LOSS/NEUTRAL"""
        
        # Check if stop loss or take profit was hit
        if signal == 'BUY':
            if current_price <= stop_loss:
                return 'LOSS'
            elif current_price >= take_profit:
                return 'WIN'
        elif signal == 'SELL':
            if current_price >= stop_loss:
                return 'LOSS'
            elif current_price <= take_profit:
                return 'WIN'
        
        # If no SL/TP hit, check price movement
        if signal == 'BUY':
            if price_change > 0.01:  # 1% threshold
                return 'WIN'
            elif price_change < -0.01:
                return 'LOSS'
        elif signal == 'SELL':
            if price_change < -0.01:
                return 'WIN'
            elif price_change > 0.01:
                return 'LOSS'
        
        return 'NEUTRAL'
    
    def calculate_profit_loss(self, signal: str, price_change: float, leverage: float,
                            stop_loss: float, take_profit: float, entry_price: float,
                            current_price: float) -> float:
        """Calculate profit/loss percentage"""
        
        # Check if SL/TP was hit first
        if signal == 'BUY':
            if current_price <= stop_loss:
                return -0.02 * leverage  # 2% loss * leverage
            elif current_price >= take_profit:
                return 0.04 * leverage  # 4% profit * leverage
        elif signal == 'SELL':
            if current_price >= stop_loss:
                return -0.02 * leverage  # 2% loss * leverage
            elif current_price <= take_profit:
                return 0.04 * leverage  # 4% profit * leverage
        
        # Calculate actual P&L based on direction
        if signal == 'BUY':
            return price_change * leverage
        elif signal == 'SELL':
            return -price_change * leverage
        else:  # HOLD
            return 0

class ReinforcementLearner:
    """Handles reinforcement learning updates"""
    
    def __init__(self, prediction_engine: PredictionEngine):
        self.prediction_engine = prediction_engine
        self.learning_rate = 0.01
        self.performance_history = []
    
    def update_from_outcome(self, prediction: Dict, outcome: Dict) -> Dict:
        """Update model based on prediction outcome"""
        
        # Calculate reward
        reward = self.calculate_reward(prediction, outcome)
        
        # Store current weights
        weights_before = self.prediction_engine.model_weights.copy()
        
        # Update model weights
        self.update_model_weights(prediction, outcome, reward)
        
        # Get new weights
        weights_after = self.prediction_engine.model_weights.copy()
        
        # Update confidence threshold
        self.update_confidence_threshold(prediction, outcome)
        
        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'reward': reward,
            'outcome': outcome['outcome'],
            'confidence': prediction['confidence'],
            'profit_loss': outcome['profit_loss']
        })
        
        logger.info(f"RL Update: reward={reward:.3f}, outcome={outcome['outcome']}")
        
        return {
            'reward': reward,
            'weights_before': weights_before,
            'weights_after': weights_after,
            'update_time': datetime.now()
        }
    
    def calculate_reward(self, prediction: Dict, outcome: Dict) -> float:
        """Calculate reward based on prediction outcome"""
        
        # Base reward
        if outcome['outcome'] == 'WIN':
            base_reward = 1.0
        elif outcome['outcome'] == 'LOSS':
            base_reward = -1.0
        else:  # NEUTRAL
            base_reward = 0.0
        
        # Confidence bonus/penalty
        confidence_factor = prediction['confidence']
        if outcome['outcome'] == 'WIN':
            confidence_bonus = confidence_factor * 0.5
        else:
            confidence_bonus = -confidence_factor * 0.3
        
        # Profit/loss bonus
        profit_bonus = outcome['profit_loss'] * 5  # Scale profit/loss
        
        # Risk adjustment
        leverage = prediction['leverage']
        risk_penalty = -(leverage - 1.0) * 0.2  # Penalize high leverage
        
        total_reward = base_reward + confidence_bonus + profit_bonus + risk_penalty
        
        return max(-2.0, min(2.0, total_reward))  # Clip reward
    
    def update_model_weights(self, prediction: Dict, outcome: Dict, reward: float):
        """Update model weights based on reward"""
        
        # Determine which components contributed to the prediction
        technical_contrib = abs(prediction.get('technical_score', 0))
        sentiment_contrib = abs(prediction.get('sentiment_score', 0))
        momentum_contrib = abs(prediction.get('momentum_score', 0))
        
        # Update weights based on reward and contribution
        adjustment = self.learning_rate * reward
        
        if outcome['outcome'] == 'WIN':
            # Increase weights for successful components
            self.prediction_engine.model_weights['technical'] += adjustment * technical_contrib
            self.prediction_engine.model_weights['sentiment'] += adjustment * sentiment_contrib
            self.prediction_engine.model_weights['momentum'] += adjustment * momentum_contrib
        else:
            # Decrease weights for failed components
            self.prediction_engine.model_weights['technical'] -= adjustment * technical_contrib * 0.5
            self.prediction_engine.model_weights['sentiment'] -= adjustment * sentiment_contrib * 0.5
            self.prediction_engine.model_weights['momentum'] -= adjustment * momentum_contrib * 0.5
        
        # Normalize weights
        total_weight = sum(self.prediction_engine.model_weights.values())
        if total_weight > 0:
            for key in self.prediction_engine.model_weights:
                self.prediction_engine.model_weights[key] /= total_weight
        
        # Ensure minimum weight
        for key in self.prediction_engine.model_weights:
            self.prediction_engine.model_weights[key] = max(0.1, self.prediction_engine.model_weights[key])
    
    def update_confidence_threshold(self, prediction: Dict, outcome: Dict):
        """Update confidence threshold based on performance"""
        
        if outcome['outcome'] == 'WIN' and prediction['confidence'] > 0.7:
            # Lower threshold for high-confidence wins
            self.prediction_engine.confidence_threshold *= 0.99
        elif outcome['outcome'] == 'LOSS' and prediction['confidence'] > 0.7:
            # Raise threshold for high-confidence losses
            self.prediction_engine.confidence_threshold *= 1.01
        
        # Clamp threshold
        self.prediction_engine.confidence_threshold = max(0.5, min(0.8, self.prediction_engine.confidence_threshold))

def main():
    """Main function to run the real-time system"""
    
    print("🚀 REAL-TIME CRYPTO PREDICTION SYSTEM")
    print("=====================================")
    print()
    
    # Initialize components
    trade_logger = TradeLogger()
    data_collector = RealTimeDataCollector()
    prediction_engine = PredictionEngine()
    outcome_evaluator = OutcomeEvaluator(data_collector)
    rl_learner = ReinforcementLearner(prediction_engine)
    
    # Show current performance
    performance = trade_logger.get_performance_summary()
    print(f"📊 Current Performance:")
    print(f"   • Total Trades: {performance['total_trades']}")
    print(f"   • Win Rate: {performance['win_rate']:.1%}")
    print(f"   • Average Return: {performance['avg_return']:+.2%}")
    print(f"   • Average Confidence: {performance['avg_confidence']:.1%}")
    print()
    
    # Step 1: Generate new predictions
    print("📋 STEP 1: GENERATING HIGH-CONFIDENCE PREDICTIONS")
    print("-" * 50)
    
    symbols = ['BTC', 'ETH']
    new_predictions = []
    
    for symbol in symbols:
        print(f"\n🔄 Collecting real-time data for {symbol}...")
        
        # Get market data
        market_data = data_collector.get_market_data([symbol])
        
        if symbol not in market_data:
            print(f"❌ Failed to get data for {symbol}")
            continue
        
        # Get technical indicators
        technical_data = data_collector.calculate_technical_indicators(symbol, market_data[symbol]['price'])
        
        # Get sentiment
        sentiment_score = data_collector.get_sentiment_score(symbol)
        
        # Generate prediction
        prediction = prediction_engine.generate_prediction(
            symbol, market_data[symbol], technical_data, sentiment_score
        )
        
        # Only store high-confidence predictions
        if prediction['confidence'] >= 0.65:  # 65% confidence threshold
            prediction_id = trade_logger.store_prediction(prediction)
            new_predictions.append((prediction_id, prediction))
            
            print(f"✅ HIGH-CONFIDENCE PREDICTION GENERATED:")
            print(f"   • Symbol: {prediction['symbol']}")
            print(f"   • Signal: {prediction['signal']}")
            print(f"   • Confidence: {prediction['confidence']:.1%}")
            print(f"   • Entry Price: ${prediction['entry_price']:,.2f}")
            print(f"   • Stop Loss: ${prediction['stop_loss']:,.2f}")
            print(f"   • Take Profit: ${prediction['take_profit']:,.2f}")
            print(f"   • Leverage: {prediction['leverage']:.1f}x")
            print(f"   • Position Size: {prediction['position_size']:.1%}")
            print(f"   • Rationale: {prediction['rationale']}")
            print(f"   • Prediction ID: {prediction_id}")
        else:
            print(f"⚠️  Low confidence ({prediction['confidence']:.1%}) - not storing")
    
    print(f"\n📊 Generated {len(new_predictions)} high-confidence predictions")
    
    # Step 2: Evaluate pending trades
    print("\n📋 STEP 2: EVALUATING PENDING TRADES (24H+ OLD)")
    print("-" * 50)
    
    pending_trades = trade_logger.get_pending_evaluations()
    
    if pending_trades:
        print(f"🔍 Found {len(pending_trades)} trades ready for evaluation")
        
        for trade in pending_trades:
            print(f"\n📊 Evaluating trade {trade['prediction_id']}:")
            print(f"   • Symbol: {trade['symbol']}")
            print(f"   • Entry: {trade['signal']} @ ${trade['entry_price']:,.2f}")
            print(f"   • Confidence: {trade['confidence']:.1%}")
            print(f"   • Entry Time: {trade['entry_time']}")
            
            # Evaluate outcome
            outcome = outcome_evaluator.evaluate_prediction(trade)
            
            # Update database
            trade_logger.update_outcome(trade['prediction_id'], outcome)
            
            print(f"   • Exit Price: ${outcome['exit_price']:,.2f}")
            print(f"   • Price Change: {outcome['price_change']:+.2%}")
            print(f"   • Outcome: {outcome['outcome']}")
            print(f"   • P&L: {outcome['profit_loss']:+.2%}")
            
            # Apply reinforcement learning
            rl_update = rl_learner.update_from_outcome(trade, outcome)
            trade_logger.update_rl_data(trade['prediction_id'], rl_update)
            
            print(f"   • RL Reward: {rl_update['reward']:+.3f}")
            print(f"   • Model Weights Updated: ✅")
    else:
        print("📝 No trades ready for evaluation (need 24h+ old trades)")
    
    # Step 3: Show updated performance
    print("\n📋 STEP 3: UPDATED PERFORMANCE METRICS")
    print("-" * 50)
    
    updated_performance = trade_logger.get_performance_summary()
    print(f"📊 Updated Performance:")
    print(f"   • Total Trades: {updated_performance['total_trades']}")
    print(f"   • Wins: {updated_performance['wins']}")
    print(f"   • Losses: {updated_performance['losses']}")
    print(f"   • Neutral: {updated_performance['neutral']}")
    print(f"   • Win Rate: {updated_performance['win_rate']:.1%}")
    print(f"   • Average Return: {updated_performance['avg_return']:+.2%}")
    print(f"   • Average Confidence: {updated_performance['avg_confidence']:.1%}")
    
    # Show current model weights
    print(f"\n🧠 Current Model Weights:")
    for component, weight in prediction_engine.model_weights.items():
        print(f"   • {component.title()}: {weight:.3f}")
    
    print(f"\n🎯 Confidence Threshold: {prediction_engine.confidence_threshold:.1%}")
    
    # Instructions for continuous operation
    print("\n📋 NEXT STEPS FOR CONTINUOUS OPERATION:")
    print("-" * 50)
    print("1. Set up a cron job to run this script every 4-6 hours")
    print("2. Monitor the trade_log.db database for results")
    print("3. Check trade_log.log for detailed logging")
    print("4. Wait 24+ hours for new predictions to be evaluated")
    print("5. The system will automatically improve with more data")
    print()
    print("📊 Database files created:")
    print("   • trade_log.db (SQLite database)")
    print("   • trade_log.log (detailed logs)")
    print()
    print("🚀 System is now running with real-time data and RL updates!")

if __name__ == "__main__":
    main()