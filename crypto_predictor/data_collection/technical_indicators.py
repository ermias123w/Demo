import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..config.config import tech_config

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Technical indicators data structure"""
    symbol: str
    timestamp: datetime
    
    # Price indicators
    price: float
    sma_20: float
    ema_10: float
    ema_50: float
    ema_200: float
    
    # Oscillators
    rsi: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    
    # MACD
    macd: float
    macd_signal: float
    macd_histogram: float
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_percent: float
    
    # Volume indicators
    volume: float
    volume_sma: float
    volume_ratio: float
    mfi: float
    
    # Volatility
    atr: float
    volatility: float
    
    # Trend indicators
    adx: float
    cci: float
    
    # Support/Resistance
    support_level: float
    resistance_level: float
    
    # Custom indicators
    price_momentum: float
    volume_momentum: float
    volatility_ratio: float
    
    # Signals
    ma_crossover_signal: int  # 1 for bullish, -1 for bearish, 0 for neutral
    rsi_signal: int
    macd_signal_cross: int
    bb_signal: int
    volume_signal: int

class TechnicalAnalyzer:
    """Technical analysis calculator"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> List[TechnicalIndicators]:
        """Calculate all technical indicators for a symbol"""
        if df.empty:
            return []
        
        # Check cache
        cache_key = f"{symbol}_{len(df)}"
        if cache_key in self.indicators_cache:
            cache_time, cached_indicators = self.indicators_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                return cached_indicators
        
        try:
            # Ensure required columns exist
            if 'price' not in df.columns:
                if 'close' in df.columns:
                    df['price'] = df['close']
                else:
                    logger.error(f"No price data found for {symbol}")
                    return []
            
            # Calculate all indicators
            indicators_df = self._calculate_all_indicators(df.copy())
            
            # Convert to TechnicalIndicators objects
            indicators_list = []
            for idx, row in indicators_df.iterrows():
                indicator = TechnicalIndicators(
                    symbol=symbol,
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    price=row.get('price', 0),
                    sma_20=row.get('sma_20', 0),
                    ema_10=row.get('ema_10', 0),
                    ema_50=row.get('ema_50', 0),
                    ema_200=row.get('ema_200', 0),
                    rsi=row.get('rsi', 50),
                    stoch_k=row.get('stoch_k', 50),
                    stoch_d=row.get('stoch_d', 50),
                    williams_r=row.get('williams_r', -50),
                    macd=row.get('macd', 0),
                    macd_signal=row.get('macd_signal', 0),
                    macd_histogram=row.get('macd_histogram', 0),
                    bb_upper=row.get('bb_upper', 0),
                    bb_middle=row.get('bb_middle', 0),
                    bb_lower=row.get('bb_lower', 0),
                    bb_width=row.get('bb_width', 0),
                    bb_percent=row.get('bb_percent', 0),
                    volume=row.get('volume', 0),
                    volume_sma=row.get('volume_sma', 0),
                    volume_ratio=row.get('volume_ratio', 1),
                    mfi=row.get('mfi', 50),
                    atr=row.get('atr', 0),
                    volatility=row.get('volatility', 0),
                    adx=row.get('adx', 25),
                    cci=row.get('cci', 0),
                    support_level=row.get('support_level', 0),
                    resistance_level=row.get('resistance_level', 0),
                    price_momentum=row.get('price_momentum', 0),
                    volume_momentum=row.get('volume_momentum', 0),
                    volatility_ratio=row.get('volatility_ratio', 1),
                    ma_crossover_signal=int(row.get('ma_crossover_signal', 0)),
                    rsi_signal=int(row.get('rsi_signal', 0)),
                    macd_signal_cross=int(row.get('macd_signal_cross', 0)),
                    bb_signal=int(row.get('bb_signal', 0)),
                    volume_signal=int(row.get('volume_signal', 0))
                )
                indicators_list.append(indicator)
            
            # Cache results
            self.indicators_cache[cache_key] = (datetime.now(), indicators_list)
            
            logger.info(f"Calculated {len(indicators_list)} technical indicators for {symbol}")
            return indicators_list
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return []
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # Ensure we have OHLCV data
        if 'high' not in df.columns:
            df['high'] = df['price']
        if 'low' not in df.columns:
            df['low'] = df['price']
        if 'open' not in df.columns:
            df['open'] = df['price']
        if 'close' not in df.columns:
            df['close'] = df['price']
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=tech_config.EMA_PERIODS[0])
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=tech_config.EMA_PERIODS[1])
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=tech_config.EMA_PERIODS[2])
        
        # Oscillators
        df['rsi'] = ta.momentum.rsi(df['close'], window=tech_config.RSI_PERIOD)
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'], window_slow=tech_config.MACD_SLOW, window_fast=tech_config.MACD_FAST)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=tech_config.MACD_SLOW, window_fast=tech_config.MACD_FAST, window_sign=tech_config.MACD_SIGNAL)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_slow=tech_config.MACD_SLOW, window_fast=tech_config.MACD_FAST, window_sign=tech_config.MACD_SIGNAL)
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=tech_config.BB_PERIOD, window_dev=tech_config.BB_STD)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=tech_config.BB_PERIOD)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=tech_config.BB_PERIOD, window_dev=tech_config.BB_STD)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=tech_config.VOLUME_SMA_PERIOD)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=tech_config.ATR_PERIOD)
        df['volatility'] = df['close'].rolling(window=tech_config.VOLATILITY_PERIOD).std()
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # Support and Resistance levels
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()
        
        # Custom indicators
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['volume_momentum'] = df['volume'].pct_change(periods=5)
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=30).mean()
        
        # Trading signals
        df['ma_crossover_signal'] = self._calculate_ma_crossover_signal(df)
        df['rsi_signal'] = self._calculate_rsi_signal(df)
        df['macd_signal_cross'] = self._calculate_macd_signal_cross(df)
        df['bb_signal'] = self._calculate_bb_signal(df)
        df['volume_signal'] = self._calculate_volume_signal(df)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_ma_crossover_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate moving average crossover signals"""
        signal = pd.Series(0, index=df.index)
        
        # EMA 10 crosses above EMA 50 (bullish)
        bullish_cross = (df['ema_10'] > df['ema_50']) & (df['ema_10'].shift(1) <= df['ema_50'].shift(1))
        signal[bullish_cross] = 1
        
        # EMA 10 crosses below EMA 50 (bearish)
        bearish_cross = (df['ema_10'] < df['ema_50']) & (df['ema_10'].shift(1) >= df['ema_50'].shift(1))
        signal[bearish_cross] = -1
        
        return signal
    
    def _calculate_rsi_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI signals"""
        signal = pd.Series(0, index=df.index)
        
        # RSI oversold (bullish)
        oversold = df['rsi'] < tech_config.RSI_OVERSOLD
        signal[oversold] = 1
        
        # RSI overbought (bearish)
        overbought = df['rsi'] > tech_config.RSI_OVERBOUGHT
        signal[overbought] = -1
        
        return signal
    
    def _calculate_macd_signal_cross(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD signal line crossover"""
        signal = pd.Series(0, index=df.index)
        
        # MACD crosses above signal line (bullish)
        bullish_cross = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        signal[bullish_cross] = 1
        
        # MACD crosses below signal line (bearish)
        bearish_cross = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        signal[bearish_cross] = -1
        
        return signal
    
    def _calculate_bb_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands signals"""
        signal = pd.Series(0, index=df.index)
        
        # Price touches lower band (potential bullish)
        lower_touch = df['close'] <= df['bb_lower']
        signal[lower_touch] = 1
        
        # Price touches upper band (potential bearish)
        upper_touch = df['close'] >= df['bb_upper']
        signal[upper_touch] = -1
        
        return signal
    
    def _calculate_volume_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume signals"""
        signal = pd.Series(0, index=df.index)
        
        # Volume spike (significant)
        volume_spike = df['volume_ratio'] > tech_config.VOLUME_SPIKE_THRESHOLD
        signal[volume_spike] = 1
        
        return signal
    
    def get_feature_vector(self, indicators: TechnicalIndicators) -> np.ndarray:
        """Convert technical indicators to feature vector for ML models"""
        features = [
            # Price features
            indicators.price,
            indicators.sma_20,
            indicators.ema_10,
            indicators.ema_50,
            indicators.ema_200,
            
            # Oscillators
            indicators.rsi / 100.0,  # Normalize to 0-1
            indicators.stoch_k / 100.0,
            indicators.stoch_d / 100.0,
            indicators.williams_r / -100.0,  # Normalize to 0-1
            
            # MACD
            indicators.macd,
            indicators.macd_signal,
            indicators.macd_histogram,
            
            # Bollinger Bands
            indicators.bb_percent,
            indicators.bb_width,
            
            # Volume
            indicators.volume_ratio,
            indicators.mfi / 100.0,
            
            # Volatility
            indicators.atr,
            indicators.volatility,
            
            # Trend
            indicators.adx / 100.0,
            indicators.cci / 100.0,
            
            # Custom indicators
            indicators.price_momentum,
            indicators.volume_momentum,
            indicators.volatility_ratio,
            
            # Signals
            indicators.ma_crossover_signal,
            indicators.rsi_signal,
            indicators.macd_signal_cross,
            indicators.bb_signal,
            indicators.volume_signal
        ]
        
        # Convert to numpy array and handle NaN values
        feature_vector = np.array(features, dtype=np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_vector
    
    def get_signal_summary(self, indicators: TechnicalIndicators) -> Dict[str, any]:
        """Get summary of trading signals"""
        signals = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'overall_signal': 'NEUTRAL',
            'confidence': 0.0,
            'key_indicators': []
        }
        
        # Count signals
        signal_values = [
            indicators.ma_crossover_signal,
            indicators.rsi_signal,
            indicators.macd_signal_cross,
            indicators.bb_signal,
            indicators.volume_signal
        ]
        
        signals['bullish_signals'] = sum(1 for s in signal_values if s > 0)
        signals['bearish_signals'] = sum(1 for s in signal_values if s < 0)
        signals['neutral_signals'] = sum(1 for s in signal_values if s == 0)
        
        # Determine overall signal
        if signals['bullish_signals'] > signals['bearish_signals']:
            signals['overall_signal'] = 'BULLISH'
            signals['confidence'] = signals['bullish_signals'] / len(signal_values)
        elif signals['bearish_signals'] > signals['bullish_signals']:
            signals['overall_signal'] = 'BEARISH'
            signals['confidence'] = signals['bearish_signals'] / len(signal_values)
        else:
            signals['overall_signal'] = 'NEUTRAL'
            signals['confidence'] = 0.5
        
        # Key indicators
        if indicators.rsi < 30:
            signals['key_indicators'].append('RSI_OVERSOLD')
        elif indicators.rsi > 70:
            signals['key_indicators'].append('RSI_OVERBOUGHT')
        
        if indicators.macd > indicators.macd_signal:
            signals['key_indicators'].append('MACD_BULLISH')
        elif indicators.macd < indicators.macd_signal:
            signals['key_indicators'].append('MACD_BEARISH')
        
        if indicators.bb_percent < 0.2:
            signals['key_indicators'].append('BB_OVERSOLD')
        elif indicators.bb_percent > 0.8:
            signals['key_indicators'].append('BB_OVERBOUGHT')
        
        if indicators.volume_ratio > 2.0:
            signals['key_indicators'].append('VOLUME_SPIKE')
        
        return signals
    
    def get_support_resistance(self, df: pd.DataFrame, periods: int = 20) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels"""
        if len(df) < periods:
            return 0.0, 0.0
        
        recent_data = df.tail(periods)
        
        # Support: lowest low in recent period
        support = recent_data['low'].min() if 'low' in recent_data.columns else recent_data['price'].min()
        
        # Resistance: highest high in recent period
        resistance = recent_data['high'].max() if 'high' in recent_data.columns else recent_data['price'].max()
        
        return support, resistance
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility metrics"""
        if len(df) < 30:
            return {'volatility': 0.0, 'volatility_ratio': 1.0, 'volatility_percentile': 0.5}
        
        # Price volatility (standard deviation of returns)
        returns = df['price'].pct_change().dropna()
        volatility = returns.std()
        
        # Volatility ratio (current vs historical)
        recent_volatility = returns.tail(7).std()  # Last 7 periods
        historical_volatility = returns.std()
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # Volatility percentile
        volatility_percentile = (returns.tail(1).std() <= returns.rolling(30).std()).mean()
        
        return {
            'volatility': volatility,
            'volatility_ratio': volatility_ratio,
            'volatility_percentile': volatility_percentile
        }

# Create global instance
technical_analyzer = TechnicalAnalyzer()