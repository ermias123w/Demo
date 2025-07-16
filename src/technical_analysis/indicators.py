import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis indicators calculator"""
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, 
                 macd_slow: int = 26, macd_signal: int = 9,
                 ema_periods: List[int] = None, bb_period: int = 20,
                 bb_std: float = 2.0, atr_period: int = 14,
                 volume_spike_threshold: float = 2.0):
        
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_periods = ema_periods or [10, 50, 200]
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volume_spike_threshold = volume_spike_threshold
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            period = period or self.rsi_period
            rsi = talib.RSI(prices.values, timeperiod=period)
            return pd.Series(rsi, index=prices.index)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index)
    
    def calculate_macd(self, prices: pd.Series, fast: int = None, 
                      slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            fast = fast or self.macd_fast
            slow = slow or self.macd_slow
            signal = signal or self.macd_signal
            
            macd, macd_signal, macd_histogram = talib.MACD(
                prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            
            return {
                'macd': pd.Series(macd, index=prices.index),
                'macd_signal': pd.Series(macd_signal, index=prices.index),
                'macd_histogram': pd.Series(macd_histogram, index=prices.index)
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series(index=prices.index),
                'macd_signal': pd.Series(index=prices.index),
                'macd_histogram': pd.Series(index=prices.index)
            }
    
    def calculate_ema(self, prices: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """Calculate Exponential Moving Averages"""
        try:
            periods = periods or self.ema_periods
            emas = {}
            
            for period in periods:
                ema = talib.EMA(prices.values, timeperiod=period)
                emas[f'ema_{period}'] = pd.Series(ema, index=prices.index)
            
            return emas
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return {f'ema_{period}': pd.Series(index=prices.index) for period in periods}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = None, 
                                 std_dev: float = None) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            period = period or self.bb_period
            std_dev = std_dev or self.bb_std
            
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            
            return {
                'bb_upper': pd.Series(bb_upper, index=prices.index),
                'bb_middle': pd.Series(bb_middle, index=prices.index),
                'bb_lower': pd.Series(bb_lower, index=prices.index)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'bb_upper': pd.Series(index=prices.index),
                'bb_middle': pd.Series(index=prices.index),
                'bb_lower': pd.Series(index=prices.index)
            }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        try:
            period = period or self.atr_period
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(atr, index=close.index)
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=close.index)
    
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate price volatility (rolling standard deviation)"""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(window)
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return pd.Series(index=prices.index)
    
    def calculate_volume_spike(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volume spikes (volume / average volume)"""
        try:
            avg_volume = volume.rolling(window=window).mean()
            volume_ratio = volume / avg_volume
            return volume_ratio
        except Exception as e:
            logger.error(f"Error calculating volume spike: {e}")
            return pd.Series(index=volume.index)
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                      fastk_period=k_period, slowk_period=d_period,
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
            
            return {
                'stoch_k': pd.Series(slowk, index=close.index),
                'stoch_d': pd.Series(slowd, index=close.index)
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {
                'stoch_k': pd.Series(index=close.index),
                'stoch_d': pd.Series(index=close.index)
            }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            willr = talib.WILLR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(willr, index=close.index)
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return pd.Series(index=close.index)
    
    def calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            roc = talib.ROC(prices.values, timeperiod=period)
            return pd.Series(roc, index=prices.index)
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return pd.Series(index=prices.index)
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(cci, index=close.index)
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return pd.Series(index=close.index)
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(adx, index=close.index)
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series(index=close.index)
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = talib.OBV(close.values, volume.values)
            return pd.Series(obv, index=close.index)
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return pd.Series(index=close.index)
    
    def detect_ma_crossover(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Detect moving average crossovers"""
        try:
            # 1 for bullish crossover, -1 for bearish crossover, 0 for no crossover
            crossover = pd.Series(0, index=fast_ma.index)
            
            # Calculate crossover points
            fast_above = fast_ma > slow_ma
            fast_above_prev = fast_above.shift(1)
            
            # Bullish crossover: fast MA crosses above slow MA
            bullish_crossover = fast_above & (~fast_above_prev)
            bearish_crossover = (~fast_above) & fast_above_prev
            
            crossover[bullish_crossover] = 1
            crossover[bearish_crossover] = -1
            
            return crossover
        except Exception as e:
            logger.error(f"Error detecting MA crossover: {e}")
            return pd.Series(0, index=fast_ma.index)
    
    def calculate_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Calculate support and resistance levels"""
        try:
            # Rolling max and min for support/resistance
            resistance = prices.rolling(window=window).max()
            support = prices.rolling(window=window).min()
            
            return {
                'support': support,
                'resistance': resistance
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {
                'support': pd.Series(index=prices.index),
                'resistance': pd.Series(index=prices.index)
            }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame"""
        try:
            # Make a copy to avoid modifying original data
            result_df = df.copy()
            
            # Ensure we have required columns
            required_columns = ['price', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error("DataFrame must contain 'price' and 'volume' columns")
                return result_df
            
            # Use price as high, low, and close for simplicity (can be enhanced with OHLC data)
            prices = df['price']
            volume = df['volume']
            high = prices
            low = prices
            close = prices
            
            # Calculate RSI
            result_df['rsi'] = self.calculate_rsi(prices)
            
            # Calculate MACD
            macd_data = self.calculate_macd(prices)
            result_df['macd'] = macd_data['macd']
            result_df['macd_signal'] = macd_data['macd_signal']
            result_df['macd_histogram'] = macd_data['macd_histogram']
            
            # Calculate EMAs
            ema_data = self.calculate_ema(prices)
            for key, value in ema_data.items():
                result_df[key] = value
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(prices)
            for key, value in bb_data.items():
                result_df[key] = value
            
            # Calculate ATR (using price as high/low/close)
            result_df['atr'] = self.calculate_atr(high, low, close)
            
            # Calculate volatility
            result_df['volatility'] = self.calculate_volatility(prices)
            
            # Calculate volume spike
            result_df['volume_spike'] = self.calculate_volume_spike(volume)
            
            # Calculate additional indicators
            result_df['roc'] = self.calculate_roc(prices)
            result_df['cci'] = self.calculate_cci(high, low, close)
            result_df['adx'] = self.calculate_adx(high, low, close)
            result_df['obv'] = self.calculate_obv(close, volume)
            result_df['williams_r'] = self.calculate_williams_r(high, low, close)
            
            # Calculate stochastic
            stoch_data = self.calculate_stochastic(high, low, close)
            for key, value in stoch_data.items():
                result_df[key] = value
            
            # Calculate support and resistance
            sr_data = self.calculate_support_resistance(prices)
            for key, value in sr_data.items():
                result_df[key] = value
            
            # Calculate moving average crossovers
            if 'ema_10' in result_df.columns and 'ema_50' in result_df.columns:
                result_df['ma_crossover_10_50'] = self.detect_ma_crossover(
                    result_df['ema_10'], result_df['ema_50']
                )
            
            if 'ema_50' in result_df.columns and 'ema_200' in result_df.columns:
                result_df['ma_crossover_50_200'] = self.detect_ma_crossover(
                    result_df['ema_50'], result_df['ema_200']
                )
            
            logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return df
    
    def get_latest_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get the latest values of all indicators"""
        try:
            if df.empty:
                return {}
            
            # Get the last row (most recent data)
            latest_row = df.iloc[-1]
            
            # Extract technical indicators
            indicators = {}
            
            # Core indicators
            core_indicators = ['rsi', 'macd', 'macd_signal', 'macd_histogram',
                              'ema_10', 'ema_50', 'ema_200', 'bb_upper', 'bb_middle', 
                              'bb_lower', 'atr', 'volatility', 'volume_spike']
            
            for indicator in core_indicators:
                if indicator in latest_row:
                    value = latest_row[indicator]
                    indicators[indicator] = float(value) if not pd.isna(value) else None
            
            # Additional indicators
            additional_indicators = ['roc', 'cci', 'adx', 'obv', 'williams_r',
                                   'stoch_k', 'stoch_d', 'support', 'resistance',
                                   'ma_crossover_10_50', 'ma_crossover_50_200']
            
            for indicator in additional_indicators:
                if indicator in latest_row:
                    value = latest_row[indicator]
                    indicators[indicator] = float(value) if not pd.isna(value) else None
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting latest indicators: {e}")
            return {}
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        try:
            if df.empty:
                return {'signal': 'HOLD', 'reason': 'No data available'}
            
            latest_indicators = self.get_latest_indicators(df)
            signals = []
            reasons = []
            
            # RSI signals
            rsi = latest_indicators.get('rsi')
            if rsi is not None:
                if rsi < 30:
                    signals.append('BUY')
                    reasons.append('RSI Oversold')
                elif rsi > 70:
                    signals.append('SELL')
                    reasons.append('RSI Overbought')
            
            # MACD signals
            macd = latest_indicators.get('macd')
            macd_signal = latest_indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    signals.append('BUY')
                    reasons.append('MACD Bullish')
                else:
                    signals.append('SELL')
                    reasons.append('MACD Bearish')
            
            # Bollinger Bands signals
            bb_upper = latest_indicators.get('bb_upper')
            bb_lower = latest_indicators.get('bb_lower')
            current_price = df['price'].iloc[-1] if not df.empty else None
            
            if bb_upper is not None and bb_lower is not None and current_price is not None:
                if current_price <= bb_lower:
                    signals.append('BUY')
                    reasons.append('BB Oversold')
                elif current_price >= bb_upper:
                    signals.append('SELL')
                    reasons.append('BB Overbought')
            
            # Moving Average Crossover signals
            ma_cross_10_50 = latest_indicators.get('ma_crossover_10_50')
            ma_cross_50_200 = latest_indicators.get('ma_crossover_50_200')
            
            if ma_cross_10_50 == 1:
                signals.append('BUY')
                reasons.append('MA10-50 Golden Cross')
            elif ma_cross_10_50 == -1:
                signals.append('SELL')
                reasons.append('MA10-50 Death Cross')
            
            if ma_cross_50_200 == 1:
                signals.append('BUY')
                reasons.append('MA50-200 Golden Cross')
            elif ma_cross_50_200 == -1:
                signals.append('SELL')
                reasons.append('MA50-200 Death Cross')
            
            # Volume spike signals
            volume_spike = latest_indicators.get('volume_spike')
            if volume_spike is not None and volume_spike > self.volume_spike_threshold:
                # High volume can confirm other signals
                reasons.append('High Volume')
            
            # Determine overall signal
            if not signals:
                return {'signal': 'HOLD', 'reason': 'No clear signals'}
            
            # Count signals
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            if buy_count > sell_count:
                overall_signal = 'BUY'
            elif sell_count > buy_count:
                overall_signal = 'SELL'
            else:
                overall_signal = 'HOLD'
            
            return {
                'signal': overall_signal,
                'reason': ' + '.join(reasons),
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'confidence': abs(buy_count - sell_count) / len(signals) if signals else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'signal': 'HOLD', 'reason': 'Error in signal generation'}
    
    def calculate_indicator_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate the strength of each indicator (0-1 scale)"""
        try:
            if df.empty:
                return {}
            
            latest_indicators = self.get_latest_indicators(df)
            strengths = {}
            
            # RSI strength (distance from neutral 50)
            rsi = latest_indicators.get('rsi')
            if rsi is not None:
                strengths['rsi_strength'] = abs(rsi - 50) / 50
            
            # MACD strength (histogram magnitude)
            macd_hist = latest_indicators.get('macd_histogram')
            if macd_hist is not None:
                # Normalize by recent max histogram value
                recent_hist = df['macd_histogram'].dropna().tail(20)
                if not recent_hist.empty:
                    max_hist = recent_hist.abs().max()
                    if max_hist > 0:
                        strengths['macd_strength'] = abs(macd_hist) / max_hist
            
            # Bollinger Bands strength (distance from bands)
            bb_upper = latest_indicators.get('bb_upper')
            bb_lower = latest_indicators.get('bb_lower')
            current_price = df['price'].iloc[-1] if not df.empty else None
            
            if all(x is not None for x in [bb_upper, bb_lower, current_price]):
                band_width = bb_upper - bb_lower
                if band_width > 0:
                    # How close to bands (0 = middle, 1 = at bands)
                    middle = (bb_upper + bb_lower) / 2
                    strengths['bb_strength'] = abs(current_price - middle) / (band_width / 2)
            
            # Volume strength
            volume_spike = latest_indicators.get('volume_spike')
            if volume_spike is not None:
                strengths['volume_strength'] = min(volume_spike / self.volume_spike_threshold, 1.0)
            
            # Volatility strength
            volatility = latest_indicators.get('volatility')
            if volatility is not None:
                # Normalize by recent volatility
                recent_vol = df['volatility'].dropna().tail(20)
                if not recent_vol.empty:
                    avg_vol = recent_vol.mean()
                    if avg_vol > 0:
                        strengths['volatility_strength'] = volatility / avg_vol
            
            return strengths
            
        except Exception as e:
            logger.error(f"Error calculating indicator strength: {e}")
            return {}