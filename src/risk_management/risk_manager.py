import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management system for crypto trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk parameters
        self.max_leverage = config.get('MAX_LEVERAGE', 5.0)
        self.min_leverage = config.get('MIN_LEVERAGE', 1.0)
        self.base_stop_loss = config.get('BASE_STOP_LOSS', 0.02)  # 2%
        self.base_take_profit = config.get('BASE_TAKE_PROFIT', 0.04)  # 4%
        self.volatility_multiplier = config.get('VOLATILITY_MULTIPLIER', 1.5)
        self.max_position_size = config.get('MAX_POSITION_SIZE', 0.1)  # 10% of portfolio
        self.confidence_scaling = config.get('CONFIDENCE_SCALING', True)
        self.min_confidence_for_leverage = config.get('MIN_CONFIDENCE_FOR_LEVERAGE', 0.8)
        
        # Risk metrics
        self.portfolio_value = 10000.0  # Default portfolio value
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_drawdown = 0.20  # 20% max drawdown
        
        logger.info("RiskManager initialized with advanced risk controls")
    
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate historical volatility"""
        try:
            if len(prices) < window:
                return 0.02  # Default 2% volatility
            
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=window).std().iloc[-1]
            
            # Annualize volatility
            annualized_volatility = volatility * np.sqrt(365)
            
            return max(0.01, min(0.5, annualized_volatility))  # Clamp between 1% and 50%
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def calculate_atr_percentage(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range as percentage of price"""
        try:
            if len(df) < period or 'atr' not in df.columns:
                return 0.02  # Default 2% ATR
            
            latest_price = df['price'].iloc[-1]
            latest_atr = df['atr'].iloc[-1]
            
            if pd.isna(latest_atr) or latest_price <= 0:
                return 0.02
            
            atr_percentage = latest_atr / latest_price
            return max(0.005, min(0.1, atr_percentage))  # Clamp between 0.5% and 10%
            
        except Exception as e:
            logger.error(f"Error calculating ATR percentage: {e}")
            return 0.02
    
    def calculate_position_size(self, confidence: float, volatility: float, 
                               portfolio_value: float = None) -> float:
        """Calculate position size based on confidence and volatility"""
        try:
            portfolio_value = portfolio_value or self.portfolio_value
            
            # Base position size
            base_size = self.max_position_size
            
            # Adjust for confidence
            confidence_adjustment = confidence if self.confidence_scaling else 1.0
            
            # Adjust for volatility (higher volatility = smaller position)
            volatility_adjustment = 1.0 / (1.0 + volatility * self.volatility_multiplier)
            
            # Calculate final position size
            position_size = base_size * confidence_adjustment * volatility_adjustment
            
            # Ensure minimum and maximum bounds
            position_size = max(0.01, min(self.max_position_size, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.05  # Default 5% position size
    
    def calculate_stop_loss(self, entry_price: float, volatility: float, 
                           atr_percentage: float, confidence: float) -> float:
        """Calculate stop loss based on volatility and confidence"""
        try:
            # Base stop loss
            base_stop = self.base_stop_loss
            
            # Adjust for volatility
            volatility_adjustment = max(1.0, volatility * self.volatility_multiplier)
            
            # Adjust for ATR
            atr_adjustment = max(1.0, atr_percentage / 0.01)  # Relative to 1% ATR
            
            # Adjust for confidence (higher confidence = tighter stop)
            confidence_adjustment = max(0.5, 1.0 - (confidence - 0.5) * 0.5)
            
            # Calculate final stop loss percentage
            stop_loss_percentage = base_stop * volatility_adjustment * atr_adjustment * confidence_adjustment
            
            # Clamp between 0.5% and 15%
            stop_loss_percentage = max(0.005, min(0.15, stop_loss_percentage))
            
            # Calculate stop loss price
            stop_loss_price = entry_price * (1.0 - stop_loss_percentage)
            
            return stop_loss_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.98  # Default 2% stop loss
    
    def calculate_take_profit(self, entry_price: float, volatility: float, 
                             confidence: float, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio"""
        try:
            # Calculate stop loss distance
            stop_loss_price = self.calculate_stop_loss(entry_price, volatility, 0.02, confidence)
            stop_loss_distance = entry_price - stop_loss_price
            
            # Calculate take profit distance using risk-reward ratio
            take_profit_distance = stop_loss_distance * risk_reward_ratio
            
            # Adjust for confidence (higher confidence = more aggressive target)
            confidence_adjustment = 1.0 + (confidence - 0.5) * 0.5
            take_profit_distance *= confidence_adjustment
            
            # Calculate take profit price
            take_profit_price = entry_price + take_profit_distance
            
            return take_profit_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price * 1.04  # Default 4% take profit
    
    def calculate_leverage(self, confidence: float, volatility: float) -> float:
        """Calculate appropriate leverage based on confidence and volatility"""
        try:
            # Only use leverage for high confidence trades
            if confidence < self.min_confidence_for_leverage:
                return 1.0
            
            # Base leverage calculation
            base_leverage = self.min_leverage + (confidence - 0.5) * 2.0 * (self.max_leverage - self.min_leverage)
            
            # Adjust for volatility (higher volatility = lower leverage)
            volatility_adjustment = 1.0 / (1.0 + volatility * 2.0)
            
            leverage = base_leverage * volatility_adjustment
            
            # Clamp between min and max leverage
            leverage = max(self.min_leverage, min(self.max_leverage, leverage))
            
            return round(leverage, 1)
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            return 1.0  # Default no leverage
    
    def calculate_entry_range(self, current_price: float, volatility: float, 
                             confidence: float) -> Tuple[float, float]:
        """Calculate entry price range"""
        try:
            # Base range is 0.5% around current price
            base_range = 0.005
            
            # Adjust for volatility
            volatility_adjustment = max(1.0, volatility * 2.0)
            
            # Adjust for confidence (higher confidence = tighter range)
            confidence_adjustment = max(0.5, 1.0 - (confidence - 0.5) * 0.3)
            
            # Calculate range
            range_percentage = base_range * volatility_adjustment * confidence_adjustment
            range_percentage = max(0.002, min(0.02, range_percentage))  # Clamp between 0.2% and 2%
            
            entry_min = current_price * (1.0 - range_percentage)
            entry_max = current_price * (1.0 + range_percentage)
            
            return entry_min, entry_max
            
        except Exception as e:
            logger.error(f"Error calculating entry range: {e}")
            return current_price * 0.995, current_price * 1.005  # Default 0.5% range
    
    def calculate_risk_parameters(self, current_price: float, confidence: float, 
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all risk parameters for a trade"""
        try:
            # Calculate volatility and ATR
            volatility = self.calculate_volatility(price_data['price'])
            atr_percentage = self.calculate_atr_percentage(price_data)
            
            # Calculate position size
            position_size = self.calculate_position_size(confidence, volatility)
            
            # Calculate entry range
            entry_min, entry_max = self.calculate_entry_range(current_price, volatility, confidence)
            
            # Calculate stop loss
            stop_loss = self.calculate_stop_loss(current_price, volatility, atr_percentage, confidence)
            
            # Calculate take profit
            take_profit = self.calculate_take_profit(current_price, volatility, confidence)
            
            # Calculate leverage
            leverage = self.calculate_leverage(confidence, volatility)
            
            # Calculate risk-reward ratio
            risk_amount = current_price - stop_loss
            reward_amount = take_profit - current_price
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Calculate maximum loss
            max_loss_percentage = (current_price - stop_loss) / current_price
            max_loss_amount = self.portfolio_value * position_size * max_loss_percentage * leverage
            
            return {
                'position_size': position_size,
                'entry_range_min': entry_min,
                'entry_range_max': entry_max,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'risk_reward_ratio': risk_reward_ratio,
                'max_loss_amount': max_loss_amount,
                'max_loss_percentage': max_loss_percentage,
                'volatility': volatility,
                'atr_percentage': atr_percentage
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
            return {
                'position_size': 0.05,
                'entry_range_min': current_price * 0.995,
                'entry_range_max': current_price * 1.005,
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.04,
                'leverage': 1.0,
                'risk_reward_ratio': 2.0,
                'max_loss_amount': 100.0,
                'max_loss_percentage': 0.02,
                'volatility': 0.02,
                'atr_percentage': 0.02
            }
    
    def validate_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters against risk limits"""
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Check position size
            if trade_params['position_size'] > self.max_position_size:
                validation_result['errors'].append(f"Position size {trade_params['position_size']:.2%} exceeds maximum {self.max_position_size:.2%}")
                validation_result['is_valid'] = False
            
            # Check leverage
            if trade_params['leverage'] > self.max_leverage:
                validation_result['errors'].append(f"Leverage {trade_params['leverage']} exceeds maximum {self.max_leverage}")
                validation_result['is_valid'] = False
            
            # Check risk-reward ratio
            if trade_params['risk_reward_ratio'] < 1.0:
                validation_result['warnings'].append(f"Risk-reward ratio {trade_params['risk_reward_ratio']:.2f} is below 1.0")
            
            # Check maximum loss
            if trade_params['max_loss_amount'] > self.portfolio_value * self.max_daily_loss:
                validation_result['errors'].append(f"Maximum loss ${trade_params['max_loss_amount']:.2f} exceeds daily limit")
                validation_result['is_valid'] = False
            
            # Check volatility
            if trade_params['volatility'] > 0.3:
                validation_result['warnings'].append(f"High volatility {trade_params['volatility']:.2%} detected")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def calculate_portfolio_risk(self, active_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall portfolio risk"""
        try:
            if not active_positions:
                return {
                    'total_exposure': 0.0,
                    'total_risk': 0.0,
                    'concentration_risk': 0.0,
                    'leverage_risk': 0.0,
                    'correlation_risk': 0.0
                }
            
            # Calculate total exposure
            total_exposure = sum(pos['position_size'] * pos.get('leverage', 1.0) for pos in active_positions)
            
            # Calculate total risk (maximum possible loss)
            total_risk = sum(pos['max_loss_amount'] for pos in active_positions)
            
            # Calculate concentration risk (largest position as % of portfolio)
            position_sizes = [pos['position_size'] for pos in active_positions]
            concentration_risk = max(position_sizes) if position_sizes else 0.0
            
            # Calculate leverage risk (weighted average leverage)
            total_position_value = sum(pos['position_size'] for pos in active_positions)
            if total_position_value > 0:
                leverage_risk = sum(pos['position_size'] * pos.get('leverage', 1.0) for pos in active_positions) / total_position_value
            else:
                leverage_risk = 1.0
            
            # Correlation risk (simplified - assume some correlation between crypto positions)
            correlation_risk = min(1.0, len(active_positions) * 0.3)
            
            portfolio_risk = {
                'total_exposure': total_exposure,
                'total_risk': total_risk,
                'concentration_risk': concentration_risk,
                'leverage_risk': leverage_risk,
                'correlation_risk': correlation_risk,
                'risk_score': self._calculate_risk_score(total_exposure, total_risk, concentration_risk, leverage_risk)
            }
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {
                'total_exposure': 0.0,
                'total_risk': 0.0,
                'concentration_risk': 0.0,
                'leverage_risk': 0.0,
                'correlation_risk': 0.0,
                'risk_score': 0.0
            }
    
    def _calculate_risk_score(self, exposure: float, risk: float, 
                             concentration: float, leverage: float) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Normalize each component
            exposure_score = min(100, exposure * 100)
            risk_score = min(100, (risk / self.portfolio_value) * 1000)
            concentration_score = min(100, concentration * 1000)
            leverage_score = min(100, (leverage - 1) * 25)
            
            # Weighted average
            overall_score = (exposure_score * 0.3 + risk_score * 0.4 + 
                           concentration_score * 0.2 + leverage_score * 0.1)
            
            return min(100, overall_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.0
    
    def get_risk_recommendations(self, trade_params: Dict[str, Any]) -> List[str]:
        """Get risk management recommendations"""
        try:
            recommendations = []
            
            # Position size recommendations
            if trade_params['position_size'] > 0.15:
                recommendations.append("Consider reducing position size - high exposure detected")
            
            # Leverage recommendations
            if trade_params['leverage'] > 3.0:
                recommendations.append("High leverage detected - consider reducing for better risk management")
            
            # Risk-reward recommendations
            if trade_params['risk_reward_ratio'] < 1.5:
                recommendations.append("Consider improving risk-reward ratio by adjusting targets")
            
            # Volatility recommendations
            if trade_params['volatility'] > 0.2:
                recommendations.append("High volatility environment - consider tighter stops and smaller positions")
            
            # Stop loss recommendations
            if trade_params['max_loss_percentage'] > 0.05:
                recommendations.append("Stop loss is wide - consider tighter risk management")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return []
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value for risk calculations"""
        try:
            if new_value > 0:
                self.portfolio_value = new_value
                logger.info(f"Portfolio value updated to ${new_value:.2f}")
            else:
                logger.warning("Invalid portfolio value provided")
                
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk management settings"""
        try:
            return {
                'max_leverage': self.max_leverage,
                'base_stop_loss': self.base_stop_loss,
                'base_take_profit': self.base_take_profit,
                'max_position_size': self.max_position_size,
                'portfolio_value': self.portfolio_value,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'confidence_scaling': self.confidence_scaling,
                'min_confidence_for_leverage': self.min_confidence_for_leverage
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}