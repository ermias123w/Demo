import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, replace

from ..config.config import system_config
from ..models.hybrid_models import PredictionResult
from ..data_collection.market_data import market_collector
from ..data_collection.technical_indicators import technical_analyzer

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a prediction"""
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    kelly_criterion: float
    risk_score: float

class RiskManager:
    """Risk management system for crypto predictions"""
    
    def __init__(self):
        self.position_sizes = {}  # Track current position sizes
        self.max_total_exposure = 0.8  # Maximum total portfolio exposure
        self.max_single_position = system_config.MAX_POSITION_SIZE
        self.max_leverage = system_config.MAX_LEVERAGE
        self.portfolio_value = 100000  # Starting portfolio value in USD
        self.performance_history = []
        
    def evaluate_prediction(self, prediction: PredictionResult) -> PredictionResult:
        """Evaluate and adjust prediction based on risk parameters"""
        logger.info(f"Evaluating risk for {prediction.symbol} prediction")
        
        try:
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(prediction)
            
            # Adjust position sizing
            adjusted_prediction = self._adjust_position_size(prediction, risk_metrics)
            
            # Apply volatility-based adjustments
            adjusted_prediction = self._apply_volatility_adjustments(adjusted_prediction, risk_metrics)
            
            # Check portfolio constraints
            adjusted_prediction = self._check_portfolio_constraints(adjusted_prediction)
            
            # Apply final risk filters
            adjusted_prediction = self._apply_risk_filters(adjusted_prediction, risk_metrics)
            
            logger.info(f"Risk evaluation completed for {prediction.symbol}")
            return adjusted_prediction
            
        except Exception as e:
            logger.error(f"Error in risk evaluation: {e}")
            # Return conservative prediction on error
            return self._make_conservative_prediction(prediction)
    
    def _calculate_risk_metrics(self, prediction: PredictionResult) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        symbol = prediction.symbol
        
        # Get historical data for risk calculations
        historical_data = market_collector.get_historical_data(symbol, days=90)
        
        if historical_data.empty:
            logger.warning(f"No historical data for risk calculation: {symbol}")
            return self._default_risk_metrics()
        
        # Calculate returns
        returns = historical_data['price'].pct_change().dropna()
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(365)  # Annualized volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - 0.02/365
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(365)
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(365)
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = returns[returns <= var_95].mean()
        
        # Kelly Criterion
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns < 0].mean()
        
        if avg_loss != 0:
            kelly_criterion = win_rate - ((1 - win_rate) * avg_win / abs(avg_loss))
        else:
            kelly_criterion = 0
        
        # Overall risk score (0-1, higher is riskier)
        risk_score = self._calculate_risk_score(
            volatility, max_drawdown, sharpe_ratio, var_95
        )
        
        return RiskMetrics(
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            kelly_criterion=kelly_criterion,
            risk_score=risk_score
        )
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """Default risk metrics when calculation fails"""
        return RiskMetrics(
            volatility=0.5,
            max_drawdown=-0.2,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            var_95=-0.05,
            expected_shortfall=-0.08,
            kelly_criterion=0.0,
            risk_score=0.8
        )
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, 
                             sharpe_ratio: float, var_95: float) -> float:
        """Calculate overall risk score"""
        # Normalize metrics to 0-1 scale
        vol_score = min(volatility / 2.0, 1.0)  # Cap at 200% volatility
        drawdown_score = min(abs(max_drawdown) / 0.5, 1.0)  # Cap at 50% drawdown
        sharpe_score = max(0, 1 - (sharpe_ratio + 1) / 3)  # Invert Sharpe ratio
        var_score = min(abs(var_95) / 0.2, 1.0)  # Cap at 20% VaR
        
        # Weighted average
        risk_score = (vol_score * 0.3 + drawdown_score * 0.3 + 
                     sharpe_score * 0.2 + var_score * 0.2)
        
        return risk_score
    
    def _adjust_position_size(self, prediction: PredictionResult, 
                             risk_metrics: RiskMetrics) -> PredictionResult:
        """Adjust position size based on risk metrics"""
        
        # Base position size
        base_size = self.max_single_position
        
        # Confidence adjustment
        confidence_multiplier = prediction.confidence
        
        # Volatility adjustment
        volatility_multiplier = max(0.1, 1.0 - risk_metrics.volatility)
        
        # Kelly Criterion adjustment (capped)
        kelly_multiplier = max(0.1, min(1.0, abs(risk_metrics.kelly_criterion)))
        
        # Risk score adjustment
        risk_multiplier = max(0.1, 1.0 - risk_metrics.risk_score)
        
        # Calculate adjusted position size
        adjusted_size = (base_size * confidence_multiplier * 
                        volatility_multiplier * kelly_multiplier * risk_multiplier)
        
        # Ensure minimum and maximum bounds
        adjusted_size = max(0.01, min(adjusted_size, self.max_single_position))
        
        # Adjust leverage based on position size
        if adjusted_size < 0.05:
            adjusted_leverage = min(prediction.leverage, 1.5)
        elif adjusted_size < 0.1:
            adjusted_leverage = min(prediction.leverage, 2.0)
        else:
            adjusted_leverage = prediction.leverage
        
        return replace(prediction, leverage=adjusted_leverage)
    
    def _apply_volatility_adjustments(self, prediction: PredictionResult, 
                                     risk_metrics: RiskMetrics) -> PredictionResult:
        """Apply volatility-based adjustments to stop loss and take profit"""
        
        # Base stop loss and take profit
        base_stop_loss = system_config.DEFAULT_STOP_LOSS
        base_take_profit = system_config.DEFAULT_TAKE_PROFIT
        
        # Volatility adjustment factor
        volatility_factor = max(0.5, min(2.0, risk_metrics.volatility))
        
        # Adjust stop loss and take profit
        adjusted_stop_loss_pct = base_stop_loss * volatility_factor
        adjusted_take_profit_pct = base_take_profit * volatility_factor
        
        # Apply to prediction
        if prediction.signal == 'BUY':
            new_stop_loss = prediction.entry_price * (1 - adjusted_stop_loss_pct)
            new_take_profit = prediction.entry_price * (1 + adjusted_take_profit_pct)
        elif prediction.signal == 'SELL':
            new_stop_loss = prediction.entry_price * (1 + adjusted_stop_loss_pct)
            new_take_profit = prediction.entry_price * (1 - adjusted_take_profit_pct)
        else:
            new_stop_loss = prediction.stop_loss
            new_take_profit = prediction.take_profit
        
        return replace(prediction, stop_loss=new_stop_loss, take_profit=new_take_profit)
    
    def _check_portfolio_constraints(self, prediction: PredictionResult) -> PredictionResult:
        """Check and enforce portfolio-level constraints"""
        
        # Calculate current total exposure
        current_exposure = sum(self.position_sizes.values())
        
        # Estimate new position size (simplified)
        estimated_position_size = 0.1  # This would be calculated based on actual position sizing
        
        # Check if adding this position would exceed maximum exposure
        if current_exposure + estimated_position_size > self.max_total_exposure:
            # Reduce position size or convert to HOLD
            if prediction.confidence > 0.8:
                # Reduce leverage instead of rejecting
                new_leverage = max(1.0, prediction.leverage * 0.5)
                logger.warning(f"Reducing leverage for {prediction.symbol} due to portfolio constraints")
                return replace(prediction, leverage=new_leverage)
            else:
                # Convert to HOLD
                logger.warning(f"Converting {prediction.symbol} to HOLD due to portfolio constraints")
                return replace(prediction, 
                              signal='HOLD',
                              leverage=1.0,
                              rationale=f"{prediction.rationale} + Portfolio constraint")
        
        return prediction
    
    def _apply_risk_filters(self, prediction: PredictionResult, 
                           risk_metrics: RiskMetrics) -> PredictionResult:
        """Apply final risk filters"""
        
        # Extremely high risk filter
        if risk_metrics.risk_score > 0.9:
            logger.warning(f"High risk detected for {prediction.symbol}, converting to HOLD")
            return replace(prediction, 
                          signal='HOLD',
                          leverage=1.0,
                          rationale=f"{prediction.rationale} + High risk")
        
        # Low confidence with high risk
        if prediction.confidence < 0.7 and risk_metrics.risk_score > 0.7:
            logger.warning(f"Low confidence + high risk for {prediction.symbol}, converting to HOLD")
            return replace(prediction, 
                          signal='HOLD',
                          leverage=1.0,
                          rationale=f"{prediction.rationale} + Low confidence + High risk")
        
        # Market volatility filter
        if risk_metrics.volatility > 1.5:  # 150% annualized volatility
            logger.warning(f"Extreme volatility for {prediction.symbol}, reducing leverage")
            return replace(prediction, 
                          leverage=min(prediction.leverage, 1.5),
                          rationale=f"{prediction.rationale} + High volatility")
        
        return prediction
    
    def _make_conservative_prediction(self, prediction: PredictionResult) -> PredictionResult:
        """Create a conservative prediction when risk evaluation fails"""
        return replace(prediction,
                      signal='HOLD',
                      confidence=0.5,
                      leverage=1.0,
                      stop_loss=prediction.entry_price * 0.95,
                      take_profit=prediction.entry_price * 1.05,
                      rationale="Conservative due to risk evaluation error")
    
    def update_position(self, symbol: str, size: float):
        """Update position size for a symbol"""
        self.position_sizes[symbol] = size
        logger.info(f"Updated position size for {symbol}: {size}")
    
    def get_portfolio_risk(self) -> Dict[str, float]:
        """Get current portfolio risk metrics"""
        total_exposure = sum(self.position_sizes.values())
        
        return {
            'total_exposure': total_exposure,
            'max_exposure': self.max_total_exposure,
            'utilization': total_exposure / self.max_total_exposure,
            'available_capacity': self.max_total_exposure - total_exposure,
            'num_positions': len(self.position_sizes),
            'portfolio_value': self.portfolio_value
        }
    
    def calculate_position_size(self, prediction: PredictionResult, 
                              risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size for a prediction"""
        
        # Kelly Criterion based sizing
        kelly_size = abs(risk_metrics.kelly_criterion)
        
        # Volatility adjusted sizing
        volatility_size = 0.1 / max(risk_metrics.volatility, 0.1)
        
        # Confidence based sizing
        confidence_size = prediction.confidence * 0.1
        
        # Take minimum of all sizing methods
        position_size = min(kelly_size, volatility_size, confidence_size)
        
        # Apply maximum position constraint
        position_size = min(position_size, self.max_single_position)
        
        return position_size
    
    def check_correlation_risk(self, symbol: str, existing_positions: List[str]) -> float:
        """Check correlation risk with existing positions"""
        
        if not existing_positions:
            return 0.0
        
        # Get correlation data (simplified - in practice would use actual correlation)
        correlations = {
            'BTC': {'ETH': 0.8, 'BTC': 1.0},
            'ETH': {'BTC': 0.8, 'ETH': 1.0}
        }
        
        max_correlation = 0.0
        for existing_symbol in existing_positions:
            correlation = correlations.get(symbol, {}).get(existing_symbol, 0.0)
            max_correlation = max(max_correlation, abs(correlation))
        
        return max_correlation
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        
        portfolio_risk = self.get_portfolio_risk()
        
        # Calculate recent performance
        recent_performance = self._calculate_recent_performance()
        
        # Risk warnings
        warnings = []
        
        if portfolio_risk['utilization'] > 0.8:
            warnings.append("High portfolio utilization")
        
        if portfolio_risk['num_positions'] > 5:
            warnings.append("High number of positions")
        
        if recent_performance['drawdown'] < -0.1:
            warnings.append("Significant recent drawdown")
        
        return {
            'portfolio_risk': portfolio_risk,
            'recent_performance': recent_performance,
            'warnings': warnings,
            'timestamp': datetime.now()
        }
    
    def _calculate_recent_performance(self) -> Dict:
        """Calculate recent portfolio performance"""
        
        # Simplified performance calculation
        # In practice, this would use actual trade history
        
        return {
            'total_return': 0.05,  # 5% return
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'drawdown': -0.03  # Current drawdown
        }
    
    def adjust_for_market_conditions(self, prediction: PredictionResult) -> PredictionResult:
        """Adjust prediction based on current market conditions"""
        
        try:
            # Get market summary
            market_summary = market_collector.get_market_summary()
            
            # Calculate market stress indicator
            total_volume = market_summary.get('total_volume', 0)
            if total_volume == 0:
                return prediction
            
            # High volume = high stress
            volume_stress = min(total_volume / 1e12, 1.0)  # Normalize
            
            # Adjust confidence based on market stress
            if volume_stress > 0.8:
                adjusted_confidence = prediction.confidence * 0.8
                logger.info(f"Reducing confidence due to high market stress: {prediction.symbol}")
                return replace(prediction, confidence=adjusted_confidence)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error adjusting for market conditions: {e}")
            return prediction
    
    def validate_prediction(self, prediction: PredictionResult) -> bool:
        """Validate prediction meets all risk criteria"""
        
        # Basic validation
        if prediction.confidence < 0.3:
            return False
        
        if prediction.leverage > self.max_leverage:
            return False
        
        if prediction.signal not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        # Risk-based validation
        if prediction.signal != 'HOLD':
            # Check stop loss and take profit are reasonable
            if prediction.signal == 'BUY':
                if prediction.stop_loss >= prediction.entry_price:
                    return False
                if prediction.take_profit <= prediction.entry_price:
                    return False
            elif prediction.signal == 'SELL':
                if prediction.stop_loss <= prediction.entry_price:
                    return False
                if prediction.take_profit >= prediction.entry_price:
                    return False
        
        return True

# Global risk manager instance
risk_manager = RiskManager()