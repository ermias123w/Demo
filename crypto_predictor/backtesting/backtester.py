import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..config.config import system_config
from ..models.hybrid_models import PredictionResult
from ..data_collection.market_data import market_collector
from ..data_collection.technical_indicators import technical_analyzer
from ..data_collection.sentiment_data import news_aggregator

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    signal: str
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    was_stopped: bool
    reason: str

@dataclass
class BacktestResults:
    """Backtest results summary"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades: List[BacktestTrade]
    
class Backtester:
    """Backtesting engine for crypto predictions"""
    
    def __init__(self):
        self.initial_capital = 100000  # Starting capital
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        
    def run_backtest(self, symbol: str, 
                    historical_data: pd.DataFrame, 
                    start_date: datetime = None,
                    end_date: datetime = None) -> BacktestResults:
        """Run comprehensive backtest on historical data"""
        
        logger.info(f"Starting backtest for {symbol}")
        
        try:
            # Prepare data
            if start_date:
                historical_data = historical_data[historical_data.index >= start_date]
            if end_date:
                historical_data = historical_data[historical_data.index <= end_date]
            
            if len(historical_data) < 100:
                logger.warning(f"Insufficient data for backtest: {len(historical_data)} points")
                return self._empty_backtest_results()
            
            # Generate predictions for historical data
            predictions = self._generate_historical_predictions(symbol, historical_data)
            
            if not predictions:
                logger.warning(f"No predictions generated for {symbol}")
                return self._empty_backtest_results()
            
            # Execute backtest
            trades = self._execute_backtest_trades(predictions, historical_data)
            
            # Calculate results
            results = self._calculate_backtest_results(trades)
            
            logger.info(f"Backtest completed for {symbol}: {results.total_trades} trades, "
                       f"{results.win_rate:.2%} win rate, {results.total_return:.2%} return")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return self._empty_backtest_results()
    
    def _generate_historical_predictions(self, symbol: str, 
                                       historical_data: pd.DataFrame) -> List[PredictionResult]:
        """Generate predictions for historical data"""
        
        predictions = []
        lookback_days = system_config.LSTM_LOOKBACK_DAYS
        
        # Generate predictions at regular intervals
        for i in range(lookback_days, len(historical_data), 5):  # Every 5 periods
            
            # Get data up to current point
            current_data = historical_data.iloc[:i]
            current_price = current_data['price'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Calculate technical indicators
            technical_indicators = technical_analyzer.calculate_indicators(current_data, symbol)
            
            if not technical_indicators:
                continue
            
            # Simplified prediction generation (in practice would use actual model)
            prediction = self._generate_mock_prediction(
                symbol, current_time, current_price, technical_indicators[-1]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _generate_mock_prediction(self, symbol: str, timestamp: datetime, 
                                 price: float, technical_indicator) -> PredictionResult:
        """Generate mock prediction for backtesting"""
        
        # Simple signal generation based on RSI
        if technical_indicator.rsi < 30:
            signal = 'BUY'
            confidence = 0.7
        elif technical_indicator.rsi > 70:
            signal = 'SELL'
            confidence = 0.7
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # Calculate stop loss and take profit
        if signal == 'BUY':
            stop_loss = price * 0.95
            take_profit = price * 1.08
        elif signal == 'SELL':
            stop_loss = price * 1.05
            take_profit = price * 0.92
        else:
            stop_loss = price * 0.98
            take_profit = price * 1.02
        
        return PredictionResult(
            symbol=symbol,
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=1.5 if signal != 'HOLD' else 1.0,
            rationale=f"RSI-based signal: {technical_indicator.rsi:.1f}",
            model_outputs={},
            technical_features=np.array([technical_indicator.rsi]),
            sentiment_score=0.0
        )
    
    def _execute_backtest_trades(self, predictions: List[PredictionResult], 
                               historical_data: pd.DataFrame) -> List[BacktestTrade]:
        """Execute trades based on predictions"""
        
        trades = []
        open_positions = {}
        
        for prediction in predictions:
            if prediction.signal == 'HOLD':
                continue
            
            # Check if we can enter this trade
            if prediction.symbol in open_positions:
                continue  # Already have position
            
            # Calculate position size
            position_size = self._calculate_position_size(prediction)
            
            # Enter trade
            trade = self._enter_trade(prediction, position_size, historical_data)
            if trade:
                open_positions[prediction.symbol] = trade
        
        # Close any remaining open positions
        for symbol, trade in open_positions.items():
            if trade.exit_time is None:
                # Close at last available price
                last_price = historical_data['price'].iloc[-1]
                last_time = historical_data.index[-1]
                trade = self._close_trade(trade, last_price, last_time, "End of backtest")
            
            trades.append(trade)
        
        return trades
    
    def _calculate_position_size(self, prediction: PredictionResult) -> float:
        """Calculate position size for backtest"""
        
        # Risk-based position sizing
        risk_per_trade = 0.02  # 2% risk per trade
        
        if prediction.signal == 'BUY':
            risk_amount = prediction.entry_price - prediction.stop_loss
        elif prediction.signal == 'SELL':
            risk_amount = prediction.stop_loss - prediction.entry_price
        else:
            risk_amount = prediction.entry_price * 0.02
        
        if risk_amount <= 0:
            return 0
        
        # Position size based on risk
        position_value = self.initial_capital * risk_per_trade / (risk_amount / prediction.entry_price)
        
        # Apply leverage
        position_value *= prediction.leverage
        
        # Convert to quantity
        quantity = position_value / prediction.entry_price
        
        return quantity
    
    def _enter_trade(self, prediction: PredictionResult, quantity: float, 
                    historical_data: pd.DataFrame) -> Optional[BacktestTrade]:
        """Enter a trade"""
        
        if quantity <= 0:
            return None
        
        # Apply slippage
        if prediction.signal == 'BUY':
            entry_price = prediction.entry_price * (1 + self.slippage)
        else:
            entry_price = prediction.entry_price * (1 - self.slippage)
        
        # Create trade
        trade = BacktestTrade(
            symbol=prediction.symbol,
            entry_time=prediction.timestamp,
            exit_time=None,
            entry_price=entry_price,
            exit_price=0,
            signal=prediction.signal,
            quantity=quantity,
            leverage=prediction.leverage,
            stop_loss=prediction.stop_loss,
            take_profit=prediction.take_profit,
            pnl=0,
            pnl_percent=0,
            was_stopped=False,
            reason=""
        )
        
        # Monitor trade for exit conditions
        trade = self._monitor_trade(trade, historical_data)
        
        return trade
    
    def _monitor_trade(self, trade: BacktestTrade, 
                      historical_data: pd.DataFrame) -> BacktestTrade:
        """Monitor trade for exit conditions"""
        
        # Get price data after entry
        future_data = historical_data[historical_data.index > trade.entry_time]
        
        if future_data.empty:
            return trade
        
        for timestamp, row in future_data.iterrows():
            current_price = row['price']
            
            # Check stop loss
            if trade.signal == 'BUY' and current_price <= trade.stop_loss:
                return self._close_trade(trade, current_price, timestamp, "Stop loss")
            elif trade.signal == 'SELL' and current_price >= trade.stop_loss:
                return self._close_trade(trade, current_price, timestamp, "Stop loss")
            
            # Check take profit
            if trade.signal == 'BUY' and current_price >= trade.take_profit:
                return self._close_trade(trade, current_price, timestamp, "Take profit")
            elif trade.signal == 'SELL' and current_price <= trade.take_profit:
                return self._close_trade(trade, current_price, timestamp, "Take profit")
            
            # Check time-based exit (max 24 hours)
            if (timestamp - trade.entry_time).total_seconds() > 86400:
                return self._close_trade(trade, current_price, timestamp, "Time exit")
        
        return trade
    
    def _close_trade(self, trade: BacktestTrade, exit_price: float, 
                    exit_time: datetime, reason: str) -> BacktestTrade:
        """Close a trade"""
        
        # Apply slippage
        if trade.signal == 'BUY':
            exit_price = exit_price * (1 - self.slippage)
        else:
            exit_price = exit_price * (1 + self.slippage)
        
        # Calculate PnL
        if trade.signal == 'BUY':
            pnl_percent = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_percent = (trade.entry_price - exit_price) / trade.entry_price
        
        # Apply leverage
        pnl_percent *= trade.leverage
        
        # Calculate dollar PnL
        position_value = trade.quantity * trade.entry_price
        pnl = position_value * pnl_percent
        
        # Apply commission
        commission_cost = position_value * self.commission * 2  # Entry + exit
        pnl -= commission_cost
        
        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.was_stopped = reason in ["Stop loss"]
        trade.reason = reason
        
        return trade
    
    def _calculate_backtest_results(self, trades: List[BacktestTrade]) -> BacktestResults:
        """Calculate backtest results"""
        
        if not trades:
            return self._empty_backtest_results()
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in trades)
        total_return = total_pnl / self.initial_capital
        
        # Win/Loss statistics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) / self.initial_capital if len(drawdown) > 0 else 0
        
        # Sharpe ratio
        trade_returns = [t.pnl / self.initial_capital for t in trades]
        sharpe_ratio = self._calculate_sharpe_ratio(trade_returns)
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(trade_returns)
        
        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(trades, lambda t: t.pnl > 0)
        max_consecutive_losses = self._max_consecutive(trades, lambda t: t.pnl < 0)
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=trades
        )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assuming daily returns, annualize
        sharpe = mean_return / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0
        
        # Annualize
        sortino = mean_return / downside_std * np.sqrt(252)
        return sortino
    
    def _max_consecutive(self, trades: List[BacktestTrade], condition) -> int:
        """Calculate maximum consecutive occurrences"""
        if not trades:
            return 0
        
        max_count = 0
        current_count = 0
        
        for trade in trades:
            if condition(trade):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _empty_backtest_results(self) -> BacktestResults:
        """Return empty backtest results"""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_return=0,
            max_drawdown=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            trades=[]
        )
    
    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        """Evaluate recent predictions against actual outcomes"""
        
        logger.info(f"Evaluating {len(predictions)} predictions")
        
        evaluated_predictions = 0
        correct_predictions = 0
        total_profit = 0
        
        for prediction in predictions:
            try:
                # Get actual outcome
                outcome = self._get_actual_outcome(prediction)
                
                if outcome is not None:
                    evaluated_predictions += 1
                    
                    # Check if prediction was correct
                    if self._is_prediction_correct(prediction, outcome):
                        correct_predictions += 1
                    
                    # Calculate profit/loss
                    profit = self._calculate_prediction_profit(prediction, outcome)
                    total_profit += profit
                    
            except Exception as e:
                logger.error(f"Error evaluating prediction: {e}")
        
        # Calculate metrics
        win_rate = correct_predictions / evaluated_predictions if evaluated_predictions > 0 else 0
        
        return {
            'evaluated_predictions': evaluated_predictions,
            'correct_predictions': correct_predictions,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_prediction': total_profit / evaluated_predictions if evaluated_predictions > 0 else 0
        }
    
    def _get_actual_outcome(self, prediction: Dict) -> Optional[Dict]:
        """Get actual market outcome for a prediction"""
        
        symbol = prediction['symbol']
        prediction_time = datetime.fromisoformat(prediction['timestamp'])
        
        # Get price data 24 hours after prediction
        end_time = prediction_time + timedelta(hours=24)
        
        try:
            historical_data = market_collector.get_historical_data(symbol, days=2)
            
            if historical_data.empty:
                return None
            
            # Find price at prediction time and 24 hours later
            prediction_price = prediction['entry_price']
            
            # Get price closest to end time
            future_data = historical_data[historical_data.index > prediction_time]
            if future_data.empty:
                return None
            
            end_price = future_data['price'].iloc[-1]
            
            return {
                'start_price': prediction_price,
                'end_price': end_price,
                'price_change': (end_price - prediction_price) / prediction_price,
                'actual_signal': self._determine_actual_signal(prediction_price, end_price)
            }
            
        except Exception as e:
            logger.error(f"Error getting actual outcome: {e}")
            return None
    
    def _determine_actual_signal(self, start_price: float, end_price: float) -> str:
        """Determine actual signal based on price movement"""
        
        price_change = (end_price - start_price) / start_price
        
        if price_change > 0.01:  # 1% threshold
            return 'BUY'
        elif price_change < -0.01:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _is_prediction_correct(self, prediction: Dict, outcome: Dict) -> bool:
        """Check if prediction was correct"""
        
        predicted_signal = prediction['signal']
        actual_signal = outcome['actual_signal']
        
        return predicted_signal == actual_signal
    
    def _calculate_prediction_profit(self, prediction: Dict, outcome: Dict) -> float:
        """Calculate profit/loss from prediction"""
        
        signal = prediction['signal']
        price_change = outcome['price_change']
        leverage = prediction.get('leverage', 1.0)
        
        if signal == 'BUY':
            profit = price_change * leverage
        elif signal == 'SELL':
            profit = -price_change * leverage
        else:  # HOLD
            profit = 0
        
        # Apply commission
        profit -= self.commission * 2  # Entry + exit
        
        return profit
    
    def generate_performance_report(self, backtest_results: BacktestResults) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'summary': {
                'total_trades': backtest_results.total_trades,
                'win_rate': backtest_results.win_rate,
                'total_return': backtest_results.total_return,
                'max_drawdown': backtest_results.max_drawdown,
                'sharpe_ratio': backtest_results.sharpe_ratio,
                'profit_factor': backtest_results.profit_factor
            },
            'trade_statistics': {
                'avg_win': backtest_results.avg_win,
                'avg_loss': backtest_results.avg_loss,
                'max_consecutive_wins': backtest_results.max_consecutive_wins,
                'max_consecutive_losses': backtest_results.max_consecutive_losses
            },
            'risk_metrics': {
                'max_drawdown': backtest_results.max_drawdown,
                'sortino_ratio': backtest_results.sortino_ratio,
                'volatility': np.std([t.pnl_percent for t in backtest_results.trades])
            },
            'trade_analysis': self._analyze_trades(backtest_results.trades)
        }
        
        return report
    
    def _analyze_trades(self, trades: List[BacktestTrade]) -> Dict:
        """Analyze trade patterns"""
        
        if not trades:
            return {}
        
        # Signal analysis
        buy_trades = [t for t in trades if t.signal == 'BUY']
        sell_trades = [t for t in trades if t.signal == 'SELL']
        
        # Time analysis
        trade_durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 
                          for t in trades if t.exit_time]
        
        # Stop loss analysis
        stopped_trades = [t for t in trades if t.was_stopped]
        
        return {
            'buy_trades': {
                'count': len(buy_trades),
                'win_rate': sum(1 for t in buy_trades if t.pnl > 0) / len(buy_trades) if buy_trades else 0,
                'avg_pnl': np.mean([t.pnl for t in buy_trades]) if buy_trades else 0
            },
            'sell_trades': {
                'count': len(sell_trades),
                'win_rate': sum(1 for t in sell_trades if t.pnl > 0) / len(sell_trades) if sell_trades else 0,
                'avg_pnl': np.mean([t.pnl for t in sell_trades]) if sell_trades else 0
            },
            'timing': {
                'avg_duration_hours': np.mean(trade_durations) if trade_durations else 0,
                'max_duration_hours': np.max(trade_durations) if trade_durations else 0,
                'min_duration_hours': np.min(trade_durations) if trade_durations else 0
            },
            'risk_management': {
                'stopped_trades': len(stopped_trades),
                'stop_loss_rate': len(stopped_trades) / len(trades) if trades else 0
            }
        }

# Global backtester instance
backtester = Backtester()