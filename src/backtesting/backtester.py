import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class Backtester:
    """Comprehensive backtesting system for crypto trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Backtesting parameters
        self.initial_capital = config.get('INITIAL_CAPITAL', 10000.0)
        self.commission = config.get('COMMISSION', 0.001)  # 0.1%
        self.slippage = config.get('SLIPPAGE', 0.0005)  # 0.05%
        self.min_win_rate = config.get('MIN_WIN_RATE', 0.55)
        self.min_accuracy = config.get('MIN_ACCURACY', 0.60)
        self.disable_threshold = config.get('DISABLE_THRESHOLD', 0.45)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
        
        logger.info("Backtester initialized with comprehensive performance analysis")
    
    def simulate_trade(self, entry_price: float, exit_price: float, 
                      prediction_type: str, position_size: float, 
                      leverage: float = 1.0) -> Dict[str, Any]:
        """Simulate a single trade"""
        try:
            # Calculate trade direction
            is_long = prediction_type.upper() == 'BUY'
            
            # Calculate slippage
            entry_slippage = self.slippage if is_long else -self.slippage
            exit_slippage = -self.slippage if is_long else self.slippage
            
            actual_entry_price = entry_price * (1 + entry_slippage)
            actual_exit_price = exit_price * (1 + exit_slippage)
            
            # Calculate position value
            position_value = self.initial_capital * position_size
            
            # Calculate quantity
            quantity = position_value / actual_entry_price
            
            # Calculate P&L
            if is_long:
                price_change = actual_exit_price - actual_entry_price
            else:
                price_change = actual_entry_price - actual_exit_price
            
            gross_pnl = quantity * price_change * leverage
            
            # Calculate commissions
            entry_commission = position_value * self.commission
            exit_commission = position_value * self.commission
            total_commission = entry_commission + exit_commission
            
            # Calculate net P&L
            net_pnl = gross_pnl - total_commission
            
            # Calculate returns
            return_percentage = net_pnl / position_value
            
            # Create trade record
            trade_record = {
                'entry_price': actual_entry_price,
                'exit_price': actual_exit_price,
                'prediction_type': prediction_type,
                'position_size': position_size,
                'leverage': leverage,
                'quantity': quantity,
                'gross_pnl': gross_pnl,
                'commission': total_commission,
                'net_pnl': net_pnl,
                'return_percentage': return_percentage,
                'is_profitable': net_pnl > 0,
                'is_long': is_long
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'prediction_type': prediction_type,
                'position_size': position_size,
                'leverage': leverage,
                'quantity': 0,
                'gross_pnl': 0,
                'commission': 0,
                'net_pnl': 0,
                'return_percentage': 0,
                'is_profitable': False,
                'is_long': prediction_type.upper() == 'BUY'
            }
    
    def run_backtest(self, price_data: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive backtest on historical data"""
        try:
            logger.info("Starting backtest simulation...")
            
            # Reset state
            self.trades = []
            self.equity_curve = []
            current_capital = self.initial_capital
            
            # Ensure data is sorted by timestamp
            price_data = price_data.sort_values('timestamp').reset_index(drop=True)
            predictions = predictions.sort_values('timestamp').reset_index(drop=True)
            
            # Merge predictions with price data
            merged_data = pd.merge_asof(
                predictions.sort_values('timestamp'),
                price_data[['timestamp', 'price']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            if merged_data.empty:
                logger.warning("No matching data found for backtest")
                return {}
            
            # Process each prediction
            for idx, prediction in merged_data.iterrows():
                try:
                    # Skip HOLD predictions
                    if prediction.get('prediction_type', 'HOLD').upper() == 'HOLD':
                        continue
                    
                    # Get entry price
                    entry_price = prediction['price']
                    
                    # Find exit price (24 hours later)
                    entry_time = pd.to_datetime(prediction['timestamp'])
                    exit_time = entry_time + timedelta(hours=24)
                    
                    # Find closest price data for exit
                    exit_data = price_data[price_data['timestamp'] >= exit_time]
                    
                    if exit_data.empty:
                        continue
                    
                    exit_price = exit_data.iloc[0]['price']
                    
                    # Check if stop loss or take profit was hit
                    stop_loss = prediction.get('stop_loss', entry_price * 0.98)
                    take_profit = prediction.get('take_profit', entry_price * 1.02)
                    
                    # Get price data between entry and exit
                    intermediate_data = price_data[
                        (price_data['timestamp'] >= entry_time) & 
                        (price_data['timestamp'] <= exit_time)
                    ]
                    
                    actual_exit_price = self._calculate_actual_exit_price(
                        intermediate_data, entry_price, exit_price, 
                        stop_loss, take_profit, prediction['prediction_type']
                    )
                    
                    # Simulate trade
                    trade_result = self.simulate_trade(
                        entry_price=entry_price,
                        exit_price=actual_exit_price,
                        prediction_type=prediction['prediction_type'],
                        position_size=prediction.get('position_size', 0.05),
                        leverage=prediction.get('leverage', 1.0)
                    )
                    
                    # Add metadata
                    trade_result.update({
                        'prediction_id': prediction.get('prediction_id'),
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'symbol': prediction.get('symbol', 'BTC'),
                        'confidence': prediction.get('confidence', 0.5),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'hit_stop_loss': actual_exit_price <= stop_loss if prediction['prediction_type'] == 'BUY' else actual_exit_price >= stop_loss,
                        'hit_take_profit': actual_exit_price >= take_profit if prediction['prediction_type'] == 'BUY' else actual_exit_price <= take_profit
                    })
                    
                    self.trades.append(trade_result)
                    
                    # Update capital
                    current_capital += trade_result['net_pnl']
                    
                    # Record equity curve
                    self.equity_curve.append({
                        'timestamp': exit_time,
                        'capital': current_capital,
                        'trade_pnl': trade_result['net_pnl'],
                        'cumulative_return': (current_capital - self.initial_capital) / self.initial_capital
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing prediction {idx}: {e}")
                    continue
            
            # Calculate performance metrics
            self.metrics = self._calculate_performance_metrics()
            
            logger.info(f"Backtest completed. {len(self.trades)} trades executed. "
                       f"Total return: {self.metrics.get('total_return', 0):.2f}%")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _calculate_actual_exit_price(self, price_data: pd.DataFrame, entry_price: float, 
                                    final_exit_price: float, stop_loss: float, 
                                    take_profit: float, prediction_type: str) -> float:
        """Calculate actual exit price considering stop loss and take profit"""
        try:
            if price_data.empty:
                return final_exit_price
            
            is_long = prediction_type.upper() == 'BUY'
            
            for _, row in price_data.iterrows():
                current_price = row['price']
                
                if is_long:
                    # Long position
                    if current_price <= stop_loss:
                        return stop_loss
                    elif current_price >= take_profit:
                        return take_profit
                else:
                    # Short position
                    if current_price >= stop_loss:
                        return stop_loss
                    elif current_price <= take_profit:
                        return take_profit
            
            return final_exit_price
            
        except Exception as e:
            logger.error(f"Error calculating actual exit price: {e}")
            return final_exit_price
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades:
                return {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'losing_trades': 0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0
                }
            
            # Basic metrics
            total_trades = len(self.trades)
            profitable_trades = sum(1 for trade in self.trades if trade['is_profitable'])
            losing_trades = total_trades - profitable_trades
            
            # P&L metrics
            total_pnl = sum(trade['net_pnl'] for trade in self.trades)
            total_return = (total_pnl / self.initial_capital) * 100
            
            # Win/Loss metrics
            wins = [trade['net_pnl'] for trade in self.trades if trade['is_profitable']]
            losses = [trade['net_pnl'] for trade in self.trades if not trade['is_profitable']]
            
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            
            average_win = np.mean(wins) if wins else 0
            average_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Risk metrics
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
            
            # Drawdown calculation
            max_drawdown = self._calculate_max_drawdown()
            
            # Sharpe ratio
            returns = [trade['return_percentage'] for trade in self.trades]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Calmar ratio
            calmar_ratio = (total_return / 100) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks()
            
            # Additional metrics
            average_trade_return = np.mean(returns) * 100 if returns else 0
            volatility = np.std(returns) * 100 if returns else 0
            
            # Time-based metrics
            if self.equity_curve:
                total_days = (self.equity_curve[-1]['timestamp'] - self.equity_curve[0]['timestamp']).days
                annual_return = (total_return / 100) * (365 / total_days) if total_days > 0 else 0
            else:
                annual_return = 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return * 100,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown * 100,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'average_win': average_win,
                'average_loss': average_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'average_trade_return': average_trade_return,
                'volatility': volatility,
                'total_commission': sum(trade['commission'] for trade in self.trades),
                'total_pnl': total_pnl,
                'final_capital': self.initial_capital + total_pnl
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.equity_curve:
                return 0.0
            
            peak = self.initial_capital
            max_drawdown = 0.0
            
            for point in self.equity_curve:
                current_capital = point['capital']
                
                if current_capital > peak:
                    peak = current_capital
                
                drawdown = (peak - current_capital) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            excess_returns = [r - risk_free_rate for r in returns]
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            
            if std_excess_return == 0:
                return 0.0
            
            return mean_excess_return / std_excess_return
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            excess_returns = [r - risk_free_rate for r in returns]
            mean_excess_return = np.mean(excess_returns)
            
            # Calculate downside deviation
            downside_returns = [r for r in excess_returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            
            if downside_deviation == 0:
                return 0.0
            
            return mean_excess_return / downside_deviation
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_consecutive_streaks(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        try:
            if not self.trades:
                return 0, 0
            
            max_wins = 0
            max_losses = 0
            current_wins = 0
            current_losses = 0
            
            for trade in self.trades:
                if trade['is_profitable']:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
            
            return max_wins, max_losses
            
        except Exception as e:
            logger.error(f"Error calculating consecutive streaks: {e}")
            return 0, 0
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """Get detailed trade analysis"""
        try:
            if not self.trades:
                return {}
            
            # Group trades by symbol
            symbol_analysis = {}
            for trade in self.trades:
                symbol = trade.get('symbol', 'UNKNOWN')
                if symbol not in symbol_analysis:
                    symbol_analysis[symbol] = {
                        'trades': [],
                        'total_pnl': 0,
                        'win_rate': 0,
                        'profit_factor': 0
                    }
                
                symbol_analysis[symbol]['trades'].append(trade)
                symbol_analysis[symbol]['total_pnl'] += trade['net_pnl']
            
            # Calculate per-symbol metrics
            for symbol, data in symbol_analysis.items():
                trades = data['trades']
                profitable = sum(1 for t in trades if t['is_profitable'])
                
                data['win_rate'] = (profitable / len(trades)) * 100
                
                wins = [t['net_pnl'] for t in trades if t['is_profitable']]
                losses = [t['net_pnl'] for t in trades if not t['is_profitable']]
                
                data['profit_factor'] = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
            
            # Time-based analysis
            if self.trades:
                df = pd.DataFrame(self.trades)
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['hour'] = df['entry_time'].dt.hour
                df['day_of_week'] = df['entry_time'].dt.dayofweek
                
                hourly_performance = df.groupby('hour')['net_pnl'].agg(['mean', 'count', 'sum']).to_dict()
                daily_performance = df.groupby('day_of_week')['net_pnl'].agg(['mean', 'count', 'sum']).to_dict()
            else:
                hourly_performance = {}
                daily_performance = {}
            
            return {
                'symbol_analysis': symbol_analysis,
                'hourly_performance': hourly_performance,
                'daily_performance': daily_performance,
                'total_trades': len(self.trades),
                'avg_trade_duration': 24,  # Fixed 24h for now
                'best_trade': max(self.trades, key=lambda x: x['net_pnl']) if self.trades else None,
                'worst_trade': min(self.trades, key=lambda x: x['net_pnl']) if self.trades else None
            }
            
        except Exception as e:
            logger.error(f"Error getting trade analysis: {e}")
            return {}
    
    def should_disable_strategy(self) -> bool:
        """Check if strategy should be disabled based on performance"""
        try:
            if not self.metrics:
                return False
            
            # Check win rate
            if self.metrics.get('win_rate', 0) < self.disable_threshold * 100:
                logger.warning(f"Strategy disabled: Win rate {self.metrics['win_rate']:.2f}% below threshold")
                return True
            
            # Check total return
            if self.metrics.get('total_return', 0) < -10:  # More than 10% loss
                logger.warning(f"Strategy disabled: Total return {self.metrics['total_return']:.2f}% too negative")
                return True
            
            # Check max drawdown
            if self.metrics.get('max_drawdown', 0) > 20:  # More than 20% drawdown
                logger.warning(f"Strategy disabled: Max drawdown {self.metrics['max_drawdown']:.2f}% too high")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking strategy disable condition: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            summary = {
                'backtest_completed': len(self.trades) > 0,
                'total_trades': len(self.trades),
                'strategy_should_disable': self.should_disable_strategy(),
                'performance_grade': self._calculate_performance_grade(),
                'key_metrics': {
                    'total_return': self.metrics.get('total_return', 0),
                    'win_rate': self.metrics.get('win_rate', 0),
                    'profit_factor': self.metrics.get('profit_factor', 0),
                    'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
                    'max_drawdown': self.metrics.get('max_drawdown', 0)
                },
                'trade_statistics': {
                    'profitable_trades': self.metrics.get('profitable_trades', 0),
                    'losing_trades': self.metrics.get('losing_trades', 0),
                    'average_win': self.metrics.get('average_win', 0),
                    'average_loss': self.metrics.get('average_loss', 0),
                    'largest_win': self.metrics.get('largest_win', 0),
                    'largest_loss': self.metrics.get('largest_loss', 0)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _calculate_performance_grade(self) -> str:
        """Calculate performance grade (A-F)"""
        try:
            if not self.metrics:
                return 'F'
            
            score = 0
            
            # Win rate (30% weight)
            win_rate = self.metrics.get('win_rate', 0)
            if win_rate >= 70:
                score += 30
            elif win_rate >= 60:
                score += 25
            elif win_rate >= 55:
                score += 20
            elif win_rate >= 50:
                score += 15
            elif win_rate >= 45:
                score += 10
            
            # Total return (25% weight)
            total_return = self.metrics.get('total_return', 0)
            if total_return >= 20:
                score += 25
            elif total_return >= 15:
                score += 20
            elif total_return >= 10:
                score += 15
            elif total_return >= 5:
                score += 10
            elif total_return >= 0:
                score += 5
            
            # Profit factor (20% weight)
            profit_factor = self.metrics.get('profit_factor', 0)
            if profit_factor >= 2.0:
                score += 20
            elif profit_factor >= 1.5:
                score += 15
            elif profit_factor >= 1.2:
                score += 10
            elif profit_factor >= 1.0:
                score += 5
            
            # Sharpe ratio (15% weight)
            sharpe_ratio = self.metrics.get('sharpe_ratio', 0)
            if sharpe_ratio >= 2.0:
                score += 15
            elif sharpe_ratio >= 1.5:
                score += 12
            elif sharpe_ratio >= 1.0:
                score += 8
            elif sharpe_ratio >= 0.5:
                score += 4
            
            # Max drawdown (10% weight)
            max_drawdown = self.metrics.get('max_drawdown', 0)
            if max_drawdown <= 5:
                score += 10
            elif max_drawdown <= 10:
                score += 8
            elif max_drawdown <= 15:
                score += 6
            elif max_drawdown <= 20:
                score += 4
            elif max_drawdown <= 25:
                score += 2
            
            # Assign grade
            if score >= 85:
                return 'A'
            elif score >= 75:
                return 'B'
            elif score >= 65:
                return 'C'
            elif score >= 55:
                return 'D'
            else:
                return 'F'
                
        except Exception as e:
            logger.error(f"Error calculating performance grade: {e}")
            return 'F'
    
    def export_results(self, filename: str = None) -> str:
        """Export backtest results to file"""
        try:
            if filename is None:
                filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results = {
                'backtest_summary': self.get_performance_summary(),
                'performance_metrics': self.metrics,
                'trade_analysis': self.get_trade_analysis(),
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'config': self.config,
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""