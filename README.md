# 🚀 Crypto Prediction System

A free, mobile-optimized, self-learning cryptocurrency prediction system built to run entirely on Replit or Google Colab with minimal resources. The system generates daily high-confidence trading signals—**BUY**, **SELL**, or **HOLD**—for Bitcoin (BTC) and Ethereum (ETH) using advanced AI models and sentiment analysis.

## 🌟 Features

### Core Functionality
- **Real-time Market Data**: Collects data every 5-15 minutes from CoinGecko and Yahoo Finance
- **Sentiment Analysis**: Processes crypto news from CryptoPanic, NewsAPI, and Reddit using FinBERT
- **Technical Analysis**: Calculates 20+ indicators including RSI, MACD, EMA, Bollinger Bands, ATR
- **Hybrid AI Architecture**: Combines LSTM, Transformer, and FinBERT models with ensemble learning
- **Risk Management**: Dynamic position sizing, stop-loss, and take-profit calculations
- **Continuous Learning**: Reinforcement learning loop for strategy optimization

### Advanced Features
- **Backtesting Engine**: Historical performance analysis with detailed metrics
- **Performance Monitoring**: Real-time tracking of win rate, drawdown, and accuracy
- **Multi-Channel Alerts**: Telegram, email, Slack, and webhook notifications
- **Database Storage**: SQLite with automatic backups and Google Drive sync
- **Streamlit Dashboard**: Live monitoring interface (optional)
- **Paper Trading**: Safe simulation mode for testing strategies

### Target Performance
- **Prediction Accuracy**: 60-70%
- **Win Rate**: 55%+
- **Risk-Adjusted Returns**: Optimized for mobile/low-power devices

## 🛠️ Installation

### Option 1: Run on Replit
1. Fork this repository to Replit
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (see Configuration section)
4. Run the system:
   ```bash
   python -m crypto_predictor.main
   ```

### Option 2: Run on Google Colab
1. Upload the project to Google Drive
2. Open the notebook in Colab
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Configure API keys
5. Run the main system

### Option 3: Local Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-predictor.git
cd crypto-predictor

# Install dependencies
pip install -r requirements.txt

# Run the system
python -m crypto_predictor.main
```

## ⚙️ Configuration

### Required API Keys
Create an `api_keys.json` file or set environment variables:

```json
{
  "COINGECKO_API_KEY": "your_coingecko_api_key",
  "CRYPTOPANIC_API_KEY": "your_cryptopanic_api_key",
  "NEWS_API_KEY": "your_news_api_key",
  "REDDIT_CLIENT_ID": "your_reddit_client_id",
  "REDDIT_CLIENT_SECRET": "your_reddit_client_secret",
  "TELEGRAM_BOT_TOKEN": "your_telegram_bot_token",
  "TELEGRAM_CHAT_ID": "your_telegram_chat_id",
  "EMAIL_USER": "your_email@gmail.com",
  "EMAIL_PASS": "your_email_password"
}
```

### Free API Sources
- **CoinGecko**: Free tier allows 100 requests/minute
- **CryptoPanic**: Free tier with basic news access
- **NewsAPI**: Free tier with 1000 requests/day
- **Reddit**: Free API access with rate limits

### System Configuration
Edit `crypto_predictor/config/config.py` to customize:
- Trading pairs
- Data collection intervals
- Model parameters
- Risk management settings
- Performance targets

## 📊 Usage

### Basic Usage
```python
from crypto_predictor.main import orchestrator

# Start the system
orchestrator.start_system()

# Generate manual prediction
prediction = orchestrator.generate_manual_prediction('BTC')
print(f"Signal: {prediction.signal}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Entry Price: ${prediction.entry_price:.2f}")

# Run backtest
results = orchestrator.run_backtest('BTC', days=30)
print(f"Win Rate: {results['win_rate']:.1%}")
```

### Dashboard Access
If enabled, access the Streamlit dashboard at:
```
http://localhost:8501
```

### Command Line Interface
```bash
# Generate prediction
python -m crypto_predictor.main --predict BTC

# Run backtest
python -m crypto_predictor.main --backtest BTC --days 30

# Check system status
python -m crypto_predictor.main --status
```

## 🧠 AI Models

### LSTM Model
- **Purpose**: Short-term time series prediction
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Features**: Technical indicators, price sequences
- **Lookback**: 30 days of historical data

### Transformer Model
- **Purpose**: Long-range pattern detection
- **Architecture**: Multi-head attention with positional encoding
- **Features**: Technical indicators, market patterns
- **Sequence Length**: 100 data points

### FinBERT Sentiment Model
- **Purpose**: Financial sentiment analysis
- **Architecture**: Fine-tuned BERT for financial text
- **Sources**: News headlines, social media posts
- **Output**: Sentiment score (-1 to 1)

### Ensemble Learning
- **Method**: Weighted voting with confidence scoring
- **Meta-learner**: Neural network combining model outputs
- **Optimization**: Continuous weight adjustment based on performance

## 📈 Performance Metrics

### Prediction Accuracy
- **Overall Accuracy**: % of correct signal predictions
- **Win Rate**: % of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum portfolio decline

### Risk Metrics
- **Value at Risk (VaR)**: 95% confidence interval
- **Volatility**: Price movement standard deviation
- **Kelly Criterion**: Optimal position sizing
- **Correlation**: Cross-asset risk assessment

## 🔧 Advanced Features

### Risk Management
```python
# Adjust risk parameters
risk_manager.max_leverage = 2.0
risk_manager.max_position_size = 0.05  # 5% of portfolio

# Custom risk filters
def custom_risk_filter(prediction):
    if prediction.confidence < 0.8:
        return 'HOLD'
    return prediction.signal
```

### Custom Indicators
```python
# Add custom technical indicator
def custom_indicator(df):
    return df['price'].rolling(10).mean() / df['price'].rolling(50).mean()

# Register indicator
technical_analyzer.add_custom_indicator('price_ratio', custom_indicator)
```

### Notification Setup
```python
# Test notifications
notification_manager.test_notifications()

# Send custom alert
notification_manager.send_system_alert('INFO', 'Custom message')
```

## 📱 Mobile Optimization

### Resource Management
- **Memory Usage**: < 512MB on Replit
- **CPU Usage**: Optimized for single-core processing
- **Storage**: SQLite database with compression
- **Network**: Efficient API call batching

### Platform-Specific Optimizations
```python
# Automatically detected platform optimizations
if platform == "colab":
    batch_size = 32
    max_workers = 2
elif platform == "replit":
    batch_size = 16
    max_workers = 1
```

## 🔄 Continuous Learning

### Reinforcement Learning
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Frequency**: Updates every 100 predictions
- **Metrics**: Win rate, profit factor, drawdown
- **Adaptation**: Strategy weights, indicator thresholds

### Model Retraining
- **Frequency**: Weekly automatic retraining
- **Trigger**: Performance degradation detection
- **Data**: Rolling window of recent market data
- **Validation**: Out-of-sample testing

## 📊 Backtesting

### Running Backtests
```python
# Single symbol backtest
results = backtester.run_backtest('BTC', historical_data, 
                                 start_date='2023-01-01',
                                 end_date='2023-12-31')

# Performance report
report = backtester.generate_performance_report(results)
print(f"Total Return: {report['summary']['total_return']:.2%}")
```

### Backtest Metrics
- **Total Trades**: Number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest portfolio decline

## 🚨 Alerts & Notifications

### Telegram Setup
1. Create a bot with @BotFather
2. Get bot token and chat ID
3. Add credentials to configuration
4. Enable notifications in settings

### Email Setup
1. Use Gmail with app password
2. Configure SMTP settings
3. Add credentials to configuration
4. Test email notifications

### Webhook Integration
```python
# Custom webhook handler
def webhook_handler(prediction):
    data = {
        'symbol': prediction.symbol,
        'signal': prediction.signal,
        'confidence': prediction.confidence,
        'timestamp': prediction.timestamp.isoformat()
    }
    # Send to your endpoint
```

## 📚 Database Schema

### Predictions Table
- **id**: Primary key
- **symbol**: Trading pair
- **timestamp**: Prediction time
- **signal**: BUY/SELL/HOLD
- **confidence**: Prediction confidence
- **entry_price**: Entry price
- **stop_loss**: Stop loss price
- **take_profit**: Take profit price
- **leverage**: Position leverage
- **rationale**: Explanation text

### Performance Metrics Table
- **id**: Primary key
- **metric_date**: Date of metrics
- **total_predictions**: Number of predictions
- **win_rate**: Percentage of wins
- **total_profit**: Total profit/loss
- **max_drawdown**: Maximum drawdown

## 🔐 Security & Privacy

### Data Protection
- **Local Storage**: All data stored locally
- **No Cloud Dependencies**: Optional cloud sync only
- **API Key Security**: Environment variables or encrypted storage
- **Log Sanitization**: Sensitive data filtering

### Rate Limiting
- **CoinGecko**: 100 requests/minute (free tier)
- **NewsAPI**: 1000 requests/day (free tier)
- **Reddit**: 60 requests/minute
- **Automatic Backoff**: Exponential retry delays

## 🐛 Troubleshooting

### Common Issues
1. **API Rate Limits**: Increase delays between requests
2. **Memory Issues**: Reduce batch size or sequence length
3. **Model Loading**: Check available disk space
4. **Database Locks**: Ensure single instance running

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system status
status = orchestrator.get_system_status()
print(status)
```

### Performance Issues
```python
# Monitor system resources
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"CPU: {psutil.cpu_percent()}%")
```

## 📈 Performance Optimization

### Model Optimization
- **Quantization**: Reduce model size for mobile devices
- **Pruning**: Remove less important model weights
- **Caching**: Store frequent calculations
- **Batch Processing**: Efficient data processing

### Database Optimization
- **Indexing**: Fast query performance
- **Compression**: Reduce storage requirements
- **Cleanup**: Automatic old data removal
- **Vacuuming**: Regular database maintenance

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black crypto_predictor/
isort crypto_predictor/
```

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves significant risk and can result in substantial losses. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.**

The developers of this system are not responsible for any financial losses incurred through the use of this software.

## 🔗 Links

- **Documentation**: [Full API Documentation](docs/)
- **Examples**: [Usage Examples](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/crypto-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/crypto-predictor/discussions)

## 🎯 Roadmap

### Version 1.1 (Next Release)
- [ ] Support for more trading pairs
- [ ] Advanced portfolio optimization
- [ ] Multi-timeframe analysis
- [ ] Options trading signals

### Version 1.2 (Future)
- [ ] Mobile app interface
- [ ] Real-time streaming data
- [ ] Advanced ML models
- [ ] Social trading features

### Version 2.0 (Long-term)
- [ ] DeFi integration
- [ ] Cross-chain analysis
- [ ] Automated trading execution
- [ ] Professional dashboard

---

**⭐ Star this repository if you find it useful!**

**💬 Join our community for support and discussions**

**🐛 Report bugs and request features through GitHub Issues**