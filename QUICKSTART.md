# 🚀 Quick Start Guide

Get your Crypto Prediction System running in 5 minutes!

## 1. Installation

### Option A: Replit (Recommended)
1. Fork this repository to Replit
2. Open the Shell and run:
```bash
pip install -r requirements.txt
```

### Option B: Local Installation
```bash
git clone https://github.com/yourusername/crypto-predictor.git
cd crypto-predictor
pip install -r requirements.txt
```

## 2. Basic Setup

### Create API Keys File
Create `api_keys.json` in the root directory:
```json
{
  "COINGECKO_API_KEY": "optional-for-higher-limits",
  "CRYPTOPANIC_API_KEY": "optional-for-news",
  "NEWS_API_KEY": "optional-for-news",
  "TELEGRAM_BOT_TOKEN": "optional-for-alerts",
  "TELEGRAM_CHAT_ID": "optional-for-alerts"
}
```

**Note**: The system works without API keys using free endpoints, but performance is limited.

## 3. Test the System

### Check Dependencies
```bash
python run.py --status
```

### Generate Your First Prediction
```bash
python run.py --predict BTC
```

### Run a Quick Backtest
```bash
python run.py --backtest BTC --days 7
```

## 4. Start the System

### Run Continuously
```bash
python run.py
```

The system will:
- Collect market data every 5 minutes
- Generate predictions for BTC and ETH
- Store results in SQLite database
- Send alerts for high-confidence signals (if configured)

### Stop the System
Press `Ctrl+C` to stop gracefully.

## 5. Free API Setup (Optional)

### CoinGecko (Market Data)
- Visit: https://www.coingecko.com/en/api
- Sign up for free tier (100 requests/minute)
- Add API key to `api_keys.json`

### CryptoPanic (News)
- Visit: https://cryptopanic.com/developers/api/
- Sign up for free tier
- Add API key to `api_keys.json`

### NewsAPI (General News)
- Visit: https://newsapi.org/
- Sign up for free tier (1000 requests/day)
- Add API key to `api_keys.json`

### Telegram Alerts
1. Create a bot with @BotFather on Telegram
2. Get bot token
3. Find your chat ID (message @userinfobot)
4. Add both to `api_keys.json`

## 6. Basic Commands

```bash
# Generate prediction
python run.py --predict ETH

# Run backtest
python run.py --backtest BTC --days 30

# Test notifications
python run.py --test-notifications

# Check system status
python run.py --status

# Enable debug logging
python run.py --debug
```

## 7. Understanding Output

### Prediction Example
```
🎯 Prediction for BTC
Signal: BUY
Confidence: 75.3%
Entry Price: $43,250.00
Stop Loss: $42,180.00
Take Profit: $44,850.00
Leverage: 2.1x
Rationale: RSI oversold + Positive news sentiment
```

### Signal Types
- **BUY**: Expect price to go up
- **SELL**: Expect price to go down
- **HOLD**: No clear direction or low confidence

### Confidence Levels
- **> 70%**: High confidence, alerts sent
- **60-70%**: Medium confidence
- **< 60%**: Low confidence, filtered out

## 8. Customization

### Edit Configuration
Modify `crypto_predictor/config/config.py`:
```python
# Change trading pairs
TRADING_PAIRS = ['BTC', 'ETH', 'ADA']

# Adjust data collection interval
DATA_COLLECTION_INTERVAL = 10  # minutes

# Change confidence threshold
CONFIDENCE_THRESHOLD = 0.70
```

### Add New Indicators
```python
from crypto_predictor import technical_analyzer

# Add custom indicator
def my_indicator(df):
    return df['price'].rolling(20).mean()

technical_analyzer.add_custom_indicator('my_sma', my_indicator)
```

## 9. Dashboard (Optional)

If you have Streamlit installed:
```bash
python run.py --dashboard
```

Access at: http://localhost:8501

## 10. Troubleshooting

### Common Issues

**"Missing packages" error:**
```bash
pip install -r requirements.txt
```

**Database locked error:**
Make sure only one instance is running.

**API rate limits:**
Add API keys or increase delays in config.

**Memory issues on Replit:**
Reduce batch size in `crypto_predictor/config/config.py`.

### Debug Mode
```bash
python run.py --debug
```

## 11. Paper Trading Mode

By default, the system runs in paper trading mode (no real money). To verify:
```python
from crypto_predictor.config.config import system_config
print(system_config.PAPER_TRADING)  # Should be True
```

## 12. Performance Monitoring

The system tracks:
- **Win Rate**: Percentage of profitable signals
- **Accuracy**: Percentage of correct predictions
- **Drawdown**: Maximum portfolio decline
- **Sharpe Ratio**: Risk-adjusted returns

Check performance:
```bash
python run.py --status
```

## 13. Next Steps

1. **Optimize**: Tune parameters based on your risk tolerance
2. **Monitor**: Watch performance over time
3. **Extend**: Add more trading pairs or indicators
4. **Automate**: Set up notifications for important signals
5. **Backtest**: Test strategies on historical data

## 🎯 Pro Tips

1. **Start Small**: Begin with paper trading to understand the system
2. **Monitor Performance**: Track win rate and adjust if needed
3. **Use Alerts**: Set up Telegram for real-time notifications
4. **Regular Backtests**: Test strategies on historical data
5. **Update Regularly**: Keep the system and dependencies updated

## 🔗 Resources

- **Full Documentation**: See README.md
- **Configuration Guide**: `crypto_predictor/config/config.py`
- **API Documentation**: Each module has detailed docstrings
- **Issues**: Report problems on GitHub Issues
- **Community**: Join discussions on GitHub

---

**Happy Trading! 🚀**

*Remember: This is for educational purposes only. Always do your own research before making financial decisions.*