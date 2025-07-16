# 🚀 Crypto Prediction System v1.0

A comprehensive, self-learning cryptocurrency prediction system that generates daily trading signals for Bitcoin (BTC) and Ethereum (ETH) using advanced AI models and sentiment analysis.

## 🌟 Features

- **Multi-Model AI Architecture**: Combines LSTM, Transformer, and FinBERT models for robust predictions
- **Real-time Data Collection**: Fetches market data from CoinGecko and news from multiple sources
- **Advanced Sentiment Analysis**: Uses FinBERT for financial sentiment classification
- **Comprehensive Technical Analysis**: RSI, MACD, EMA, Bollinger Bands, ATR, and more
- **Intelligent Risk Management**: Dynamic position sizing and stop-loss calculations
- **Performance Tracking**: Detailed backtesting and performance metrics
- **Self-Learning**: Continuous model improvement through reinforcement learning
- **Mobile-Friendly**: Runs on phones via Replit or Google Colab
- **Minimal Resources**: Optimized for low-resource environments

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for data feeds

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced performance:
```bash
# For GPU acceleration (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For TA-Lib (technical analysis)
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/crypto-prediction-system.git
cd crypto-prediction-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Configuration
Create a `.env` file in the project root:
```bash
# API Keys (optional but recommended)
COINGECKO_API_KEY=your_coingecko_api_key
NEWSAPI_KEY=your_newsapi_key
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Google Drive Backup (optional)
GOOGLE_DRIVE_ENABLED=false
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
```

### 4. Run the System
```bash
python main.py
```

## 📊 System Architecture

### Data Flow
```
Market Data (CoinGecko) → Database → Technical Analysis
        ↓
News Data (Multiple Sources) → Sentiment Analysis → AI Models
        ↓
LSTM + Transformer + FinBERT → Ensemble Prediction
        ↓
Risk Management → Trading Signal → Performance Tracking
```

### Key Components

#### 1. Data Collection (`src/data_collection/`)
- **MarketDataCollector**: Fetches real-time price, volume, and market data
- **SentimentAnalyzer**: Processes news headlines using FinBERT
- **Database**: SQLite storage for all data and predictions

#### 2. Technical Analysis (`src/technical_analysis/`)
- **Indicators**: RSI, MACD, EMA, Bollinger Bands, ATR, volatility
- **Signal Generation**: Moving average crossovers, pattern recognition
- **Strength Calculation**: Indicator confidence scoring

#### 3. AI Models (`src/models/`)
- **LSTM**: Time series pattern recognition
- **Transformer**: Long-range trend analysis
- **FinBERT**: Financial sentiment classification
- **Ensemble**: Weighted combination of all models

#### 4. Risk Management (`src/risk_management/`)
- **Position Sizing**: Based on confidence and volatility
- **Stop Loss/Take Profit**: Dynamic calculation
- **Leverage**: Adaptive based on market conditions

#### 5. Backtesting (`src/backtesting/`)
- **Performance Metrics**: Win rate, Sharpe ratio, drawdown
- **Trade Analysis**: Detailed trade-by-trade breakdown
- **Strategy Validation**: Automatic strategy disable on poor performance

## 🎯 Trading Signals

### Signal Format
Each prediction includes:
- **Signal Type**: BUY, SELL, or HOLD
- **Entry Price Range**: Optimal entry prices
- **Stop Loss**: Risk management level
- **Take Profit**: Target profit level
- **Leverage**: Recommended leverage (1-5x)
- **Confidence**: Model confidence (0-100%)
- **Reasoning**: Human-readable explanation

### Example Signal
```json
{
  "symbol": "bitcoin",
  "signal": "BUY",
  "entry_price": 45000,
  "entry_range": [44800, 45200],
  "stop_loss": 43500,
  "take_profit": 47000,
  "leverage": 2.0,
  "confidence": 0.75,
  "reasoning": "RSI Oversold + MACD Bullish + Positive News Sentiment"
}
```

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: Prediction correctness
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough loss

### Tracking
- Real-time performance monitoring
- Daily performance reports
- Weekly model retraining
- Automatic strategy adjustment

## 🔧 Configuration

### Basic Settings (`config/config.py`)
```python
# Update frequency (seconds)
UPDATE_INTERVAL = 300  # 5 minutes

# Supported cryptocurrencies
SYMBOLS = ["bitcoin", "ethereum"]

# Model confidence threshold
MIN_CONFIDENCE = 0.6

# Risk management
MAX_LEVERAGE = 5.0
BASE_STOP_LOSS = 0.02  # 2%
BASE_TAKE_PROFIT = 0.04  # 4%
```

### Advanced Configuration
Modify `config/config.py` for:
- Model hyperparameters
- Technical indicator settings
- Risk management rules
- Notification preferences

## 🤖 AI Models

### LSTM Network
- **Purpose**: Time series pattern recognition
- **Architecture**: 2-layer LSTM with 128 hidden units
- **Input**: 60-period price sequences
- **Output**: 3-class prediction (BUY/SELL/HOLD)

### Transformer Model
- **Purpose**: Long-range dependency modeling
- **Architecture**: 6-layer transformer encoder
- **Features**: Positional encoding, multi-head attention
- **Advantage**: Better context understanding

### FinBERT Integration
- **Purpose**: Financial sentiment analysis
- **Model**: Pre-trained on financial text
- **Input**: News headlines and descriptions
- **Output**: Sentiment score (-1 to 1)

### Ensemble Learning
- **Method**: Weighted voting
- **Weights**: Dynamically adjusted based on performance
- **Confidence**: Ensemble agreement measurement

## 💡 Usage Examples

### Running on Different Platforms

#### Local Machine
```bash
python main.py
```

#### Google Colab
```python
# Upload files to Colab
!git clone https://github.com/yourusername/crypto-prediction-system.git
%cd crypto-prediction-system
!pip install -r requirements.txt
!python main.py
```

#### Replit
1. Import the GitHub repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

### Paper Trading Mode
The system starts in paper trading mode by default. To enable live trading:
```python
# In main.py
app.paper_trading_mode = False
```

### Custom Indicators
Add your own technical indicators:
```python
# In src/technical_analysis/indicators.py
def calculate_custom_indicator(self, data):
    # Your indicator logic here
    return indicator_values
```

## 📱 Mobile Optimization

### Resource Management
- **Memory**: Efficient data structures and garbage collection
- **CPU**: Optimized calculations and batch processing
- **Storage**: Automatic cleanup of old data

### Performance Tips
- Use smaller model sizes for mobile devices
- Reduce update frequency if needed
- Enable data compression
- Use cloud storage for large datasets

## 🔐 Security & Privacy

### Data Protection
- Local SQLite database (no cloud storage by default)
- API keys stored in environment variables
- No personal trading data transmitted

### API Security
- Rate limiting compliance
- Secure API key management
- Optional VPN support

## 📊 Monitoring & Alerts

### Built-in Monitoring
- System health checks
- Performance degradation alerts
- Model accuracy tracking
- Database integrity monitoring

### Telegram Notifications
Set up alerts for:
- New trading signals
- System errors
- Performance milestones
- Weekly reports

## 🚨 Risk Disclaimer

**⚠️ Important Warning:**
This system is for educational and research purposes only. Cryptocurrency trading involves significant risk, and you should never invest more than you can afford to lose. The predictions are not guaranteed to be accurate, and past performance does not guarantee future results.

### Risk Management
- Always use proper position sizing
- Set appropriate stop losses
- Never risk more than 2% of your capital per trade
- Test thoroughly in paper trading mode first
- Monitor performance regularly

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/crypto-prediction-system.git
cd crypto-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Run tests
python -m pytest tests/
```

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Areas for Contribution
- Additional data sources
- New technical indicators
- Enhanced ML models
- Better risk management
- UI/UX improvements
- Documentation updates

## 📚 Documentation

### API Documentation
- [Database Schema](docs/database_schema.md)
- [Model Architecture](docs/model_architecture.md)
- [API Endpoints](docs/api_endpoints.md)

### Tutorials
- [Getting Started Guide](docs/getting_started.md)
- [Advanced Configuration](docs/advanced_config.md)
- [Custom Indicators](docs/custom_indicators.md)
- [Deployment Guide](docs/deployment.md)

## 🔄 Changelog

### v1.0 (Current)
- Initial release
- Multi-model AI architecture
- Real-time data collection
- Advanced risk management
- Comprehensive backtesting
- Mobile optimization

### Planned Features
- Additional cryptocurrencies
- Options trading signals
- Portfolio optimization
- Advanced charting
- Web dashboard
- Mobile app

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If TA-Lib installation fails
pip install --upgrade setuptools
pip install numpy
pip install TA-Lib

# If PyTorch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Memory Issues
```python
# Reduce model size in config
LSTM_HIDDEN_SIZE = 64  # Instead of 128
TRANSFORMER_D_MODEL = 256  # Instead of 512
```

#### Data Issues
```bash
# Clear database and restart
rm data/crypto_data.db
python main.py
```

### Getting Help
- Check the [FAQ](docs/faq.md)
- Search existing [Issues](https://github.com/yourusername/crypto-prediction-system/issues)
- Create a new issue with detailed description
- Join our [Discord](https://discord.gg/your-discord) community

## 📞 Support

### Community
- **Discord**: [Join our server](https://discord.gg/your-discord)
- **Telegram**: [Join our channel](https://t.me/your-channel)
- **Reddit**: [r/CryptoPrediction](https://reddit.com/r/CryptoPrediction)

### Professional Support
- Email: support@cryptoprediction.com
- Documentation: [docs.cryptoprediction.com](https://docs.cryptoprediction.com)
- Enterprise: [enterprise@cryptoprediction.com](mailto:enterprise@cryptoprediction.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CoinGecko** for market data API
- **Hugging Face** for FinBERT model
- **TA-Lib** for technical analysis
- **PyTorch** for deep learning framework
- **Streamlit** for dashboard framework

---

<p align="center">
  <strong>⭐ Star this repository if you find it useful!</strong><br>
  <sub>Made with ❤️ by the Crypto Prediction Team</sub>
</p>