import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..config.config import system_config, api_config, platform_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    market_cap: float
    price_change_24h: float
    volume_change_24h: float
    market_cap_change_24h: float
    high_24h: float
    low_24h: float
    supply_circulating: float
    supply_total: float

class CoinGeckoClient:
    """CoinGecko API client for real-time market data"""
    
    def __init__(self):
        self.base_url = system_config.COINGECKO_API_URL
        self.api_key = api_config.get_api_key('COINGECKO_API_KEY')
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 10 requests per second for free tier
        
        # Headers for API requests
        self.headers = {
            'User-Agent': 'CryptoPredictorBot/1.0',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            self.headers['x-cg-pro-api-key'] = self.api_key
            self.rate_limit_delay = 0.02  # 50 requests per second for pro tier
    
    def get_current_price(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get current price and market data for symbols"""
        try:
            # Convert symbols to CoinGecko format
            coin_ids = self._get_coin_ids(symbols)
            
            # Prepare request parameters
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = self.session.get(
                f"{self.base_url}/simple/price",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Process response
            market_data = {}
            for i, coin_id in enumerate(coin_ids):
                if coin_id in data:
                    coin_data = data[coin_id]
                    market_data[symbols[i]] = MarketData(
                        symbol=symbols[i],
                        timestamp=datetime.fromtimestamp(coin_data.get('last_updated_at', time.time())),
                        price=coin_data.get('usd', 0),
                        volume=coin_data.get('usd_24h_vol', 0),
                        market_cap=coin_data.get('usd_market_cap', 0),
                        price_change_24h=coin_data.get('usd_24h_change', 0),
                        volume_change_24h=0,  # Not available in simple API
                        market_cap_change_24h=0,  # Not available in simple API
                        high_24h=0,  # Not available in simple API
                        low_24h=0,  # Not available in simple API
                        supply_circulating=0,  # Not available in simple API
                        supply_total=0  # Not available in simple API
                    )
            
            time.sleep(self.rate_limit_delay)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return {}
    
    def get_detailed_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get detailed market data including high/low, supply, etc."""
        try:
            coin_ids = self._get_coin_ids(symbols)
            market_data = {}
            
            # Use coins/{id} endpoint for detailed data
            for i, coin_id in enumerate(coin_ids):
                try:
                    response = self.session.get(
                        f"{self.base_url}/coins/{coin_id}",
                        headers=self.headers,
                        timeout=10
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    market_stats = data.get('market_data', {})
                    
                    market_data[symbols[i]] = MarketData(
                        symbol=symbols[i],
                        timestamp=datetime.now(),
                        price=market_stats.get('current_price', {}).get('usd', 0),
                        volume=market_stats.get('total_volume', {}).get('usd', 0),
                        market_cap=market_stats.get('market_cap', {}).get('usd', 0),
                        price_change_24h=market_stats.get('price_change_percentage_24h', 0),
                        volume_change_24h=0,
                        market_cap_change_24h=market_stats.get('market_cap_change_percentage_24h', 0),
                        high_24h=market_stats.get('high_24h', {}).get('usd', 0),
                        low_24h=market_stats.get('low_24h', {}).get('usd', 0),
                        supply_circulating=market_stats.get('circulating_supply', 0),
                        supply_total=market_stats.get('total_supply', 0)
                    )
                    
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error fetching detailed data for {symbols[i]}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching detailed data: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for backtesting and model training"""
        try:
            coin_id = self._get_coin_ids([symbol])[0]
            
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            response = self.session.get(
                f"{self.base_url}/coins/{coin_id}/market_chart",
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts/1000) for ts in data['prices']],
                'price': [price[1] for price in data['prices']],
                'volume': [vol[1] for vol in data['total_volumes']],
                'market_cap': [mc[1] for mc in data['market_caps']]
            })
            
            df.set_index('timestamp', inplace=True)
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_coin_ids(self, symbols: List[str]) -> List[str]:
        """Convert symbols to CoinGecko coin IDs"""
        coin_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'MATIC': 'polygon',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'ATOM': 'cosmos'
        }
        
        return [coin_mapping.get(symbol.upper(), symbol.lower()) for symbol in symbols]

class YahooFinanceClient:
    """Yahoo Finance client as backup data source"""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def get_current_price(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get current price from Yahoo Finance"""
        try:
            market_data = {}
            
            for symbol in symbols:
                yahoo_symbol = f"{symbol}-USD"
                url = f"{self.base_url}/{yahoo_symbol}"
                
                params = {
                    'range': '1d',
                    'interval': '1m',
                    'includePrePost': 'false',
                    'events': 'div%2Csplit',
                    'corsDomain': 'finance.yahoo.com'
                }
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                result = data['chart']['result'][0]
                
                # Get latest price
                closes = result['indicators']['quote'][0]['close']
                volumes = result['indicators']['quote'][0]['volume']
                current_price = closes[-1] if closes else 0
                current_volume = volumes[-1] if volumes else 0
                
                market_data[symbol] = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    volume=current_volume,
                    market_cap=0,  # Not available
                    price_change_24h=0,  # Calculate from historical data
                    volume_change_24h=0,
                    market_cap_change_24h=0,
                    high_24h=0,
                    low_24h=0,
                    supply_circulating=0,
                    supply_total=0
                )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return {}

class MarketDataCollector:
    """Main market data collector with multiple data sources"""
    
    def __init__(self):
        self.coingecko = CoinGeckoClient()
        self.yahoo = YahooFinanceClient()
        self.last_update = {}
        self.cache = {}
        self.cache_duration = 60  # seconds
    
    def collect_market_data(self, symbols: List[str] = None, detailed: bool = False) -> Dict[str, MarketData]:
        """Collect market data from multiple sources with fallback"""
        if symbols is None:
            symbols = system_config.TRADING_PAIRS
        
        # Check cache first
        cache_key = f"{'_'.join(symbols)}_{'detailed' if detailed else 'simple'}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_data
        
        market_data = {}
        
        try:
            # Primary source: CoinGecko
            if detailed:
                market_data = self.coingecko.get_detailed_data(symbols)
            else:
                market_data = self.coingecko.get_current_price(symbols)
            
            # Fill missing data with Yahoo Finance
            missing_symbols = [s for s in symbols if s not in market_data]
            if missing_symbols:
                yahoo_data = self.yahoo.get_current_price(missing_symbols)
                market_data.update(yahoo_data)
            
            # Cache the result
            self.cache[cache_key] = (time.time(), market_data)
            
            logger.info(f"Collected market data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for training and backtesting"""
        try:
            # Try CoinGecko first
            df = self.coingecko.get_historical_data(symbol, days)
            
            if df.empty:
                logger.warning(f"No historical data from CoinGecko for {symbol}")
                # Could implement Yahoo Finance historical data as fallback
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def start_real_time_collection(self, callback_func, symbols: List[str] = None):
        """Start real-time data collection with callback"""
        if symbols is None:
            symbols = system_config.TRADING_PAIRS
        
        def collect_and_notify():
            while True:
                try:
                    market_data = self.collect_market_data(symbols)
                    if market_data:
                        callback_func(market_data)
                    
                    time.sleep(system_config.DATA_COLLECTION_INTERVAL * 60)
                    
                except Exception as e:
                    logger.error(f"Error in real-time collection: {e}")
                    time.sleep(30)  # Wait before retrying
        
        return collect_and_notify
    
    def get_market_summary(self, symbols: List[str] = None) -> Dict:
        """Get market summary with key metrics"""
        if symbols is None:
            symbols = system_config.TRADING_PAIRS
        
        market_data = self.collect_market_data(symbols, detailed=True)
        
        summary = {
            'total_market_cap': sum(data.market_cap for data in market_data.values()),
            'total_volume': sum(data.volume for data in market_data.values()),
            'gainers': [],
            'losers': [],
            'highest_volume': [],
            'timestamp': datetime.now()
        }
        
        # Sort by price change
        sorted_by_change = sorted(market_data.items(), key=lambda x: x[1].price_change_24h, reverse=True)
        summary['gainers'] = [(symbol, data.price_change_24h) for symbol, data in sorted_by_change[:5]]
        summary['losers'] = [(symbol, data.price_change_24h) for symbol, data in sorted_by_change[-5:]]
        
        # Sort by volume
        sorted_by_volume = sorted(market_data.items(), key=lambda x: x[1].volume, reverse=True)
        summary['highest_volume'] = [(symbol, data.volume) for symbol, data in sorted_by_volume[:5]]
        
        return summary

# Create global instance
market_collector = MarketDataCollector()