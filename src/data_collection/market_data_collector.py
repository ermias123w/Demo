import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import aiohttp
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    market_cap: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None

class CoinGeckoCollector:
    """Collects market data from CoinGecko API"""
    
    def __init__(self, api_key: str = ""):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # 1 second between requests for free tier
        self.last_request_time = 0
        
        # Symbol mapping (CoinGecko uses different IDs)
        self.symbol_mapping = {
            "bitcoin": "bitcoin",
            "ethereum": "ethereum",
            "btc": "bitcoin",
            "eth": "ethereum"
        }
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            self._wait_for_rate_limit()
            
            url = urljoin(self.base_url, endpoint)
            headers = {}
            
            if self.api_key:
                headers['x-cg-demo-api-key'] = self.api_key
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[MarketData]:
        """Get current price data for a symbol"""
        try:
            coin_id = self.symbol_mapping.get(symbol.lower(), symbol.lower())
            
            endpoint = "/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            data = self._make_request(endpoint, params)
            if not data or coin_id not in data:
                return None
            
            coin_data = data[coin_id]
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=coin_data.get('usd', 0),
                volume=coin_data.get('usd_24h_vol', 0),
                market_cap=coin_data.get('usd_market_cap'),
                price_change_24h=coin_data.get('usd_24h_change')
            )
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Get historical price data"""
        try:
            coin_id = self.symbol_mapping.get(symbol.lower(), symbol.lower())
            
            endpoint = f"/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 30 else 'daily'
            }
            
            data = self._make_request(endpoint, params)
            if not data:
                return []
            
            historical_data = []
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            for i, price_data in enumerate(prices):
                timestamp = datetime.fromtimestamp(price_data[0] / 1000)
                price = price_data[1]
                volume = volumes[i][1] if i < len(volumes) else 0
                market_cap = market_caps[i][1] if i < len(market_caps) else None
                
                historical_data.append(MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    market_cap=market_cap
                ))
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        try:
            endpoint = "/global"
            data = self._make_request(endpoint)
            
            if not data or 'data' not in data:
                return {}
            
            market_data = data['data']
            return {
                'total_market_cap': market_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                'market_cap_percentage': market_data.get('market_cap_percentage', {}),
                'market_cap_change_24h': market_data.get('market_cap_change_percentage_24h_usd', 0),
                'active_cryptocurrencies': market_data.get('active_cryptocurrencies', 0),
                'markets': market_data.get('markets', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}

class NewsCollector:
    """Collects crypto news from various sources"""
    
    def __init__(self, newsapi_key: str = "", cryptopanic_key: str = ""):
        self.newsapi_key = newsapi_key
        self.cryptopanic_key = cryptopanic_key
        self.session = requests.Session()
        self.rate_limit_delay = 1.0
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_crypto_news_newsapi(self, query: str = "bitcoin OR ethereum OR crypto", 
                               hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get crypto news from NewsAPI"""
        try:
            if not self.newsapi_key:
                logger.warning("NewsAPI key not provided")
                return []
            
            self._wait_for_rate_limit()
            
            url = "https://newsapi.org/v2/everything"
            
            # Calculate from date
            from_date = datetime.now() - timedelta(hours=hours_back)
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key,
                'language': 'en',
                'pageSize': 50
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', '')
                })
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error getting news from NewsAPI: {e}")
            return []
    
    def get_crypto_news_cryptopanic(self, currencies: List[str] = None, 
                                   hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get crypto news from CryptoPanic"""
        try:
            if not self.cryptopanic_key:
                logger.warning("CryptoPanic key not provided")
                return []
            
            self._wait_for_rate_limit()
            
            url = "https://cryptopanic.com/api/v1/posts/"
            
            params = {
                'auth_token': self.cryptopanic_key,
                'public': 'true',
                'kind': 'news',
                'filter': 'important',
                'region': 'en'
            }
            
            if currencies:
                params['currencies'] = ','.join(currencies).upper()
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            processed_articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for article in results:
                # Parse published date
                published_at = datetime.fromisoformat(
                    article.get('published_at', '').replace('Z', '+00:00')
                )
                
                # Filter by time
                if published_at < cutoff_time:
                    continue
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('title', ''),  # CryptoPanic doesn't have separate description
                    'content': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('title', ''),
                    'published_at': article.get('published_at', ''),
                    'author': '',
                    'votes': article.get('votes', {})
                })
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error getting news from CryptoPanic: {e}")
            return []
    
    def get_reddit_crypto_posts(self, subreddit: str = "cryptocurrency", 
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get crypto posts from Reddit (no API key required)"""
        try:
            self._wait_for_rate_limit()
            
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            
            headers = {
                'User-Agent': 'CryptoPredictionBot/1.0'
            }
            
            params = {
                'limit': limit
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            processed_posts = []
            for post in posts:
                post_data = post.get('data', {})
                
                # Filter out non-crypto related posts (basic keyword filtering)
                title = post_data.get('title', '').lower()
                crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 'altcoin']
                
                if not any(keyword in title for keyword in crypto_keywords):
                    continue
                
                processed_posts.append({
                    'title': post_data.get('title', ''),
                    'description': post_data.get('selftext', ''),
                    'content': post_data.get('selftext', ''),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'source': f"r/{subreddit}",
                    'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                    'author': post_data.get('author', ''),
                    'score': post_data.get('score', 0),
                    'comments': post_data.get('num_comments', 0)
                })
            
            return processed_posts
            
        except Exception as e:
            logger.error(f"Error getting Reddit posts: {e}")
            return []

class MarketDataCollector:
    """Main market data collector that orchestrates all data sources"""
    
    def __init__(self, coingecko_key: str = "", newsapi_key: str = "", 
                 cryptopanic_key: str = ""):
        self.coingecko = CoinGeckoCollector(coingecko_key)
        self.news_collector = NewsCollector(newsapi_key, cryptopanic_key)
        self.symbols = ["bitcoin", "ethereum"]
    
    def collect_current_market_data(self) -> Dict[str, MarketData]:
        """Collect current market data for all symbols"""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                data = self.coingecko.get_current_price(symbol)
                if data:
                    market_data[symbol] = data
                    logger.info(f"Collected market data for {symbol}: ${data.price:.2f}")
                else:
                    logger.warning(f"Failed to collect market data for {symbol}")
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
        
        return market_data
    
    def collect_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Collect historical market data"""
        try:
            return self.coingecko.get_historical_data(symbol, days)
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return []
    
    def collect_news_data(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect news data from all sources"""
        all_news = []
        
        # Get news from different sources
        try:
            # NewsAPI
            newsapi_articles = self.news_collector.get_crypto_news_newsapi(
                "bitcoin OR ethereum OR crypto", hours_back
            )
            all_news.extend(newsapi_articles)
            
            # CryptoPanic
            cryptopanic_articles = self.news_collector.get_crypto_news_cryptopanic(
                ["BTC", "ETH"], hours_back
            )
            all_news.extend(cryptopanic_articles)
            
            # Reddit
            reddit_posts = self.news_collector.get_reddit_crypto_posts()
            all_news.extend(reddit_posts)
            
            logger.info(f"Collected {len(all_news)} news articles/posts")
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
        
        return all_news
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        return self.coingecko.get_market_overview()
    
    async def collect_data_async(self) -> Dict[str, Any]:
        """Collect all data asynchronously"""
        try:
            # Run data collection in parallel
            tasks = [
                asyncio.create_task(self._async_market_data()),
                asyncio.create_task(self._async_news_data()),
                asyncio.create_task(self._async_market_overview())
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                'market_data': results[0] if not isinstance(results[0], Exception) else {},
                'news_data': results[1] if not isinstance(results[1], Exception) else [],
                'market_overview': results[2] if not isinstance(results[2], Exception) else {}
            }
            
        except Exception as e:
            logger.error(f"Error in async data collection: {e}")
            return {'market_data': {}, 'news_data': [], 'market_overview': {}}
    
    async def _async_market_data(self) -> Dict[str, MarketData]:
        """Async wrapper for market data collection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.collect_current_market_data)
    
    async def _async_news_data(self) -> List[Dict[str, Any]]:
        """Async wrapper for news data collection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.collect_news_data)
    
    async def _async_market_overview(self) -> Dict[str, Any]:
        """Async wrapper for market overview"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_market_overview)