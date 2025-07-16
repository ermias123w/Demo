import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime, timedelta
import re
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from ..config.config import system_config, api_config, platform_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_label: str
    confidence: float
    keywords: List[str]
    relevance_score: float

class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial news"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'ProsusAI/finbert'
        self.tokenizer = None
        self.model = None
        self.labels = ['negative', 'neutral', 'positive']
        self.load_model()
    
    def load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            logger.info("Falling back to VADER sentiment analysis")
            self.model = None
            self.tokenizer = None
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using FinBERT or fallback to VADER"""
        if self.model is None or self.tokenizer is None:
            return self._analyze_with_vader(text)
        
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get sentiment score and label
            probs = predictions.cpu().numpy()[0]
            sentiment_idx = np.argmax(probs)
            sentiment_label = self.labels[sentiment_idx]
            confidence = float(probs[sentiment_idx])
            
            # Convert to score (-1 to 1)
            sentiment_score = probs[2] - probs[0]  # positive - negative
            
            return sentiment_score, sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return self._analyze_with_vader(text)
    
    def _analyze_with_vader(self, text: str) -> Tuple[float, str, float]:
        """Fallback sentiment analysis using VADER"""
        try:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            sentiment_score = scores['compound']
            
            if sentiment_score >= 0.05:
                sentiment_label = 'positive'
            elif sentiment_score <= -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            confidence = abs(sentiment_score)
            
            return sentiment_score, sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return 0.0, 'neutral', 0.0
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > 500:
            text = text[:500]
        
        return text
    
    def batch_analyze(self, texts: List[str]) -> List[Tuple[float, str, float]]:
        """Batch sentiment analysis for efficiency"""
        results = []
        
        if self.model is None or self.tokenizer is None:
            for text in texts:
                results.append(self._analyze_with_vader(text))
            return results
        
        try:
            # Process in batches to avoid memory issues
            batch_size = platform_config.get('batch_size', 8)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_results = []
                
                for text in batch_texts:
                    result = self.analyze_sentiment(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Small delay to prevent overheating on limited hardware
                time.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [(0.0, 'neutral', 0.0) for _ in texts]

class CryptoPanicClient:
    """CryptoPanic news API client"""
    
    def __init__(self):
        self.base_url = system_config.CRYPTOPANIC_API_URL
        self.api_key = api_config.get_api_key('CRYPTOPANIC_API_KEY')
        self.session = requests.Session()
    
    def get_news(self, currencies: List[str] = None, limit: int = 20) -> List[NewsArticle]:
        """Get latest crypto news from CryptoPanic"""
        if not self.api_key:
            logger.warning("CryptoPanic API key not found")
            return []
        
        try:
            params = {
                'auth_token': self.api_key,
                'public': 'true',
                'kind': 'news',
                'filter': 'rising',
                'limit': limit
            }
            
            if currencies:
                params['currencies'] = ','.join(currencies)
            
            response = self.session.get(
                f"{self.base_url}/posts/",
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('results', []):
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('title', ''),  # CryptoPanic doesn't provide full content
                    source=item.get('source', {}).get('title', 'CryptoPanic'),
                    url=item.get('url', ''),
                    published_at=datetime.fromisoformat(item.get('published_at', '').replace('Z', '+00:00')),
                    sentiment_score=0.0,
                    sentiment_label='neutral',
                    confidence=0.0,
                    keywords=[],
                    relevance_score=0.0
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            return []

class NewsAPIClient:
    """News API client for general crypto news"""
    
    def __init__(self):
        self.base_url = system_config.NEWS_API_URL
        self.api_key = api_config.get_api_key('NEWS_API_KEY')
        self.session = requests.Session()
    
    def get_news(self, query: str = "cryptocurrency OR bitcoin OR ethereum", limit: int = 20) -> List[NewsArticle]:
        """Get crypto news from News API"""
        if not self.api_key:
            logger.warning("News API key not found")
            return []
        
        try:
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': limit,
                'apiKey': self.api_key
            }
            
            response = self.session.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                if item.get('title') and item.get('publishedAt'):
                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('description', '') or item.get('content', ''),
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        url=item.get('url', ''),
                        published_at=datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                        sentiment_score=0.0,
                        sentiment_label='neutral',
                        confidence=0.0,
                        keywords=[],
                        relevance_score=0.0
                    )
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching News API articles: {e}")
            return []

class RedditClient:
    """Reddit client for crypto community sentiment"""
    
    def __init__(self):
        self.client_id = api_config.get_api_key('REDDIT_CLIENT_ID')
        self.client_secret = api_config.get_api_key('REDDIT_CLIENT_SECRET')
        self.reddit = None
        self.subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets', 'cryptonews']
        
        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent='CryptoPredictorBot/1.0'
                )
            except Exception as e:
                logger.error(f"Error initializing Reddit client: {e}")
    
    def get_posts(self, limit: int = 20) -> List[NewsArticle]:
        """Get hot posts from crypto subreddits"""
        if not self.reddit:
            logger.warning("Reddit client not initialized")
            return []
        
        try:
            articles = []
            
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    for submission in subreddit.hot(limit=limit // len(self.subreddits)):
                        if submission.is_self:  # Text posts
                            content = submission.selftext
                        else:
                            content = submission.title
                        
                        article = NewsArticle(
                            title=submission.title,
                            content=content,
                            source=f"Reddit r/{subreddit_name}",
                            url=f"https://reddit.com{submission.permalink}",
                            published_at=datetime.fromtimestamp(submission.created_utc),
                            sentiment_score=0.0,
                            sentiment_label='neutral',
                            confidence=0.0,
                            keywords=[],
                            relevance_score=submission.score / 100.0  # Use Reddit score as relevance
                        )
                        articles.append(article)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching from r/{subreddit_name}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []

class CryptoNewsAggregator:
    """Main news aggregator that combines multiple sources"""
    
    def __init__(self):
        self.finbert = FinBERTSentimentAnalyzer()
        self.cryptopanic = CryptoPanicClient()
        self.newsapi = NewsAPIClient()
        self.reddit = RedditClient()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def collect_news(self, currencies: List[str] = None, limit_per_source: int = 10) -> List[NewsArticle]:
        """Collect news from all sources"""
        if currencies is None:
            currencies = system_config.TRADING_PAIRS
        
        # Check cache
        cache_key = f"news_{'_'.join(currencies)}"
        if cache_key in self.cache:
            cache_time, cached_articles = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_articles
        
        all_articles = []
        
        # Collect from all sources in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # CryptoPanic
            futures.append(executor.submit(self.cryptopanic.get_news, currencies, limit_per_source))
            
            # News API
            query = " OR ".join([f"{curr} OR {curr.lower()}" for curr in currencies])
            futures.append(executor.submit(self.newsapi.get_news, query, limit_per_source))
            
            # Reddit
            futures.append(executor.submit(self.reddit.get_posts, limit_per_source))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error collecting news from source: {e}")
        
        # Remove duplicates based on title similarity
        unique_articles = self._remove_duplicates(all_articles)
        
        # Add sentiment analysis
        self._add_sentiment_analysis(unique_articles)
        
        # Add keywords and relevance
        self._add_keywords_and_relevance(unique_articles, currencies)
        
        # Sort by relevance and recency
        unique_articles.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)
        
        # Cache results
        self.cache[cache_key] = (time.time(), unique_articles)
        
        logger.info(f"Collected {len(unique_articles)} unique news articles")
        return unique_articles
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title
            title_key = re.sub(r'[^\w\s]', '', article.title.lower())
            title_key = ' '.join(title_key.split())
            
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def _add_sentiment_analysis(self, articles: List[NewsArticle]):
        """Add sentiment analysis to articles"""
        texts = [f"{article.title} {article.content}" for article in articles]
        
        # Batch analyze sentiments
        sentiments = self.finbert.batch_analyze(texts)
        
        for article, (score, label, confidence) in zip(articles, sentiments):
            article.sentiment_score = score
            article.sentiment_label = label
            article.confidence = confidence
    
    def _add_keywords_and_relevance(self, articles: List[NewsArticle], currencies: List[str]):
        """Add keywords and calculate relevance scores"""
        crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft', 'trading', 'price', 'bull', 'bear']
        
        for article in articles:
            text = f"{article.title} {article.content}".lower()
            
            # Extract keywords
            keywords = []
            for keyword in crypto_keywords:
                if keyword in text:
                    keywords.append(keyword)
            
            # Add currency mentions
            for currency in currencies:
                if currency.lower() in text:
                    keywords.append(currency.lower())
            
            article.keywords = keywords
            
            # Calculate relevance score
            relevance = 0.0
            
            # Currency mentions
            for currency in currencies:
                if currency.lower() in text:
                    relevance += 0.3
            
            # Keyword matches
            relevance += len(keywords) * 0.1
            
            # Sentiment confidence
            relevance += article.confidence * 0.2
            
            # Recency bonus (newer articles get higher score)
            hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
            recency_bonus = max(0, 1 - hours_old / 24)  # Bonus decreases over 24 hours
            relevance += recency_bonus * 0.2
            
            article.relevance_score = min(relevance, 1.0)
    
    def get_sentiment_summary(self, articles: List[NewsArticle]) -> Dict:
        """Get overall sentiment summary"""
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': 0.0,
                'total_articles': 0
            }
        
        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        confidence_sum = 0
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in articles:
            weight = article.relevance_score * article.confidence
            weighted_sentiment += article.sentiment_score * weight
            total_weight += weight
            confidence_sum += article.confidence
            sentiment_counts[article.sentiment_label] += 1
        
        overall_sentiment_score = weighted_sentiment / total_weight if total_weight > 0 else 0
        avg_confidence = confidence_sum / len(articles) if articles else 0
        
        # Determine overall sentiment label
        if overall_sentiment_score > 0.05:
            overall_sentiment = 'positive'
        elif overall_sentiment_score < -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': overall_sentiment_score,
            'positive_count': sentiment_counts['positive'],
            'negative_count': sentiment_counts['negative'],
            'neutral_count': sentiment_counts['neutral'],
            'confidence': avg_confidence,
            'total_articles': len(articles)
        }

# Create global instance
news_aggregator = CryptoNewsAggregator()