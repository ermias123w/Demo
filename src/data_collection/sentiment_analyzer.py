import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analysis using FinBERT and other models"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.initialize_models()
        
        # Sentiment mapping
        self.sentiment_mapping = {
            'positive': 1,
            'neutral': 0,
            'negative': -1,
            'POSITIVE': 1,
            'NEUTRAL': 0,
            'NEGATIVE': -1
        }
        
        # Crypto-specific keywords for filtering and weighting
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc', 'satoshi'],
            'ethereum': ['ethereum', 'eth', 'ether', 'vitalik'],
            'general': ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'altcoin', 'hodl']
        }
        
        # Market impact keywords
        self.bullish_keywords = [
            'bullish', 'moon', 'rocket', 'pump', 'surge', 'rally', 'breakout',
            'adoption', 'institutional', 'etf', 'approved', 'partnership',
            'upgrade', 'launch', 'innovation', 'breakthrough', 'positive'
        ]
        
        self.bearish_keywords = [
            'bearish', 'crash', 'dump', 'sell-off', 'decline', 'drop',
            'regulation', 'ban', 'hack', 'exploit', 'scam', 'lawsuit',
            'fear', 'uncertainty', 'doubt', 'negative', 'bearish'
        ]
    
    def initialize_models(self):
        """Initialize FinBERT and other sentiment models"""
        try:
            # Try to load FinBERT
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}")
            logger.info("Falling back to basic sentiment analysis...")
            
            # Fallback to basic sentiment analysis
            self.sentiment_pipeline = None
            self.tokenizer = None
            self.model = None
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\-\:]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate to prevent token limit issues
        if len(text) > 512:
            text = text[:512]
        
        return text.strip()
    
    def analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        try:
            if not self.sentiment_pipeline:
                raise ValueError("FinBERT model not available")
            
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
            
            # Get sentiment scores
            results = self.sentiment_pipeline(cleaned_text)
            
            # Process results
            if results and len(results) > 0:
                # Get the result with highest score
                best_result = max(results[0], key=lambda x: x['score'])
                
                label = best_result['label'].lower()
                score = best_result['score']
                
                # Convert to standardized format
                sentiment_score = self.sentiment_mapping.get(label, 0)
                
                return {
                    'label': label,
                    'score': sentiment_score,
                    'confidence': score,
                    'raw_results': results[0]
                }
            
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using TextBlob"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
            
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to label
            if polarity > 0.1:
                label = 'positive'
                score = 1
            elif polarity < -0.1:
                label = 'negative'
                score = -1
            else:
                label = 'neutral'
                score = 0
            
            # Use absolute polarity as confidence
            confidence = abs(polarity)
            
            return {
                'label': label,
                'score': score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def analyze_crypto_specific_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with crypto-specific context"""
        try:
            text_lower = text.lower()
            
            # Check for crypto-specific bullish/bearish keywords
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            
            # Calculate keyword-based sentiment
            keyword_sentiment = 0
            if bullish_count > bearish_count:
                keyword_sentiment = 1
            elif bearish_count > bullish_count:
                keyword_sentiment = -1
            
            # Get symbol-specific relevance
            symbol_relevance = self.get_symbol_relevance(text)
            
            return {
                'keyword_sentiment': keyword_sentiment,
                'bullish_keywords': bullish_count,
                'bearish_keywords': bearish_count,
                'symbol_relevance': symbol_relevance
            }
            
        except Exception as e:
            logger.error(f"Error in crypto-specific sentiment analysis: {e}")
            return {
                'keyword_sentiment': 0,
                'bullish_keywords': 0,
                'bearish_keywords': 0,
                'symbol_relevance': {}
            }
    
    def get_symbol_relevance(self, text: str) -> Dict[str, float]:
        """Calculate relevance score for each crypto symbol"""
        try:
            text_lower = text.lower()
            relevance = {}
            
            for symbol, keywords in self.crypto_keywords.items():
                if symbol == 'general':
                    continue
                
                # Count symbol-specific keywords
                symbol_count = sum(1 for keyword in keywords if keyword in text_lower)
                
                # Count general crypto keywords
                general_count = sum(1 for keyword in self.crypto_keywords['general'] if keyword in text_lower)
                
                # Calculate relevance score
                total_words = len(text_lower.split())
                if total_words > 0:
                    relevance[symbol] = (symbol_count * 2 + general_count * 0.5) / total_words
                else:
                    relevance[symbol] = 0.0
            
            return relevance
            
        except Exception as e:
            logger.error(f"Error calculating symbol relevance: {e}")
            return {}
    
    def analyze_sentiment(self, text: str, symbol: str = None) -> Dict[str, Any]:
        """Main sentiment analysis function"""
        try:
            if not text:
                return {
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'symbol_relevance': 0.0,
                    'method': 'none'
                }
            
            # Try FinBERT first
            if self.sentiment_pipeline:
                finbert_result = self.analyze_with_finbert(text)
                method = 'finbert'
            else:
                # Fallback to TextBlob
                finbert_result = self.analyze_with_textblob(text)
                method = 'textblob'
            
            # Get crypto-specific analysis
            crypto_analysis = self.analyze_crypto_specific_sentiment(text)
            
            # Combine results
            base_score = finbert_result['score']
            base_confidence = finbert_result['confidence']
            
            # Adjust score based on crypto-specific keywords
            keyword_adjustment = crypto_analysis['keyword_sentiment'] * 0.2
            adjusted_score = base_score + keyword_adjustment
            
            # Clamp score to [-1, 1]
            adjusted_score = max(-1, min(1, adjusted_score))
            
            # Calculate symbol relevance
            symbol_relevance = 0.0
            if symbol:
                symbol_relevance = crypto_analysis['symbol_relevance'].get(symbol, 0.0)
            
            # Adjust confidence based on relevance
            adjusted_confidence = base_confidence * (1 + symbol_relevance)
            adjusted_confidence = min(1.0, adjusted_confidence)
            
            # Determine final label
            if adjusted_score > 0.1:
                final_label = 'positive'
            elif adjusted_score < -0.1:
                final_label = 'negative'
            else:
                final_label = 'neutral'
            
            return {
                'sentiment_label': final_label,
                'sentiment_score': adjusted_score,
                'confidence': adjusted_confidence,
                'symbol_relevance': symbol_relevance,
                'method': method,
                'base_score': base_score,
                'keyword_adjustment': keyword_adjustment,
                'bullish_keywords': crypto_analysis['bullish_keywords'],
                'bearish_keywords': crypto_analysis['bearish_keywords']
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'symbol_relevance': 0.0,
                'method': 'error'
            }
    
    def analyze_news_batch(self, news_articles: List[Dict[str, Any]], 
                          symbol: str = None) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of news articles"""
        try:
            results = []
            
            for article in news_articles:
                # Combine title and description for analysis
                text_parts = []
                if article.get('title'):
                    text_parts.append(article['title'])
                if article.get('description'):
                    text_parts.append(article['description'])
                
                combined_text = ' '.join(text_parts)
                
                # Analyze sentiment
                sentiment_result = self.analyze_sentiment(combined_text, symbol)
                
                # Add article metadata
                result = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published_at': article.get('published_at', ''),
                    'timestamp': datetime.now().isoformat(),
                    **sentiment_result
                }
                
                results.append(result)
            
            logger.info(f"Analyzed sentiment for {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return []
    
    def calculate_aggregated_sentiment(self, sentiment_results: List[Dict[str, Any]], 
                                     time_window_hours: int = 24) -> Dict[str, Any]:
        """Calculate aggregated sentiment metrics"""
        try:
            if not sentiment_results:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'article_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'weighted_sentiment': 0.0
                }
            
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            filtered_results = []
            for result in sentiment_results:
                try:
                    published_at = datetime.fromisoformat(result.get('published_at', '').replace('Z', '+00:00'))
                    if published_at >= cutoff_time:
                        filtered_results.append(result)
                except:
                    # Include if we can't parse the date
                    filtered_results.append(result)
            
            if not filtered_results:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'article_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'weighted_sentiment': 0.0
                }
            
            # Calculate sentiment statistics
            sentiment_scores = [result['sentiment_score'] for result in filtered_results]
            confidences = [result['confidence'] for result in filtered_results]
            
            # Simple average
            overall_sentiment = np.mean(sentiment_scores)
            
            # Weighted average (by confidence)
            total_weight = sum(confidences)
            if total_weight > 0:
                weighted_sentiment = sum(score * conf for score, conf in zip(sentiment_scores, confidences)) / total_weight
            else:
                weighted_sentiment = overall_sentiment
            
            # Count sentiment labels
            positive_count = sum(1 for result in filtered_results if result['sentiment_label'] == 'positive')
            negative_count = sum(1 for result in filtered_results if result['sentiment_label'] == 'negative')
            neutral_count = sum(1 for result in filtered_results if result['sentiment_label'] == 'neutral')
            
            # Overall confidence
            overall_confidence = np.mean(confidences)
            
            # Determine overall label
            if weighted_sentiment > 0.1:
                overall_label = 'positive'
            elif weighted_sentiment < -0.1:
                overall_label = 'negative'
            else:
                overall_label = 'neutral'
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_label': overall_label,
                'confidence': overall_confidence,
                'article_count': len(filtered_results),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'weighted_sentiment': weighted_sentiment,
                'sentiment_distribution': {
                    'positive': positive_count / len(filtered_results),
                    'negative': negative_count / len(filtered_results),
                    'neutral': neutral_count / len(filtered_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregated sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'weighted_sentiment': 0.0
            }
    
    def get_sentiment_trends(self, sentiment_results: List[Dict[str, Any]], 
                           hours_back: int = 48) -> Dict[str, Any]:
        """Calculate sentiment trends over time"""
        try:
            if not sentiment_results:
                return {'trend': 'neutral', 'slope': 0.0, 'hourly_sentiment': []}
            
            # Group by hour
            hourly_sentiment = {}
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for result in sentiment_results:
                try:
                    published_at = datetime.fromisoformat(result.get('published_at', '').replace('Z', '+00:00'))
                    if published_at >= cutoff_time:
                        hour_key = published_at.strftime('%Y-%m-%d %H:00:00')
                        
                        if hour_key not in hourly_sentiment:
                            hourly_sentiment[hour_key] = []
                        
                        hourly_sentiment[hour_key].append(result['sentiment_score'])
                except:
                    continue
            
            # Calculate hourly averages
            hourly_averages = []
            for hour, scores in sorted(hourly_sentiment.items()):
                avg_sentiment = np.mean(scores)
                hourly_averages.append({
                    'hour': hour,
                    'sentiment': avg_sentiment,
                    'article_count': len(scores)
                })
            
            # Calculate trend
            if len(hourly_averages) < 2:
                return {
                    'trend': 'neutral',
                    'slope': 0.0,
                    'hourly_sentiment': hourly_averages
                }
            
            # Simple linear regression for trend
            x = np.arange(len(hourly_averages))
            y = [avg['sentiment'] for avg in hourly_averages]
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            
            # Determine trend direction
            if slope > 0.01:
                trend = 'bullish'
            elif slope < -0.01:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'trend': trend,
                'slope': slope,
                'hourly_sentiment': hourly_averages,
                'latest_sentiment': hourly_averages[-1]['sentiment'] if hourly_averages else 0.0,
                'sentiment_change': (hourly_averages[-1]['sentiment'] - hourly_averages[0]['sentiment']) if len(hourly_averages) >= 2 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment trends: {e}")
            return {'trend': 'neutral', 'slope': 0.0, 'hourly_sentiment': []}
    
    def get_sentiment_strength(self, sentiment_results: List[Dict[str, Any]]) -> float:
        """Calculate sentiment strength (0-1 scale)"""
        try:
            if not sentiment_results:
                return 0.0
            
            # Calculate average absolute sentiment score weighted by confidence
            total_weight = 0
            weighted_strength = 0
            
            for result in sentiment_results:
                sentiment_score = result.get('sentiment_score', 0)
                confidence = result.get('confidence', 0)
                strength = abs(sentiment_score) * confidence
                
                weighted_strength += strength
                total_weight += confidence
            
            if total_weight > 0:
                return weighted_strength / total_weight
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sentiment strength: {e}")
            return 0.0