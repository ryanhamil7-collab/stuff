import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict
import numpy as np
from src.utils import log, config

class SentimentAnalyzer:
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = config.get('sentiment.model', 'ProsusAI/finbert')
        
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        log.info(f"Loading sentiment model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=512,
                truncation=True
            )
            
            log.info("Sentiment model loaded successfully")
            
        except Exception as e:
            log.error(f"Error loading sentiment model: {str(e)}")
            self.sentiment_pipeline = None
    
    def analyze_text(self, text: str) -> Dict:
        if not self.sentiment_pipeline:
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            
            label_map = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0,
                'POSITIVE': 1.0,
                'NEGATIVE': -1.0,
                'NEUTRAL': 0.0
            }
            
            sentiment_score = label_map.get(result['label'], 0.0) * result['score']
            
            return {
                'label': result['label'].lower(),
                'score': result['score'],
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            log.error(f"Error analyzing text: {str(e)}")
            return {'label': 'neutral', 'score': 0.0, 'sentiment_score': 0.0}
    
    def analyze_news_batch(self, news_articles: List[Dict]) -> Dict:
        if not news_articles:
            return {
                'overall_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles_analyzed': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in news_articles:
            text = article.get('headline', '') + ' ' + article.get('summary', '')
            
            if not text.strip():
                continue
            
            result = self.analyze_text(text)
            sentiments.append(result['sentiment_score'])
            
            if result['label'] == 'positive':
                positive_count += 1
            elif result['label'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles_analyzed': len(sentiments),
            'sentiment_std': float(np.std(sentiments)) if sentiments else 0.0
        }
    
    def analyze_symbol_sentiment(self, symbol: str, news_data: List[Dict]) -> Dict:
        log.info(f"Analyzing sentiment for {symbol}")
        
        sentiment_result = self.analyze_news_batch(news_data)
        sentiment_result['symbol'] = symbol
        
        threshold = config.get('sentiment.sentiment_threshold', 0.1)
        
        if sentiment_result['overall_sentiment'] > threshold:
            sentiment_result['signal'] = 'bullish'
        elif sentiment_result['overall_sentiment'] < -threshold:
            sentiment_result['signal'] = 'bearish'
        else:
            sentiment_result['signal'] = 'neutral'
        
        log.info(f"Sentiment for {symbol}: {sentiment_result['signal']} (score: {sentiment_result['overall_sentiment']:.3f})")
        
        return sentiment_result
    
    def get_sentiment_signals(self, news_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        log.info(f"Analyzing sentiment for {len(news_data)} symbols")
        
        sentiment_signals = {}
        
        for symbol, news in news_data.items():
            sentiment_signals[symbol] = self.analyze_symbol_sentiment(symbol, news)
        
        return sentiment_signals
