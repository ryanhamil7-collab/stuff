import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data_pipeline import DataFetcher
from src.models import SentimentAnalyzer
from src.utils import log, config

class SymbolDiscovery:
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scaler = StandardScaler()
        log.info("SymbolDiscovery initialized")
    
    def discover_symbols(self) -> List[str]:
        log.info("Starting symbol discovery process")
        
        base_symbols = self.fetcher.get_watchlist_symbols()
        
        log.info(f"Analyzing {len(base_symbols[:100])} candidate symbols")
        
        filtered_symbols = self.fetcher.filter_symbols_by_criteria(base_symbols[:100])
        
        if len(filtered_symbols) > 20:
            clustered_symbols = self.cluster_symbols(filtered_symbols[:50])
        else:
            clustered_symbols = filtered_symbols
        
        log.info(f"Discovered {len(clustered_symbols)} symbols")
        return clustered_symbols
    
    def cluster_symbols(self, symbols: List[str], n_clusters: int = 10) -> List[str]:
        log.info(f"Clustering {len(symbols)} symbols into {n_clusters} groups")
        
        features_list = []
        valid_symbols = []
        
        for symbol in symbols:
            try:
                features = self.extract_symbol_features(symbol)
                if features is not None:
                    features_list.append(features)
                    valid_symbols.append(symbol)
            except Exception as e:
                log.error(f"Error extracting features for {symbol}: {str(e)}")
                continue
        
        if len(features_list) < n_clusters:
            log.warning(f"Not enough symbols for clustering, returning all {len(valid_symbols)}")
            return valid_symbols
        
        features_array = np.array(features_list)
        features_scaled = self.scaler.fit_transform(features_array)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(valid_symbols)), random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        selected_symbols = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                distances = kmeans.transform(features_scaled[cluster_indices])[:, cluster_id]
                best_idx = cluster_indices[np.argmin(distances)]
                selected_symbols.append(valid_symbols[best_idx])
        
        log.info(f"Selected {len(selected_symbols)} representative symbols from clusters")
        return selected_symbols
    
    def extract_symbol_features(self, symbol: str) -> np.ndarray:
        try:
            data = self.fetcher.fetch_realtime_data(symbol)
            
            if not data:
                return None
            
            volatility = self.fetcher.calculate_volatility(symbol, days=30)
            
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            market_cap = data.get('market_cap', 0)
            pe_ratio = data.get('pe_ratio', 0)
            
            day_range = (data.get('day_high', 0) - data.get('day_low', 0)) / price if price > 0 else 0
            year_range = (data.get('fifty_two_week_high', 0) - data.get('fifty_two_week_low', 0)) / price if price > 0 else 0
            
            features = np.array([
                np.log1p(price),
                np.log1p(volume),
                np.log1p(market_cap),
                pe_ratio if pe_ratio > 0 else 0,
                volatility,
                day_range,
                year_range
            ])
            
            return features
            
        except Exception as e:
            log.error(f"Error extracting features for {symbol}: {str(e)}")
            return None
    
    def rank_symbols_by_potential(self, symbols: List[str]) -> List[Dict]:
        log.info(f"Ranking {len(symbols)} symbols by potential")
        
        ranked_symbols = []
        
        for symbol in symbols:
            try:
                score = self.calculate_potential_score(symbol)
                ranked_symbols.append({
                    'symbol': symbol,
                    'score': score
                })
            except Exception as e:
                log.error(f"Error ranking {symbol}: {str(e)}")
                continue
        
        ranked_symbols.sort(key=lambda x: x['score'], reverse=True)
        
        log.info(f"Ranked {len(ranked_symbols)} symbols")
        return ranked_symbols
    
    def calculate_potential_score(self, symbol: str) -> float:
        score = 0.0
        
        volatility = self.fetcher.calculate_volatility(symbol, days=30)
        score += min(volatility * 10, 3.0)
        
        data = self.fetcher.fetch_realtime_data(symbol)
        if data:
            volume = data.get('volume', 0)
            if volume > 5000000:
                score += 2.0
            elif volume > 1000000:
                score += 1.0
        
        news = self.fetcher.fetch_news(symbol, days=3)
        if news:
            sentiment_result = self.sentiment_analyzer.analyze_news_batch(news)
            sentiment_score = sentiment_result.get('overall_sentiment', 0)
            score += sentiment_score * 2
        
        return score
