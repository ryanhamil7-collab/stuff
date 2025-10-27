import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from src.data_pipeline import DataFetcher
from src.utils import log, config
from src.utils.indicators import TechnicalIndicators

class DataAgent:
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.cache = {}
        log.info("DataAgent initialized")
    
    def collect_market_data(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        log.info(f"Collecting market data for {len(symbols)} symbols")
        
        if start_date is None:
            from datetime import timedelta
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            log.info(f"Using recent date range for live trading: {start_date} to {end_date}")
        else:
            log.info(f"Date range: {start_date} to {end_date}")
        
        data = self.fetcher.fetch_multiple_symbols(symbols, start_date, end_date)
        
        log.info(f"Successfully collected data for {len(data)} symbols")
        return data
    
    def process_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log.info(f"Processing data for {len(data)} symbols")
        
        processed_data = {}
        indicator_config = config.get('indicators', {})
        
        for symbol, df in data.items():
            try:
                if df.empty or len(df) < 50:
                    log.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                log.info(f"Processing {symbol}: {len(df)} rows before indicators")
                df_processed = self.indicators.calculate_all_indicators(df, indicator_config)
                
                essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df_processed = df_processed.dropna(subset=essential_cols)
                
                if not df_processed.empty:
                    processed_data[symbol] = df_processed
                    log.info(f"Successfully processed {symbol}: {len(df_processed)} rows after processing")
                
            except Exception as e:
                log.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        log.info(f"Successfully processed {len(processed_data)} symbols")
        return processed_data
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        log.info(f"Fetching realtime data for {len(symbols)} symbols")
        
        realtime_data = {}
        for symbol in symbols:
            data = self.fetcher.fetch_realtime_data(symbol)
            if data:
                realtime_data[symbol] = data
        
        return realtime_data
    
    def get_news_data(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        log.info(f"Fetching news for {len(symbols)} symbols")
        
        lookback_days = config.get('sentiment.lookback_days', 7)
        news_data = {}
        
        for symbol in symbols:
            news = self.fetcher.fetch_news(symbol, lookback_days)
            if news:
                news_data[symbol] = news
        
        return news_data
    
    def discover_symbols(self) -> List[str]:
        log.info("Starting symbol discovery")
        
        intraday_symbols = config.get('autopilot.hybrid.intraday_symbols', [])
        interday_symbols = config.get('autopilot.hybrid.interday_symbols', [])
        
        discovered_symbols = list(set(intraday_symbols + interday_symbols))
        
        if discovered_symbols:
            log.info(f"Using {len(discovered_symbols)} auto-discovered symbols from web scraping")
            return discovered_symbols
        
        custom_symbols = config.get('symbols.custom_symbols', [])
        if custom_symbols:
            log.info(f"Using {len(custom_symbols)} custom symbols from config")
            return custom_symbols
        
        log.warning("No discovered symbols found, falling back to watchlist")
        all_symbols = self.fetcher.get_watchlist_symbols()
        filtered_symbols = self.fetcher.filter_symbols_by_criteria(all_symbols[:100])
        
        log.info(f"Discovered {len(filtered_symbols)} symbols from watchlist")
        return filtered_symbols
    
    def get_market_regime(self, symbol: str, data: pd.DataFrame) -> str:
        regime = self.indicators.detect_market_regime(data)
        log.info(f"Market regime for {symbol}: {regime}")
        return regime
    
    def generate_technical_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log.info(f"Generating technical signals for {len(data)} symbols")
        
        signals = {}
        for symbol, df in data.items():
            try:
                df_with_signals = self.indicators.generate_signals(df)
                signals[symbol] = df_with_signals
            except Exception as e:
                log.error(f"Error generating signals for {symbol}: {str(e)}")
                continue
        
        return signals
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict:
        summary = {
            'num_symbols': len(data),
            'symbols': list(data.keys()),
            'date_range': {},
            'total_rows': 0
        }
        
        for symbol, df in data.items():
            if not df.empty:
                summary['date_range'][symbol] = {
                    'start': df['Date'].min() if 'Date' in df.columns else df.index.min(),
                    'end': df['Date'].max() if 'Date' in df.columns else df.index.max(),
                    'rows': len(df)
                }
                summary['total_rows'] += len(df)
        
        return summary
    
    def run(self) -> Dict:
        log.info("DataAgent starting data collection and processing")
        
        symbols = self.discover_symbols()
        
        if not symbols:
            log.error("No symbols discovered")
            return {}
        
        log.info(f"Working with {len(symbols)} symbols")
        
        market_data = self.collect_market_data(symbols[:20])
        
        processed_data = self.process_data(market_data)
        
        signals = self.generate_technical_signals(processed_data)
        
        news_data = self.get_news_data(list(processed_data.keys())[:10])
        
        result = {
            'processed_data': processed_data,
            'signals': signals,
            'news': news_data,
            'summary': self.get_data_summary(processed_data),
            'timestamp': datetime.now()
        }
        
        log.info("DataAgent completed successfully")
        return result
