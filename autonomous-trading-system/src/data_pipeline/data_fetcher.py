import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils import log, config

class DataFetcher:
    
    def __init__(self):
        self.alpha_vantage_key = config.get('api_keys.alpha_vantage', '')
        self.finnhub_key = config.get('api_keys.finnhub', '')
        self.alpaca_api_key = config.get('api_keys.alpaca_api_key', '')
        self.alpaca_secret_key = config.get('api_keys.alpaca_secret_key', '')
        self.use_alpaca = config.get('data.use_alpaca', False)
        self.cache = {}
        
        if self.use_alpaca and self.alpaca_api_key and self.alpaca_secret_key:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                self.alpaca_client = StockHistoricalDataClient(self.alpaca_api_key, self.alpaca_secret_key)
                log.info("✓ Alpaca data client initialized successfully")
            except Exception as e:
                log.error(f"Failed to initialize Alpaca client: {e}")
                self.use_alpaca = False
                self.alpaca_client = None
        else:
            self.alpaca_client = None
            if self.use_alpaca:
                log.warning(f"Alpaca enabled but missing credentials: api_key={bool(self.alpaca_api_key)}, secret_key={bool(self.alpaca_secret_key)}")
    
    def fetch_from_alpaca(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.alpaca_client.get_stock_bars(request_params)
            df = bars.df
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df = df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df['Symbol'] = symbol
            
            log.info(f"Successfully fetched {len(df)} rows from Alpaca for {symbol}")
            return df
            
        except Exception as e:
            log.error(f"Error fetching Alpaca data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        try:
            if start_date is None:
                days = config.get('data.historical_days', 365)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            log.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            if self.use_alpaca and self.alpaca_client:
                df = self.fetch_from_alpaca(symbol, start_date, end_date)
                if not df.empty:
                    return df
                log.warning(f"Alpaca returned no data for {symbol}, falling back to yfinance")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                log.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            df.reset_index(inplace=True)
            df['Symbol'] = symbol
            
            log.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            log.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_realtime_data(self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            data = {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            log.error(f"Error fetching realtime data for {symbol}: {str(e)}")
            return {}
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        import time
        data = {}
        for i, symbol in enumerate(symbols):
            df = self.fetch_historical_data(symbol, start_date, end_date)
            if not df.empty:
                data[symbol] = df
            
            if i < len(symbols) - 1:
                time.sleep(2)
        return data
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        try:
            if not self.finnhub_key:
                log.warning("Finnhub API key not configured")
                return []
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news = response.json()
            log.info(f"Fetched {len(news)} news articles for {symbol}")
            return news
            
        except Exception as e:
            log.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def get_sp500_symbols(self) -> List[str]:
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            log.info(f"Fetched {len(symbols)} S&P 500 symbols")
            return symbols
        except Exception as e:
            log.error(f"Error fetching S&P 500 symbols: {str(e)}")
            return []
    
    def get_nasdaq100_symbols(self) -> List[str]:
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            df = tables[4]
            symbols = df['Ticker'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            log.info(f"Fetched {len(symbols)} NASDAQ-100 symbols")
            return symbols
        except Exception as e:
            log.error(f"Error fetching NASDAQ-100 symbols: {str(e)}")
            return []
    
    def get_watchlist_symbols(self) -> List[str]:
        symbols = set()
        
        watchlists = config.get('symbols.watchlists', [])
        if 'SP500' in watchlists:
            symbols.update(self.get_sp500_symbols())
        if 'NASDAQ100' in watchlists:
            symbols.update(self.get_nasdaq100_symbols())
        
        custom = config.get('symbols.custom_symbols', [])
        symbols.update(custom)
        
        return list(symbols)
    
    def calculate_volatility(self, symbol: str, days: int = 30) -> float:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.fetch_historical_data(
                symbol, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty or len(df) < 2:
                return 0.0
            
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std()
            
            return volatility
            
        except Exception as e:
            log.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.0
    
    def filter_symbols_by_criteria(self, symbols: List[str]) -> List[str]:
        filtered = []
        criteria = config.get('symbols.discovery.criteria', {})
        
        min_volume = criteria.get('min_volume', 1000000)
        min_price = criteria.get('min_price', 5.0)
        max_price = criteria.get('max_price', 1000.0)
        min_volatility = criteria.get('min_volatility', 0.02)
        
        for symbol in symbols:
            try:
                data = self.fetch_realtime_data(symbol)
                if not data:
                    continue
                
                price = data.get('price', 0)
                volume = data.get('volume', 0)
                volatility = self.calculate_volatility(symbol)
                
                if (volume >= min_volume and 
                    min_price <= price <= max_price and 
                    volatility >= min_volatility):
                    filtered.append(symbol)
                    log.info(f"Symbol {symbol} passed criteria: price={price}, volume={volume}, volatility={volatility}")
                
            except Exception as e:
                log.error(f"Error filtering {symbol}: {str(e)}")
                continue
        
        log.info(f"Filtered {len(filtered)} symbols from {len(symbols)} candidates")
        return filtered
