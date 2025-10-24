"""
Improved data fetcher with robust error handling, validation, and caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from functools import lru_cache
import time

from src.utils import log, config
from src.utils.exceptions import (
    DataFetchError, InvalidSymbolError, RateLimitError, 
    NetworkError, InsufficientDataError
)
from src.utils.validators import (
    validate_symbol, validate_symbols, validate_date_range,
    validate_positive_number
)


class ImprovedDataFetcher:
    """
    Enhanced data fetcher with comprehensive error handling, validation, and caching.
    """
    
    def __init__(self):
        """Initialize the data fetcher with API keys and cache."""
        self.alpha_vantage_key = config.get('api_keys.alpha_vantage', '')
        self.finnhub_key = config.get('api_keys.finnhub', '')
        self.cache = {}
        self.rate_limit_tracker = {}
        self.max_retries = config.get('data.max_retries', 3)
        log.info("ImprovedDataFetcher initialized")
    
    def _check_rate_limit(self, service: str, max_calls_per_minute: int = 60) -> None:
        """
        Check if we're within rate limits for a service.
        
        Args:
            service: Name of the service
            max_calls_per_minute: Maximum calls allowed per minute
            
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        now = time.time()
        if service not in self.rate_limit_tracker:
            self.rate_limit_tracker[service] = []
        
        self.rate_limit_tracker[service] = [
            t for t in self.rate_limit_tracker[service] 
            if now - t < 60
        ]
        
        if len(self.rate_limit_tracker[service]) >= max_calls_per_minute:
            raise RateLimitError(f"Rate limit exceeded for {service}")
        
        self.rate_limit_tracker[service].append(now)
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """Generate a cache key for data requests."""
        return f"{symbol}_{start_date}_{end_date}_{interval}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError))
    )
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol with robust error handling.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1h', etc.)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with historical data
            
        Raises:
            InvalidSymbolError: If symbol is invalid
            DataFetchError: If data fetch fails
            InsufficientDataError: If insufficient data is returned
        """
        try:
            validate_symbol(symbol)
            symbol = symbol.strip().upper()
            
            if start_date is None:
                days = config.get('data.historical_days', 365)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            validate_date_range(start_date, end_date)
            
            cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
            if use_cache and cache_key in self.cache:
                log.info(f"Using cached data for {symbol}")
                return self.cache[cache_key].copy()
            
            self._check_rate_limit('yfinance', max_calls_per_minute=60)
            
            log.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise InsufficientDataError(f"No data found for {symbol}")
            
            if len(df) < 10:
                log.warning(f"Limited data for {symbol}: only {len(df)} rows")
            
            df.reset_index(inplace=True)
            df['Symbol'] = symbol
            
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataFetchError(f"Missing required columns for {symbol}: {missing_columns}")
            
            if df['Close'].isna().sum() > len(df) * 0.1:
                log.warning(f"High number of missing values in {symbol} data")
            
            if use_cache:
                self.cache[cache_key] = df.copy()
            
            log.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except InvalidSymbolError:
            raise
        except InsufficientDataError:
            raise
        except RateLimitError:
            raise
        except Exception as e:
            log.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
            raise DataFetchError(f"Failed to fetch data for {symbol}: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError))
    )
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """
        Fetch real-time market data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with real-time data
            
        Raises:
            InvalidSymbolError: If symbol is invalid
            DataFetchError: If data fetch fails
        """
        try:
            validate_symbol(symbol)
            symbol = symbol.strip().upper()
            
            self._check_rate_limit('yfinance_realtime', max_calls_per_minute=30)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                raise InvalidSymbolError(f"Invalid or delisted symbol: {symbol}")
            
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
            
            if data['price'] == 0:
                log.warning(f"Zero price for {symbol}, data may be stale")
            
            return data
            
        except InvalidSymbolError:
            raise
        except Exception as e:
            log.error(f"Error fetching realtime data for {symbol}: {str(e)}", exc_info=True)
            raise DataFetchError(f"Failed to fetch realtime data for {symbol}: {str(e)}") from e
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols with parallel processing.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        try:
            validated_symbols = validate_symbols(symbols)
            log.info(f"Fetching data for {len(validated_symbols)} symbols")
            
            data = {}
            failed_symbols = []
            
            for symbol in validated_symbols:
                try:
                    df = self.fetch_historical_data(symbol, start_date, end_date)
                    if not df.empty:
                        data[symbol] = df
                except (InvalidSymbolError, InsufficientDataError) as e:
                    log.warning(f"Skipping {symbol}: {str(e)}")
                    failed_symbols.append(symbol)
                except DataFetchError as e:
                    log.error(f"Failed to fetch {symbol}: {str(e)}")
                    failed_symbols.append(symbol)
            
            if failed_symbols:
                log.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:10]}")
            
            log.info(f"Successfully fetched data for {len(data)}/{len(validated_symbols)} symbols")
            return data
            
        except Exception as e:
            log.error(f"Error in fetch_multiple_symbols: {str(e)}", exc_info=True)
            raise DataFetchError(f"Failed to fetch multiple symbols: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(NetworkError)
    )
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
            
        Raises:
            DataFetchError: If news fetch fails
        """
        try:
            validate_symbol(symbol)
            validate_positive_number(days, "days")
            
            if not self.finnhub_key:
                log.warning("Finnhub API key not configured, skipping news fetch")
                return []
            
            self._check_rate_limit('finnhub', max_calls_per_minute=30)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news = response.json()
            
            if not isinstance(news, list):
                log.warning(f"Unexpected news response format for {symbol}")
                return []
            
            log.info(f"Fetched {len(news)} news articles for {symbol}")
            return news
            
        except requests.exceptions.Timeout as e:
            raise NetworkError(f"Timeout fetching news for {symbol}") from e
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error fetching news for {symbol}: {str(e)}") from e
        except Exception as e:
            log.error(f"Error fetching news for {symbol}: {str(e)}", exc_info=True)
            raise DataFetchError(f"Failed to fetch news for {symbol}: {str(e)}") from e
    
    @lru_cache(maxsize=10)
    def get_sp500_symbols(self) -> List[str]:
        """
        Fetch S&P 500 symbols from Wikipedia.
        
        Returns:
            List of S&P 500 symbols
        """
        try:
            log.info("Fetching S&P 500 symbols")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            log.info(f"Fetched {len(symbols)} S&P 500 symbols")
            return symbols
        except Exception as e:
            log.error(f"Error fetching S&P 500 symbols: {str(e)}", exc_info=True)
            return []
    
    @lru_cache(maxsize=10)
    def get_nasdaq100_symbols(self) -> List[str]:
        """
        Fetch NASDAQ-100 symbols from Wikipedia.
        
        Returns:
            List of NASDAQ-100 symbols
        """
        try:
            log.info("Fetching NASDAQ-100 symbols")
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            df = tables[4]
            symbols = df['Ticker'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            log.info(f"Fetched {len(symbols)} NASDAQ-100 symbols")
            return symbols
        except Exception as e:
            log.error(f"Error fetching NASDAQ-100 symbols: {str(e)}", exc_info=True)
            return []
    
    def get_watchlist_symbols(self) -> List[str]:
        """
        Get symbols from configured watchlists.
        
        Returns:
            List of symbols from watchlists
        """
        symbols = set()
        
        watchlists = config.get('symbols.watchlists', [])
        if 'SP500' in watchlists:
            symbols.update(self.get_sp500_symbols())
        if 'NASDAQ100' in watchlists:
            symbols.update(self.get_nasdaq100_symbols())
        
        custom = config.get('symbols.custom_symbols', [])
        if custom:
            try:
                validated_custom = validate_symbols(custom)
                symbols.update(validated_custom)
            except Exception as e:
                log.error(f"Error validating custom symbols: {str(e)}")
        
        return list(symbols)
    
    def calculate_volatility(self, symbol: str, days: int = 30) -> float:
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days for calculation
            
        Returns:
            Annualized volatility
        """
        try:
            validate_symbol(symbol)
            validate_positive_number(days, "days")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra buffer
            
            df = self.fetch_historical_data(
                symbol, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty or len(df) < 10:
                log.warning(f"Insufficient data for volatility calculation: {symbol}")
                return 0.0
            
            returns = df['Close'].pct_change().dropna()
            
            if len(returns) < 5:
                return 0.0
            
            volatility = returns.std() * np.sqrt(252)
            
            return float(volatility)
            
        except Exception as e:
            log.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.0
    
    def filter_symbols_by_criteria(
        self, 
        symbols: List[str],
        min_volume: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volatility: Optional[float] = None
    ) -> List[str]:
        """
        Filter symbols based on criteria.
        
        Args:
            symbols: List of symbols to filter
            min_volume: Minimum daily volume
            min_price: Minimum price
            max_price: Maximum price
            min_volatility: Minimum volatility
            
        Returns:
            List of filtered symbols
        """
        try:
            validated_symbols = validate_symbols(symbols)
            
            criteria = config.get('symbols.discovery.criteria', {})
            min_volume = min_volume or criteria.get('min_volume', 1000000)
            min_price = min_price or criteria.get('min_price', 5.0)
            max_price = max_price or criteria.get('max_price', 1000.0)
            min_volatility = min_volatility or criteria.get('min_volatility', 0.02)
            
            log.info(f"Filtering {len(validated_symbols)} symbols with criteria: "
                    f"volume>={min_volume}, {min_price}<=price<={max_price}, volatility>={min_volatility}")
            
            filtered = []
            
            for symbol in validated_symbols:
                try:
                    data = self.fetch_realtime_data(symbol)
                    if not data:
                        continue
                    
                    price = data.get('price', 0)
                    volume = data.get('volume', 0)
                    
                    if not (min_price <= price <= max_price and volume >= min_volume):
                        continue
                    
                    volatility = self.calculate_volatility(symbol)
                    
                    if volatility >= min_volatility:
                        filtered.append(symbol)
                        log.debug(f"Symbol {symbol} passed: price={price:.2f}, "
                                 f"volume={volume:,}, volatility={volatility:.4f}")
                    
                except Exception as e:
                    log.warning(f"Error filtering {symbol}: {str(e)}")
                    continue
            
            log.info(f"Filtered {len(filtered)} symbols from {len(validated_symbols)} candidates")
            return filtered
            
        except Exception as e:
            log.error(f"Error in filter_symbols_by_criteria: {str(e)}", exc_info=True)
            return []
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        log.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        }
