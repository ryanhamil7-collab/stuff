"""Tests for data fetcher module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data_pipeline.data_fetcher_improved import ImprovedDataFetcher
from src.utils.exceptions import (
    DataFetchError, InvalidSymbolError, RateLimitError,
    InsufficientDataError
)


class TestImprovedDataFetcher:
    """Test suite for ImprovedDataFetcher."""
    
    @pytest.fixture
    def fetcher(self):
        """Create a data fetcher instance."""
        return ImprovedDataFetcher()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Open': 100 + pd.Series(range(len(dates))),
            'High': 105 + pd.Series(range(len(dates))),
            'Low': 95 + pd.Series(range(len(dates))),
            'Close': 100 + pd.Series(range(len(dates))),
            'Volume': 1000000 + pd.Series(range(len(dates))) * 1000
        })
        return df
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher is not None
        assert isinstance(fetcher.cache, dict)
        assert isinstance(fetcher.rate_limit_tracker, dict)
    
    def test_validate_symbol_valid(self, fetcher):
        """Test symbol validation with valid symbols."""
        from src.utils.validators import validate_symbol
        
        valid_symbols = ['AAPL', 'MSFT', 'GOOGL', 'BRK-B', 'SPY']
        for symbol in valid_symbols:
            assert validate_symbol(symbol) is True
    
    def test_validate_symbol_invalid(self, fetcher):
        """Test symbol validation with invalid symbols."""
        from src.utils.validators import validate_symbol
        from src.utils.exceptions import ValidationError
        
        invalid_symbols = ['', '123', 'TOOLONG', 'ABC-DEF-GHI', None]
        for symbol in invalid_symbols:
            with pytest.raises(ValidationError):
                validate_symbol(symbol)
    
    def test_cache_key_generation(self, fetcher):
        """Test cache key generation."""
        key = fetcher._get_cache_key('AAPL', '2023-01-01', '2023-12-31', '1d')
        assert key == 'AAPL_2023-01-01_2023-12-31_1d'
    
    @patch('yfinance.Ticker')
    def test_fetch_historical_data_success(self, mock_ticker, fetcher, sample_dataframe):
        """Test successful historical data fetch."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_dataframe
        mock_ticker.return_value = mock_ticker_instance
        
        result = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2023-12-31')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Symbol' in result.columns
        assert result['Symbol'].iloc[0] == 'AAPL'
    
    @patch('yfinance.Ticker')
    def test_fetch_historical_data_empty(self, mock_ticker, fetcher):
        """Test handling of empty data response."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        with pytest.raises(InsufficientDataError):
            fetcher.fetch_historical_data('INVALID', '2023-01-01', '2023-12-31')
    
    def test_fetch_historical_data_invalid_symbol(self, fetcher):
        """Test handling of invalid symbol."""
        with pytest.raises(InvalidSymbolError):
            fetcher.fetch_historical_data('', '2023-01-01', '2023-12-31')
    
    def test_fetch_historical_data_invalid_dates(self, fetcher):
        """Test handling of invalid date range."""
        from src.utils.exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            fetcher.fetch_historical_data('AAPL', '2023-12-31', '2023-01-01')
    
    @patch('yfinance.Ticker')
    def test_fetch_historical_data_caching(self, mock_ticker, fetcher, sample_dataframe):
        """Test data caching functionality."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_dataframe
        mock_ticker.return_value = mock_ticker_instance
        
        result1 = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2023-12-31')
        
        result2 = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2023-12-31')
        
        assert mock_ticker_instance.history.call_count == 1
        assert len(result1) == len(result2)
    
    @patch('yfinance.Ticker')
    def test_fetch_realtime_data_success(self, mock_ticker, fetcher):
        """Test successful realtime data fetch."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
            'currentPrice': 150.0,
            'volume': 50000000,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'dayHigh': 152.0,
            'dayLow': 148.0,
            'fiftyTwoWeekHigh': 180.0,
            'fiftyTwoWeekLow': 120.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        result = fetcher.fetch_realtime_data('AAPL')
        
        assert isinstance(result, dict)
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 150.0
        assert result['volume'] == 50000000
    
    @patch('yfinance.Ticker')
    def test_fetch_realtime_data_invalid_symbol(self, mock_ticker, fetcher):
        """Test handling of invalid symbol in realtime data."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance
        
        with pytest.raises(InvalidSymbolError):
            fetcher.fetch_realtime_data('INVALID')
    
    def test_rate_limiting(self, fetcher):
        """Test rate limiting functionality."""
        for i in range(65):
            fetcher._check_rate_limit('test_service', max_calls_per_minute=60)
            if i >= 60:
                with pytest.raises(RateLimitError):
                    fetcher._check_rate_limit('test_service', max_calls_per_minute=60)
                break
    
    @patch('yfinance.Ticker')
    def test_fetch_multiple_symbols(self, mock_ticker, fetcher, sample_dataframe):
        """Test fetching multiple symbols."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_dataframe
        mock_ticker.return_value = mock_ticker_instance
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = fetcher.fetch_multiple_symbols(symbols, '2023-01-01', '2023-12-31')
        
        assert isinstance(result, dict)
        assert len(result) <= len(symbols)
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get, fetcher):
        """Test successful news fetch."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {'headline': 'Test news 1', 'summary': 'Summary 1'},
            {'headline': 'Test news 2', 'summary': 'Summary 2'}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        fetcher.finnhub_key = 'test_key'
        
        result = fetcher.fetch_news('AAPL', days=7)
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @patch('requests.get')
    def test_fetch_news_no_api_key(self, mock_get, fetcher):
        """Test news fetch without API key."""
        fetcher.finnhub_key = ''
        
        result = fetcher.fetch_news('AAPL', days=7)
        
        assert isinstance(result, list)
        assert len(result) == 0
        mock_get.assert_not_called()
    
    @patch('pandas.read_html')
    def test_get_sp500_symbols(self, mock_read_html, fetcher):
        """Test fetching S&P 500 symbols."""
        mock_df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        })
        mock_read_html.return_value = [mock_df]
        
        result = fetcher.get_sp500_symbols()
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert 'AAPL' in result
    
    @patch('pandas.read_html')
    def test_get_nasdaq100_symbols(self, mock_read_html, fetcher):
        """Test fetching NASDAQ-100 symbols."""
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        })
        mock_read_html.return_value = [None, None, None, None, mock_df]
        
        result = fetcher.get_nasdaq100_symbols()
        
        assert isinstance(result, list)
        assert len(result) == 5
    
    @patch('yfinance.Ticker')
    def test_calculate_volatility(self, mock_ticker, fetcher, sample_dataframe):
        """Test volatility calculation."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_dataframe
        mock_ticker.return_value = mock_ticker_instance
        
        volatility = fetcher.calculate_volatility('AAPL', days=30)
        
        assert isinstance(volatility, float)
        assert volatility >= 0
    
    def test_clear_cache(self, fetcher):
        """Test cache clearing."""
        fetcher.cache['test_key'] = 'test_value'
        assert len(fetcher.cache) > 0
        
        fetcher.clear_cache()
        assert len(fetcher.cache) == 0
    
    def test_get_cache_stats(self, fetcher):
        """Test cache statistics."""
        fetcher.cache['key1'] = 'value1'
        fetcher.cache['key2'] = 'value2'
        
        stats = fetcher.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert stats['size'] == 2
        assert 'key1' in stats['keys']
        assert 'key2' in stats['keys']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
