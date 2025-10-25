"""Custom exceptions for the autonomous trading system."""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    pass


class DataFetchError(TradingSystemError):
    """Raised when data fetching fails."""
    pass


class DataProcessingError(TradingSystemError):
    """Raised when data processing fails."""
    pass


class ModelError(TradingSystemError):
    """Raised when model operations fail."""
    pass


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(TradingSystemError):
    """Raised when validation fails."""
    pass


class APIError(TradingSystemError):
    """Raised when API calls fail."""
    pass


class InsufficientDataError(DataProcessingError):
    """Raised when there is insufficient data for processing."""
    pass


class InvalidSymbolError(DataFetchError):
    """Raised when a symbol is invalid or not found."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class NetworkError(TradingSystemError):
    """Raised when network operations fail."""
    pass
