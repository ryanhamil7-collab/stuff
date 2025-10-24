"""Input validation utilities for the autonomous trading system."""

import re
from typing import List, Optional, Any
from datetime import datetime
import pandas as pd

from .exceptions import ValidationError


def validate_symbol(symbol: str) -> bool:
    """
    Validate a stock symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a non-empty string, got: {type(symbol)}")
    
    symbol = symbol.strip().upper()
    
    if not re.match(r'^[A-Z]{1,5}(-[A-Z]{1,2})?$', symbol):
        raise ValidationError(f"Invalid symbol format: {symbol}")
    
    return True


def validate_symbols(symbols: List[str]) -> List[str]:
    """
    Validate a list of stock symbols.
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        List of validated symbols
        
    Raises:
        ValidationError: If any symbol is invalid
    """
    if not symbols or not isinstance(symbols, list):
        raise ValidationError("Symbols must be a non-empty list")
    
    validated = []
    for symbol in symbols:
        try:
            validate_symbol(symbol)
            validated.append(symbol.strip().upper())
        except ValidationError as e:
            raise ValidationError(f"Invalid symbol in list: {e}")
    
    return validated


def validate_date(date_str: str) -> bool:
    """
    Validate a date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If date is invalid
    """
    if not date_str or not isinstance(date_str, str):
        raise ValidationError(f"Date must be a non-empty string, got: {type(date_str)}")
    
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError as e:
        raise ValidationError(f"Invalid date format (expected YYYY-MM-DD): {date_str}") from e


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate a date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If date range is invalid
    """
    validate_date(start_date)
    validate_date(end_date)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start >= end:
        raise ValidationError(f"Start date must be before end date: {start_date} >= {end_date}")
    
    if end > datetime.now():
        raise ValidationError(f"End date cannot be in the future: {end_date}")
    
    return True


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected pandas DataFrame, got: {type(df)}")
    
    if df.empty:
        raise ValidationError("DataFrame is empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"DataFrame missing required columns: {missing}")
    
    return True


def validate_positive_number(value: Any, name: str = "value") -> bool:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If value is invalid
    """
    try:
        num = float(value)
        if num <= 0:
            raise ValidationError(f"{name} must be positive, got: {num}")
        return True
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got: {type(value)}") from e


def validate_percentage(value: Any, name: str = "value") -> bool:
    """
    Validate that a value is a valid percentage (0-100).
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If value is invalid
    """
    try:
        num = float(value)
        if not 0 <= num <= 100:
            raise ValidationError(f"{name} must be between 0 and 100, got: {num}")
        return True
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got: {type(value)}") from e


def validate_config(config: dict, required_keys: List[str]) -> bool:
    """
    Validate that a configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Config must be a dictionary, got: {type(config)}")
    
    missing = set(required_keys) - set(config.keys())
    if missing:
        raise ValidationError(f"Config missing required keys: {missing}")
    
    return True


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        service_name: Name of the service for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise ValidationError(f"{service_name} API key is missing or invalid")
    
    if len(api_key.strip()) < 10:
        raise ValidationError(f"{service_name} API key appears to be too short")
    
    return True
