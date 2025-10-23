from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List
from datetime import datetime

class TradingSignal(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    action: Literal["BUY", "SELL", "HOLD"] = Field(..., description="Trading action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., max_length=200, description="Brief reasoning (max 200 chars, must reference real indicators)")
    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        if len(v) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        
        valid_indicators = [
            'rsi', 'macd', 'sma', 'ema', 'bollinger', 'volume', 'atr', 
            'sentiment', 'alpha', 'momentum', 'volatility', 'trend',
            'support', 'resistance', 'breakout', 'oversold', 'overbought'
        ]
        
        v_lower = v.lower()
        if not any(indicator in v_lower for indicator in valid_indicators):
            raise ValueError(f"Reasoning must reference at least one valid indicator: {', '.join(valid_indicators)}")
        
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        if 'action' in values and values['action'] == 'HOLD' and v > 0.5:
            raise ValueError("HOLD action should have confidence <= 0.5")
        return v

class AlphaFormula(BaseModel):
    formula_id: str = Field(..., description="Unique identifier for the alpha formula")
    expression: str = Field(..., max_length=500, description="Mathematical expression using valid operators and functions")
    description: str = Field(..., max_length=200, description="Brief description of what the alpha captures")
    expected_rankic: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Expected RankIC score")
    
    @validator('expression')
    def validate_expression(cls, v):
        valid_operators = ['+', '-', '*', '/', '(', ')', 'abs', 'log', 'sqrt', 'rank', 'mean', 'std', 'max', 'min']
        valid_fields = ['close', 'open', 'high', 'low', 'volume', 'vwap', 'returns']
        
        v_lower = v.lower()
        
        if not any(field in v_lower for field in valid_fields):
            raise ValueError(f"Expression must use at least one valid field: {', '.join(valid_fields)}")
        
        invalid_chars = ['$', '@', '#', '&', '!', '?', ';', ':', '"', "'"]
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Expression contains invalid characters: {invalid_chars}")
        
        return v

class MarketAnalysis(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    market_regime: Literal["bull", "bear", "sideways"] = Field(..., description="Current market regime")
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="Trend strength (0=weak, 1=strong)")
    volatility_level: Literal["low", "medium", "high"] = Field(..., description="Volatility classification")
    key_levels: dict = Field(..., description="Support and resistance levels")
    
    @validator('key_levels')
    def validate_key_levels(cls, v):
        required_keys = ['support', 'resistance']
        if not all(key in v for key in required_keys):
            raise ValueError(f"key_levels must contain: {required_keys}")
        
        if not isinstance(v['support'], (int, float)) or not isinstance(v['resistance'], (int, float)):
            raise ValueError("Support and resistance must be numeric values")
        
        if v['support'] >= v['resistance']:
            raise ValueError("Support must be less than resistance")
        
        return v

class RiskAssessment(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    risk_level: Literal["low", "medium", "high", "extreme"] = Field(..., description="Overall risk level")
    position_size_pct: float = Field(..., ge=0.0, le=0.2, description="Recommended position size as % of portfolio (max 20%)")
    stop_loss_pct: float = Field(..., ge=0.01, le=0.1, description="Stop loss percentage (1-10%)")
    take_profit_pct: float = Field(..., ge=0.02, le=0.5, description="Take profit percentage (2-50%)")
    max_holding_days: int = Field(..., ge=1, le=365, description="Maximum holding period in days")
    
    @validator('position_size_pct')
    def validate_position_size(cls, v, values):
        if 'risk_level' in values:
            risk_level = values['risk_level']
            if risk_level == 'extreme' and v > 0.05:
                raise ValueError("Extreme risk positions should not exceed 5%")
            elif risk_level == 'high' and v > 0.10:
                raise ValueError("High risk positions should not exceed 10%")
        return v

class PortfolioRecommendation(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    symbols: List[str] = Field(..., min_items=1, max_items=20, description="List of recommended symbols")
    allocation: dict = Field(..., description="Allocation percentages for each symbol")
    total_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall portfolio risk score")
    expected_sharpe: Optional[float] = Field(None, ge=-2.0, le=5.0, description="Expected Sharpe ratio")
    reasoning: str = Field(..., max_length=500, description="Portfolio construction reasoning")
    
    @validator('allocation')
    def validate_allocation(cls, v, values):
        if 'symbols' in values:
            symbols = values['symbols']
            
            if set(v.keys()) != set(symbols):
                raise ValueError("Allocation must include all symbols and no extras")
            
            total_allocation = sum(v.values())
            if not (0.95 <= total_allocation <= 1.05):
                raise ValueError(f"Total allocation must sum to ~100%, got {total_allocation*100:.1f}%")
            
            for symbol, alloc in v.items():
                if not (0.0 <= alloc <= 0.25):
                    raise ValueError(f"Individual allocation for {symbol} must be 0-25%, got {alloc*100:.1f}%")
        
        return v

def validate_llm_output(output_dict: dict, schema_class: type[BaseModel]) -> BaseModel:
    try:
        validated = schema_class(**output_dict)
        return validated
    except Exception as e:
        raise ValueError(f"LLM output validation failed: {str(e)}")
