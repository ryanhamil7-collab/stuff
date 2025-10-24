from pydantic import BaseModel, Field, validator
from typing import List, Literal
from datetime import datetime

class EvidenceItem(BaseModel):
    """Evidence item for trading decision."""
    type: Literal["technical", "sentiment", "alpha", "fundamental"] = Field(..., description="Evidence type")
    indicator: str = Field(..., description="Indicator name (e.g., RSI, FinBERT)")
    value: float = Field(..., description="Indicator value")
    signal: str = Field(..., max_length=50, description="Signal interpretation")
    source: str = Field(default="", description="Data source")
    headline: str = Field(default="", max_length=200, description="Related headline if applicable")

class TradingR1Decision(BaseModel):
    """
    Trading-R1 style structured decision output.
    
    Based on 2025 research showing improved risk-adjusted returns
    with normalized multi-horizon predictions.
    """
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    horizon_days: List[int] = Field(default=[1, 5, 20], description="Prediction horizons")
    predicted_return: List[float] = Field(..., min_items=3, max_items=3, description="Predicted returns for each horizon")
    confidence: List[float] = Field(..., min_items=3, max_items=3, description="Confidence for each horizon (0-1)")
    
    decision: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = Field(..., description="5-level decision")
    
    evidence: List[EvidenceItem] = Field(..., min_items=2, max_items=10, description="Supporting evidence")
    
    risk_flags: List[str] = Field(default=[], description="Risk warnings")
    
    sharpe_estimate: float = Field(default=0.0, description="Estimated Sharpe ratio")
    max_drawdown_estimate: float = Field(default=0.0, ge=-1.0, le=0.0, description="Estimated max drawdown")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("All confidence values must be between 0 and 1")
        return v
    
    @validator('predicted_return')
    def validate_returns(cls, v, values):
        if 'decision' in values:
            decision = values['decision']
            avg_return = sum(v) / len(v)
            
            if decision in ['strong_buy', 'buy'] and avg_return <= 0:
                raise ValueError(f"Buy decision requires positive predicted returns, got {avg_return}")
            elif decision in ['strong_sell', 'sell'] and avg_return >= 0:
                raise ValueError(f"Sell decision requires negative predicted returns, got {avg_return}")
        
        return v
    
    @validator('evidence')
    def validate_evidence(cls, v):
        evidence_types = [e.type for e in v]
        if 'technical' not in evidence_types:
            raise ValueError("Must include at least one technical evidence item")
        return v

def create_trading_r1_prompt(symbol: str, technical_data: dict, sentiment_data: dict, alpha_data: dict) -> str:
    """
    Create Trading-R1 style prompt for structured output.
    
    Args:
        symbol: Stock symbol
        technical_data: Technical indicators
        sentiment_data: Sentiment analysis
        alpha_data: Alpha signals
    
    Returns:
        Structured prompt
    """
    prompt = f"""You are a quantitative trading analyst using the Trading-R1 framework. 
Provide a STRUCTURED DECISION for {symbol} in strict JSON format.

MARKET DATA:
- Current Price: ${technical_data.get('Close', 0):.2f}
- RSI(14): {technical_data.get('RSI', 50):.2f}
- MACD: {technical_data.get('MACD', 0):.4f}
- MACD Signal: {technical_data.get('MACD_Signal', 0):.4f}
- SMA(20): ${technical_data.get('SMA_20', 0):.2f}
- SMA(50): ${technical_data.get('SMA_50', 0):.2f}
- Bollinger Position: {technical_data.get('BB_Position', 0.5):.2f}
- ATR: ${technical_data.get('ATR', 0):.2f}
- ADX: {technical_data.get('ADX', 0):.2f}
- Volume Ratio: {technical_data.get('Volume_Ratio', 1.0):.2f}x

SENTIMENT:
- Score: {sentiment_data.get('overall_sentiment', 0):.3f}
- Articles: {sentiment_data.get('articles_analyzed', 0)}

ALPHA:
- Combined Score: {alpha_data.get('combined_score', 0):.3f}

OUTPUT REQUIRED JSON:
{{
  "symbol": "{symbol}",
  "horizon_days": [1, 5, 20],
  "predicted_return": [<1d_return>, <5d_return>, <20d_return>],
  "confidence": [<1d_conf>, <5d_conf>, <20d_conf>],
  "decision": "strong_buy | buy | hold | sell | strong_sell",
  "evidence": [
    {{"type": "technical", "indicator": "RSI", "value": {technical_data.get('RSI', 50):.2f}, "signal": "<interpretation>"}},
    {{"type": "sentiment", "indicator": "FinBERT", "value": {sentiment_data.get('overall_sentiment', 0):.3f}, "signal": "<interpretation>"}},
    {{"type": "alpha", "indicator": "combined_alpha", "value": {alpha_data.get('combined_score', 0):.3f}, "signal": "<interpretation>"}}
  ],
  "risk_flags": ["<flag1>", "<flag2>"],
  "sharpe_estimate": <estimated_sharpe>,
  "max_drawdown_estimate": <estimated_dd>
}}

RULES:
1. predicted_return: Normalized returns as decimals (e.g., 0.05 = 5%)
2. confidence: 0.0 to 1.0, decreasing with longer horizons
3. decision: Must align with predicted_return signs
4. evidence: Minimum 2 items, cite ONLY real indicators from data above
5. risk_flags: Include if RSI>70, volatility>3%, earnings nearby, etc.
6. sharpe_estimate: Expected Sharpe ratio (0.5-2.0 typical)
7. max_drawdown_estimate: Expected max drawdown as negative decimal

Output ONLY the JSON, no additional text:"""
    
    return prompt

def parse_trading_r1_response(response: str) -> dict:
    """
    Parse Trading-R1 response into structured format.
    
    Args:
        response: LLM response text
    
    Returns:
        Parsed dictionary
    """
    import json
    import re
    
    try:
        if '{' in response and '}' in response:
            json_start = response.index('{')
            json_end = response.rindex('}') + 1
            json_str = response[json_start:json_end]
            
            parsed = json.loads(json_str)
            
            validated = TradingR1Decision(**parsed)
            
            return validated.dict()
    
    except Exception as e:
        return {
            'symbol': 'UNKNOWN',
            'horizon_days': [1, 5, 20],
            'predicted_return': [0.0, 0.0, 0.0],
            'confidence': [0.5, 0.4, 0.3],
            'decision': 'hold',
            'evidence': [
                {'type': 'technical', 'indicator': 'fallback', 'value': 0.0, 'signal': 'parsing_failed'}
            ],
            'risk_flags': ['llm_parsing_error'],
            'sharpe_estimate': 0.0,
            'max_drawdown_estimate': 0.0
        }
