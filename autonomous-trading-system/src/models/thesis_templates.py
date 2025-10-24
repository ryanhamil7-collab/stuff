from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal, Optional
from datetime import datetime

class EvidenceItem(BaseModel):
    indicator: str = Field(..., description="Specific indicator name (e.g., RSI, MACD, sentiment)")
    value: float = Field(..., description="Current value of the indicator")
    interpretation: str = Field(..., max_length=100, description="Brief interpretation")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in decision (0-1)")
    
    @validator('indicator')
    def validate_indicator(cls, v):
        valid_indicators = [
            'rsi', 'macd', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'bollinger_upper', 'bollinger_lower', 'atr', 'adx', 'obv', 'vwap',
            'sentiment_score', 'alpha_score', 'volume', 'volatility', 'momentum'
        ]
        if v.lower() not in valid_indicators:
            raise ValueError(f"Indicator must be one of: {', '.join(valid_indicators)}")
        return v

class MarketDataSection(BaseModel):
    current_price: float = Field(..., gt=0, description="Current stock price")
    price_change_1d: float = Field(..., description="1-day price change (%)")
    price_change_5d: float = Field(..., description="5-day price change (%)")
    price_change_20d: float = Field(..., description="20-day price change (%)")
    volume_ratio: float = Field(..., gt=0, description="Volume vs 20-day average")
    volatility_20d: float = Field(..., ge=0, description="20-day volatility")
    
    evidence: List[EvidenceItem] = Field(..., min_items=2, max_items=5, description="Technical evidence")

class FundamentalsSection(BaseModel):
    market_regime: Literal["bull", "bear", "sideways"] = Field(..., description="Current market regime")
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="Trend strength (0=weak, 1=strong)")
    support_level: Optional[float] = Field(None, description="Key support level")
    resistance_level: Optional[float] = Field(None, description="Key resistance level")
    
    evidence: List[EvidenceItem] = Field(..., min_items=1, max_items=3, description="Fundamental evidence")

class SentimentSection(BaseModel):
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    news_count: int = Field(..., ge=0, description="Number of news articles analyzed")
    sentiment_trend: Literal["improving", "stable", "deteriorating"] = Field(..., description="Sentiment trend")
    
    evidence: List[EvidenceItem] = Field(..., min_items=0, max_items=3, description="Sentiment evidence")

class RiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high", "extreme"] = Field(..., description="Overall risk level")
    max_position_size: float = Field(..., ge=0.0, le=0.2, description="Max position size (% of portfolio)")
    stop_loss: float = Field(..., ge=0.01, le=0.15, description="Recommended stop loss (%)")
    take_profit: float = Field(..., ge=0.02, le=0.5, description="Recommended take profit (%)")
    holding_horizon: Literal["short", "medium", "long"] = Field(..., description="Recommended holding period")
    
    risk_factors: List[str] = Field(..., min_items=1, max_items=5, description="Key risk factors")

class TradingDecision(BaseModel):
    action: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"] = Field(..., description="5-level decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    expected_return_1d: float = Field(..., description="Expected 1-day return (%)")
    expected_return_5d: float = Field(..., description="Expected 5-day return (%)")
    expected_return_20d: float = Field(..., description="Expected 20-day return (%)")
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        if 'action' in values:
            action = values['action']
            if action == 'HOLD' and v > 0.6:
                raise ValueError("HOLD action should have confidence <= 0.6")
            if action in ['STRONG_BUY', 'STRONG_SELL'] and v < 0.7:
                raise ValueError("STRONG actions require confidence >= 0.7")
        return v

class StructuredThesis(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    market_data: MarketDataSection = Field(..., description="Market data analysis")
    fundamentals: FundamentalsSection = Field(..., description="Fundamental analysis")
    sentiment: SentimentSection = Field(..., description="Sentiment analysis")
    risk: RiskAssessment = Field(..., description="Risk assessment")
    decision: TradingDecision = Field(..., description="Final trading decision")
    
    synthesis: str = Field(..., min_length=50, max_length=500, description="Synthesis of all evidence")
    
    @validator('synthesis')
    def validate_synthesis(cls, v, values):
        if 'decision' in values:
            action = values['decision'].action
            if action.lower() not in v.lower():
                raise ValueError(f"Synthesis must mention the decision action: {action}")
        
        required_terms = ['because', 'evidence', 'indicates']
        if not any(term in v.lower() for term in required_terms):
            raise ValueError("Synthesis must use causal language (because/evidence/indicates)")
        
        return v

class ThesisPromptTemplate:
    
    @staticmethod
    def create_structured_prompt(
        symbol: str,
        technical_data: Dict,
        sentiment_data: Dict,
        alpha_data: Dict,
        market_regime: str
    ) -> str:
        prompt = f"""You are a disciplined quantitative analyst. Provide a STRUCTURED THESIS for {symbol}.

CRITICAL RULES:
1. Every claim MUST cite specific indicators with exact values
2. Use ONLY real indicators from the data provided
3. Follow the exact format below
4. Provide normalized returns for 1/5/20 day horizons
5. Map to 5-level decision: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL

=== MARKET DATA SECTION ===
Current Price: ${technical_data.get('Close', 0):.2f}
1-Day Change: {technical_data.get('Returns_1d', 0)*100:.2f}%
5-Day Change: {technical_data.get('Returns_5d', 0)*100:.2f}%
20-Day Change: {technical_data.get('Returns_20d', 0)*100:.2f}%
Volume Ratio: {technical_data.get('Volume_Ratio', 1.0):.2f}x
Volatility (20d): {technical_data.get('Volatility_20d', 0)*100:.2f}%

Technical Indicators:
- RSI(14): {technical_data.get('RSI', 50):.2f}
- MACD: {technical_data.get('MACD', 0):.4f}
- MACD Signal: {technical_data.get('MACD_Signal', 0):.4f}
- SMA(20): ${technical_data.get('SMA_20', 0):.2f}
- SMA(50): ${technical_data.get('SMA_50', 0):.2f}
- Bollinger Position: {technical_data.get('BB_Position', 0):.2f}
- ATR: ${technical_data.get('ATR', 0):.2f}
- ADX: {technical_data.get('ADX', 0):.2f}

Evidence (cite 2-5 indicators with interpretations):
1. [Indicator]: [Value] → [Interpretation] (Weight: 0.X)
2. [Indicator]: [Value] → [Interpretation] (Weight: 0.X)
...

=== FUNDAMENTALS SECTION ===
Market Regime: {market_regime}
Trend Strength: [0.0-1.0]
Support Level: $[price]
Resistance Level: $[price]

Evidence (cite 1-3 factors):
1. [Factor]: [Analysis] (Weight: 0.X)
...

=== SENTIMENT SECTION ===
Overall Sentiment: {sentiment_data.get('overall_sentiment', 0):.3f}
News Count: {sentiment_data.get('articles_analyzed', 0)}
Sentiment Trend: [improving/stable/deteriorating]

Evidence (cite 0-3 sentiment signals):
1. [Signal]: [Analysis] (Weight: 0.X)
...

=== RISK ASSESSMENT ===
Risk Level: [low/medium/high/extreme]
Max Position Size: [0.0-0.2] (% of portfolio)
Stop Loss: [0.01-0.15] (%)
Take Profit: [0.02-0.5] (%)
Holding Horizon: [short/medium/long]

Risk Factors:
1. [Factor 1]
2. [Factor 2]
...

=== TRADING DECISION ===
Action: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
Confidence: [0.0-1.0]
Expected Returns:
- 1-Day: [X.XX]%
- 5-Day: [X.XX]%
- 20-Day: [X.XX]%

=== SYNTHESIS ===
[50-500 chars: Synthesize all evidence into a coherent thesis. Must use causal language 
(because/evidence/indicates) and mention the decision action. Example: "STRONG_BUY because 
RSI at 28 indicates oversold conditions, MACD crossed above signal, and positive sentiment 
from 15 articles. Evidence suggests 5-day upside of 8% with medium risk."]

Output in JSON format matching the StructuredThesis schema."""
        
        return prompt
    
    @staticmethod
    def parse_thesis_response(response: str) -> Dict:
        import json
        import re
        
        try:
            if '{' in response and '}' in response:
                json_start = response.index('{')
                json_end = response.rindex('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            thesis_dict = {}
            
            action_match = re.search(r'Action:\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)', response, re.IGNORECASE)
            if action_match:
                thesis_dict['action'] = action_match.group(1).upper()
            
            confidence_match = re.search(r'Confidence:\s*(0?\.\d+|1\.0)', response)
            if confidence_match:
                thesis_dict['confidence'] = float(confidence_match.group(1))
            
            return_1d_match = re.search(r'1-Day:\s*([-+]?\d+\.?\d*)%', response)
            if return_1d_match:
                thesis_dict['expected_return_1d'] = float(return_1d_match.group(1))
            
            return thesis_dict
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'expected_return_1d': 0.0,
                'expected_return_5d': 0.0,
                'expected_return_20d': 0.0
            }
