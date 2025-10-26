"""
Prompt Agent for LLM Trading Decisions

Synthesizes inputs from all agents (technical, sentiment, alpha, risk) into
cohesive, context-aware prompts that encourage balanced trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from src.utils import log

class PromptAgent:
    """
    Aggregates signals from all agents and builds context-aware prompts for LLM.
    
    Ensures balanced decision-making by providing:
    - Technical indicator context
    - Sentiment analysis
    - Alpha factor signals
    - Risk management constraints
    - Portfolio state
    - Market regime
    """
    
    def __init__(self):
        self.decision_history = []
        log.info("PromptAgent initialized")
    
    def build_trading_prompt(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_signals: Dict,
        sentiment_score: float,
        alpha_signals: Optional[List[Dict]] = None,
        portfolio_state: Optional[Dict] = None,
        risk_context: Optional[Dict] = None
    ) -> str:
        """
        Build comprehensive trading prompt from all agent inputs.
        
        Args:
            symbol: Stock symbol
            market_data: Historical price data with indicators
            technical_signals: Technical indicator signals
            sentiment_score: Sentiment analysis score (-1 to 1)
            alpha_signals: Alpha factor signals
            portfolio_state: Current portfolio positions and cash
            risk_context: Risk management constraints
            
        Returns:
            Formatted prompt string for LLM
        """
        
        current_price = market_data['Close'].iloc[-1]
        prev_price = market_data['Close'].iloc[-2] if len(market_data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        prompt_parts = []
        
        prompt_parts.append(f"# Trading Decision Request for {symbol}")
        prompt_parts.append(f"Current Date: {market_data.index[-1].strftime('%Y-%m-%d')}")
        prompt_parts.append(f"Current Price: ${current_price:.2f} ({price_change:+.2f}%)")
        prompt_parts.append("")
        
        prompt_parts.append("## Technical Analysis")
        prompt_parts.append(self._format_technical_signals(market_data, technical_signals))
        prompt_parts.append("")
        
        prompt_parts.append("## Sentiment Analysis")
        prompt_parts.append(self._format_sentiment(sentiment_score))
        prompt_parts.append("")
        
        if alpha_signals:
            prompt_parts.append("## Alpha Factors")
            prompt_parts.append(self._format_alpha_signals(alpha_signals))
            prompt_parts.append("")
        
        if portfolio_state:
            prompt_parts.append("## Portfolio Context")
            prompt_parts.append(self._format_portfolio_state(symbol, portfolio_state))
            prompt_parts.append("")
        
        if risk_context:
            prompt_parts.append("## Risk Management")
            prompt_parts.append(self._format_risk_context(risk_context))
            prompt_parts.append("")
        
        prompt_parts.append("## Decision Framework")
        prompt_parts.append(self._get_decision_framework(symbol, portfolio_state))
        prompt_parts.append("")
        
        prompt_parts.append("## Required Output")
        prompt_parts.append(self._get_output_format())
        
        return "\n".join(prompt_parts)
    
    def _format_technical_signals(self, market_data: pd.DataFrame, signals: Dict) -> str:
        """Format technical indicator signals"""
        lines = []
        
        latest = market_data.iloc[-1]
        
        lines.append(f"- RSI: {latest.get('RSI', 50):.1f} " + 
                    ("(Oversold)" if latest.get('RSI', 50) < 30 else 
                     "(Overbought)" if latest.get('RSI', 50) > 70 else "(Neutral)"))
        
        lines.append(f"- MACD: {latest.get('MACD', 0):.2f}, Signal: {latest.get('MACD_Signal', 0):.2f} " +
                    ("(Bullish crossover)" if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else "(Bearish)"))
        
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        lines.append(f"- Price vs SMA20: {((latest['Close'] - sma_20) / sma_20 * 100):+.1f}%")
        lines.append(f"- SMA20 vs SMA50: {((sma_20 - sma_50) / sma_50 * 100):+.1f}% " +
                    ("(Golden cross)" if sma_20 > sma_50 else "(Death cross)"))
        
        bb_pos = latest.get('BB_Position', 0.5)
        lines.append(f"- Bollinger Band Position: {bb_pos:.2f} " +
                    ("(Near upper band)" if bb_pos > 0.8 else 
                     "(Near lower band)" if bb_pos < 0.2 else "(Middle)"))
        
        atr = latest.get('ATR', 0)
        lines.append(f"- Volatility (ATR): ${atr:.2f}")
        
        volume_ratio = latest['Volume'] / market_data['Volume'].rolling(20).mean().iloc[-1]
        lines.append(f"- Volume: {volume_ratio:.1f}x average " +
                    ("(High)" if volume_ratio > 1.5 else "(Normal)"))
        
        return "\n".join(lines)
    
    def _format_sentiment(self, sentiment_score: float) -> str:
        """Format sentiment analysis"""
        if sentiment_score > 0.3:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.3:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        return f"- Market Sentiment: {sentiment_label} (score: {sentiment_score:.2f})"
    
    def _format_alpha_signals(self, alpha_signals: List[Dict]) -> str:
        """Format alpha factor signals"""
        if not alpha_signals:
            return "- No alpha signals available"
        
        lines = []
        for alpha in alpha_signals[:3]:
            lines.append(f"- {alpha.get('name', 'Unknown')}: {alpha.get('signal', 0):.2f} " +
                        f"(IC: {alpha.get('rank_ic', 0):.3f})")
        
        return "\n".join(lines)
    
    def _format_portfolio_state(self, symbol: str, portfolio_state: Dict) -> str:
        """Format current portfolio state"""
        lines = []
        
        has_position = symbol in portfolio_state.get('positions', {})
        
        if has_position:
            position = portfolio_state['positions'][symbol]
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', 0)
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            
            lines.append(f"- Current Position: {position.get('shares', 0)} shares @ ${entry_price:.2f}")
            lines.append(f"- Position P&L: {pnl_pct:+.2f}%")
            lines.append(f"- Holding Period: {position.get('days_held', 0)} days")
        else:
            lines.append(f"- Current Position: None (available to buy)")
        
        lines.append(f"- Available Cash: ${portfolio_state.get('cash', 0):,.2f}")
        lines.append(f"- Portfolio Value: ${portfolio_state.get('total_value', 0):,.2f}")
        lines.append(f"- Total Positions: {len(portfolio_state.get('positions', {}))}")
        
        return "\n".join(lines)
    
    def _format_risk_context(self, risk_context: Dict) -> str:
        """Format risk management context"""
        lines = []
        
        max_drawdown = risk_context.get('max_drawdown', 0)
        current_drawdown = risk_context.get('current_drawdown', 0)
        
        lines.append(f"- Current Drawdown: {current_drawdown:.1f}%")
        lines.append(f"- Max Drawdown: {max_drawdown:.1f}%")
        
        if abs(current_drawdown) > 10:
            lines.append("- ⚠️ WARNING: Significant drawdown - consider defensive positions")
        
        risk_level = risk_context.get('risk_level', 'MEDIUM')
        lines.append(f"- Risk Level: {risk_level}")
        
        if risk_level == 'HIGH':
            lines.append("- ⚠️ High risk detected - prioritize capital preservation")
        
        return "\n".join(lines)
    
    def _get_decision_framework(self, symbol: str, portfolio_state: Optional[Dict]) -> str:
        """Provide decision-making framework"""
        has_position = False
        if portfolio_state:
            has_position = symbol in portfolio_state.get('positions', {})
        
        if has_position:
            return """You currently HOLD this position. Consider:
- **HOLD**: If fundamentals remain strong and no exit signals
- **SELL**: If technical indicators turn bearish, sentiment deteriorates, or profit target reached
- **DO NOT BUY MORE**: You already have a position

Exit signals to watch for:
- RSI > 70 (overbought)
- MACD bearish crossover
- Price breaks below SMA20
- Negative sentiment shift
- Profit target reached (>10% gain)
- Stop loss triggered (>5% loss)"""
        else:
            return """You currently have NO position. Consider:
- **BUY**: Only if multiple bullish signals align (technical + sentiment + alpha)
- **HOLD**: If signals are mixed or unclear - wait for better entry
- **DO NOT SELL**: You have no position to sell

Entry signals to watch for:
- RSI < 40 (not oversold but approaching value)
- MACD bullish crossover
- Price above SMA20 with upward momentum
- Positive sentiment
- Strong alpha signals
- Low volatility environment"""
    
    def _get_output_format(self) -> str:
        """Specify required output format"""
        return """Provide your decision in JSON format:
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of decision (max 200 chars)"
}

IMPORTANT:
- Be conservative - prefer HOLD over risky trades
- Only BUY when multiple signals align
- Only SELL when clear exit signals present
- Confidence should reflect signal strength (0.5-0.7 typical, 0.8+ rare)"""
    
    def track_decision(self, symbol: str, decision: Dict):
        """Track decision for analysis"""
        self.decision_history.append({
            'symbol': symbol,
            'action': decision.get('action'),
            'confidence': decision.get('confidence'),
            'timestamp': datetime.now(),
            'reasoning': decision.get('reasoning')
        })
        
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_decision_stats(self) -> Dict:
        """Get statistics on decision history"""
        if not self.decision_history:
            return {}
        
        actions = [d['action'] for d in self.decision_history]
        
        return {
            'total_decisions': len(self.decision_history),
            'buy_count': actions.count('BUY'),
            'sell_count': actions.count('SELL'),
            'hold_count': actions.count('HOLD'),
            'buy_pct': actions.count('BUY') / len(actions) * 100,
            'sell_pct': actions.count('SELL') / len(actions) * 100,
            'hold_pct': actions.count('HOLD') / len(actions) * 100,
            'avg_confidence': np.mean([d['confidence'] for d in self.decision_history])
        }
