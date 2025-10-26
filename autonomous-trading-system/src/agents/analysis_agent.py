import pandas as pd
from typing import Dict, List
from datetime import datetime
from src.models import SentimentAnalyzer
from src.models.llm_trader import LLMTrader
from src.strategies import AlphaMining
from src.utils import log, config

class AnalysisAgent:
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.llm_trader = LLMTrader()
        self.alpha_miner = AlphaMining()
        self.top_alphas = []
        log.info("AnalysisAgent initialized")
    
    def analyze_market_data(
        self, 
        processed_data: Dict[str, pd.DataFrame],
        news_data: Dict[str, List[Dict]]
    ) -> Dict:
        log.info(f"Analyzing market data for {len(processed_data)} symbols")
        
        sentiment_signals = {}
        if news_data:
            sentiment_signals = self.sentiment_analyzer.get_sentiment_signals(news_data)
        
        if not self.top_alphas:
            log.info("Mining alpha factors...")
            self.top_alphas = self.alpha_miner.mine_alphas(processed_data)
        
        alpha_signals = {}
        for symbol, df in processed_data.items():
            if len(df) > 0:
                alpha_signal = self.alpha_miner.get_alpha_signals(df, self.top_alphas)
                alpha_signals[symbol] = {
                    'combined_score': float(alpha_signal.iloc[-1]) if len(alpha_signal) > 0 else 0.0
                }
        
        analysis_results = {
            'sentiment_signals': sentiment_signals,
            'alpha_signals': alpha_signals,
            'top_alphas': self.top_alphas[:10],
            'timestamp': datetime.now()
        }
        
        log.info("Market analysis completed")
        return analysis_results
    
    def generate_trading_signals(
        self,
        processed_data: Dict[str, pd.DataFrame],
        analysis_results: Dict,
        portfolio_state: Dict = None,
        risk_context: Dict = None
    ) -> Dict[str, Dict]:
        log.info(f"Generating trading signals for {len(processed_data)} symbols")
        
        trading_signals = {}
        
        for symbol, df in processed_data.items():
            if len(df) == 0:
                continue
            
            latest_data = df.iloc[-1].to_dict()
            
            sentiment_data = analysis_results['sentiment_signals'].get(symbol, {
                'overall_sentiment': 0.0,
                'signal': 'neutral'
            })
            
            alpha_data = analysis_results['alpha_signals'].get(symbol, {
                'combined_score': 0.0
            })
            
            from src.utils.indicators import TechnicalIndicators
            market_regime = TechnicalIndicators.detect_market_regime(df)
            
            alpha_data_with_top = alpha_data.copy()
            alpha_data_with_top['top_alphas'] = analysis_results.get('top_alphas', [])
            
            llm_decision = self.llm_trader.generate_trading_decision(
                symbol=symbol,
                technical_data=latest_data,
                sentiment_data=sentiment_data,
                alpha_signals=alpha_data_with_top,
                market_regime=market_regime,
                market_data=df,
                portfolio_state=portfolio_state,
                risk_context=risk_context
            )

            if llm_decision.get('reasoning', '') == 'Fallback decision based on technical indicators':
                log.warning(f"LLM fallback used for {symbol} - model may not be loaded")
            else:
                log.info(f"LLM active for {symbol}: {llm_decision['action']} ({llm_decision['confidence']:.2f})")

            
            technical_signal = latest_data.get('Signal', 0)
            sentiment_score = sentiment_data.get('overall_sentiment', 0)
            alpha_score = alpha_data.get('combined_score', 0)
            
            combined_score = (
                0.4 * technical_signal +
                0.3 * sentiment_score +
                0.3 * alpha_score
            )
            
            if llm_decision['action'] == 'BUY':
                combined_score += 0.5 * llm_decision['confidence']
            elif llm_decision['action'] == 'SELL':
                combined_score -= 0.5 * llm_decision['confidence']
            
            if combined_score > 0.1:  # Lowered from 0.3 for more trades
                final_action = 'BUY'
            elif combined_score < -0.1:  # Lowered from -0.3 for more trades
                final_action = 'SELL'
            else:
                final_action = 'HOLD'
            
            trading_signals[symbol] = {
                'action': final_action,
                'llm_action': llm_decision['action'],
                'llm_confidence': llm_decision['confidence'],
                'llm_reasoning': llm_decision.get('reasoning', ''),
                'combined_score': combined_score,
                'technical_signal': technical_signal,
                'sentiment_score': sentiment_score,
                'alpha_score': alpha_score,
                'market_regime': market_regime,
                'current_price': latest_data.get('Close', 0),
                'timestamp': datetime.now()
            }
            
            log.info(f"Signal for {symbol}: {final_action} (score: {combined_score:.3f}, LLM: {llm_decision['action']})")
        
        return trading_signals
    
    def forecast_price_movement(
        self,
        symbol: str,
        df: pd.DataFrame,
        horizon: int = 5
    ) -> Dict:
        if len(df) < 50:
            return {'forecast': 'NEUTRAL', 'confidence': 0.0}
        
        recent_returns = df['Close'].pct_change().tail(20)
        momentum = recent_returns.mean()
        
        volatility = recent_returns.std()
        
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
        
        forecast_score = 0
        
        if momentum > 0.001:
            forecast_score += 1
        elif momentum < -0.001:
            forecast_score -= 1
        
        if rsi < 30:
            forecast_score += 1
        elif rsi > 70:
            forecast_score -= 1
        
        if macd > macd_signal:
            forecast_score += 1
        elif macd < macd_signal:
            forecast_score -= 1
        
        if forecast_score > 1:
            forecast = 'BULLISH'
            confidence = min(0.8, 0.5 + abs(forecast_score) * 0.1)
        elif forecast_score < -1:
            forecast = 'BEARISH'
            confidence = min(0.8, 0.5 + abs(forecast_score) * 0.1)
        else:
            forecast = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'forecast': forecast,
            'confidence': confidence,
            'momentum': momentum,
            'volatility': volatility,
            'horizon_days': horizon
        }
    
    def run(
        self,
        processed_data: Dict[str, pd.DataFrame],
        news_data: Dict[str, List[Dict]]
    ) -> Dict:
        log.info("AnalysisAgent starting analysis")
        
        analysis_results = self.analyze_market_data(processed_data, news_data)
        
        trading_signals = self.generate_trading_signals(processed_data, analysis_results)
        
        forecasts = {}
        for symbol, df in processed_data.items():
            forecasts[symbol] = self.forecast_price_movement(symbol, df)
        
        result = {
            'analysis': analysis_results,
            'signals': trading_signals,
            'forecasts': forecasts,
            'timestamp': datetime.now()
        }
        
        log.info("AnalysisAgent completed successfully")
        return result
