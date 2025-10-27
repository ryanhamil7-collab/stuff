"""
Position Monitor Agent

Continuously monitors and analyzes held positions like professional traders:
- Real-time news monitoring for held stocks
- Earnings calendar tracking
- Technical indicator updates
- Dynamic stop-loss/take-profit adjustments
- Position sizing rebalancing
- Risk exposure monitoring
- Exit signal generation
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils import log, config
from src.data_pipeline.data_fetcher import DataFetcher
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.utils.indicators import TechnicalIndicators

class PositionMonitor:
    """
    Monitors held positions and generates actionable insights
    """
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_indicators = TechnicalIndicators()
        self.position_history = {}
        log.info("PositionMonitor initialized")
    
    def monitor_positions(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        Monitor all held positions and generate insights
        
        Args:
            positions: Dict of symbol -> position data
            current_prices: Dict of symbol -> current price
            
        Returns:
            Dict of symbol -> monitoring insights
        """
        if not positions:
            log.info("No positions to monitor")
            return {}
        
        log.info(f"Monitoring {len(positions)} held positions")
        
        insights = {}
        for symbol, position in positions.items():
            try:
                insight = self._analyze_position(symbol, position, current_prices.get(symbol))
                insights[symbol] = insight
                
                if insight.get('action_required'):
                    log.warning(f"⚠️ {symbol}: {insight['action_required']}")
                
            except Exception as e:
                log.error(f"Error monitoring {symbol}: {str(e)}")
                insights[symbol] = {'error': str(e)}
        
        return insights
    
    def _analyze_position(
        self,
        symbol: str,
        position: Dict,
        current_price: Optional[float]
    ) -> Dict:
        """
        Comprehensive analysis of a single position
        """
        insight = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'action_required': None,
            'signals': []
        }
        
        if not current_price:
            insight['error'] = 'No current price available'
            return insight
        
        entry_price = position.get('entry_price', current_price)
        shares = position.get('shares', 0)
        
        pnl = (current_price - entry_price) * shares
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        insight['current_price'] = current_price
        insight['entry_price'] = entry_price
        insight['pnl'] = pnl
        insight['pnl_pct'] = pnl_pct
        
        news_sentiment = self._check_news_sentiment(symbol)
        insight['news_sentiment'] = news_sentiment
        
        if news_sentiment.get('alert_level') == 'high':
            insight['signals'].append(f"High-impact news: {news_sentiment.get('summary', '')}")
        
        earnings_info = self._check_earnings_calendar(symbol)
        insight['earnings'] = earnings_info
        
        if earnings_info.get('days_until_earnings', 999) <= 3:
            insight['signals'].append(f"Earnings in {earnings_info['days_until_earnings']} days")
        
        technical_signals = self._check_technical_signals(symbol, current_price)
        insight['technical'] = technical_signals
        
        if technical_signals.get('trend') == 'bearish':
            insight['signals'].append("Bearish technical trend detected")
        
        stop_loss_check = self._check_stop_loss(position, current_price)
        if stop_loss_check['triggered']:
            insight['action_required'] = f"STOP LOSS TRIGGERED: {stop_loss_check['reason']}"
            insight['recommended_action'] = 'SELL'
        
        take_profit_check = self._check_take_profit(position, current_price)
        if take_profit_check['triggered']:
            insight['action_required'] = f"TAKE PROFIT: {take_profit_check['reason']}"
            insight['recommended_action'] = 'SELL'
        
        risk_check = self._check_risk_exposure(position, current_price)
        if risk_check['high_risk']:
            insight['signals'].append(f"High risk exposure: {risk_check['reason']}")
        
        rebalance_check = self._check_rebalancing_need(position, current_price, pnl_pct)
        if rebalance_check['needed']:
            insight['signals'].append(f"Rebalancing suggested: {rebalance_check['reason']}")
            insight['rebalance_action'] = rebalance_check['action']
        
        self._update_position_history(symbol, insight)
        
        return insight
    
    def _check_news_sentiment(self, symbol: str) -> Dict:
        """
        Check recent news sentiment for the symbol
        """
        try:
            news_articles = self.data_fetcher.fetch_news(symbol, days=1)
            
            if not news_articles:
                return {'sentiment': 'neutral', 'alert_level': 'low'}
            
            sentiment_data = self.sentiment_analyzer.analyze_symbol_sentiment(symbol, news_articles)
            
            alert_level = 'low'
            if abs(sentiment_data.get('overall_sentiment', 0)) > 0.7:
                alert_level = 'high'
            elif abs(sentiment_data.get('overall_sentiment', 0)) > 0.4:
                alert_level = 'medium'
            
            recent_headlines = [article.get('headline', '')[:100] for article in news_articles[:3]]
            
            return {
                'sentiment': sentiment_data.get('signal', 'neutral'),
                'score': sentiment_data.get('overall_sentiment', 0),
                'alert_level': alert_level,
                'article_count': len(news_articles),
                'recent_headlines': recent_headlines,
                'summary': f"{len(news_articles)} articles, {sentiment_data.get('signal', 'neutral')} sentiment"
            }
            
        except Exception as e:
            log.error(f"Error checking news for {symbol}: {str(e)}")
            return {'sentiment': 'unknown', 'alert_level': 'low', 'error': str(e)}
    
    def _check_earnings_calendar(self, symbol: str) -> Dict:
        """
        Check upcoming earnings date for the symbol
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                earnings_date = calendar.get('Earnings Date')
                if earnings_date is not None and len(earnings_date) > 0:
                    next_earnings = pd.to_datetime(earnings_date[0])
                    days_until = (next_earnings - datetime.now()).days
                    
                    return {
                        'has_earnings': True,
                        'earnings_date': next_earnings.strftime('%Y-%m-%d'),
                        'days_until_earnings': days_until,
                        'approaching': days_until <= 7
                    }
            
            return {'has_earnings': False}
            
        except Exception as e:
            log.debug(f"Could not fetch earnings for {symbol}: {str(e)}")
            return {'has_earnings': False, 'error': str(e)}
    
    def _check_technical_signals(self, symbol: str, current_price: float) -> Dict:
        """
        Check technical indicators for the position
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            df = self.data_fetcher.fetch_historical_data(symbol, start_date, end_date)
            
            if df is None or len(df) < 20:
                return {'trend': 'unknown', 'strength': 0}
            
            df = self.technical_indicators.add_all_indicators(df)
            
            latest = df.iloc[-1]
            
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            
            trend = 'neutral'
            if rsi > 70 and macd < macd_signal:
                trend = 'bearish'
            elif rsi < 30 and macd > macd_signal:
                trend = 'bullish'
            elif macd > macd_signal:
                trend = 'bullish'
            elif macd < macd_signal:
                trend = 'bearish'
            
            return {
                'trend': trend,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'support': latest.get('BB_Lower', current_price * 0.95),
                'resistance': latest.get('BB_Upper', current_price * 1.05)
            }
            
        except Exception as e:
            log.error(f"Error checking technicals for {symbol}: {str(e)}")
            return {'trend': 'unknown', 'error': str(e)}
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> Dict:
        """
        Check if stop-loss should be triggered
        """
        entry_price = position.get('entry_price', current_price)
        stop_loss_pct = config.get('risk.stop_loss_pct', 0.05)
        
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        if current_price <= stop_loss_price:
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            return {
                'triggered': True,
                'reason': f"Price ${current_price:.2f} below stop-loss ${stop_loss_price:.2f} ({loss_pct:.1f}%)"
            }
        
        trailing_stop = position.get('trailing_stop')
        if trailing_stop and current_price <= trailing_stop:
            return {
                'triggered': True,
                'reason': f"Trailing stop triggered at ${trailing_stop:.2f}"
            }
        
        return {'triggered': False}
    
    def _check_take_profit(self, position: Dict, current_price: float) -> Dict:
        """
        Check if take-profit should be triggered
        """
        entry_price = position.get('entry_price', current_price)
        take_profit_pct = config.get('risk.take_profit_pct', 0.15)
        
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        if current_price >= take_profit_price:
            gain_pct = ((current_price - entry_price) / entry_price) * 100
            return {
                'triggered': True,
                'reason': f"Price ${current_price:.2f} above take-profit ${take_profit_price:.2f} (+{gain_pct:.1f}%)"
            }
        
        return {'triggered': False}
    
    def _check_risk_exposure(self, position: Dict, current_price: float) -> Dict:
        """
        Check if position has high risk exposure
        """
        position_value = position.get('shares', 0) * current_price
        max_position_size = config.get('risk.max_position_size', 10000)
        
        if position_value > max_position_size * 1.5:
            return {
                'high_risk': True,
                'reason': f"Position value ${position_value:.2f} exceeds max size ${max_position_size:.2f}"
            }
        
        return {'high_risk': False}
    
    def _check_rebalancing_need(
        self,
        position: Dict,
        current_price: float,
        pnl_pct: float
    ) -> Dict:
        """
        Check if position needs rebalancing
        """
        if pnl_pct > 50:
            return {
                'needed': True,
                'reason': f"Large gain (+{pnl_pct:.1f}%) - consider taking partial profits",
                'action': 'PARTIAL_SELL'
            }
        
        if pnl_pct < -20:
            return {
                'needed': True,
                'reason': f"Large loss ({pnl_pct:.1f}%) - consider cutting position",
                'action': 'REDUCE'
            }
        
        return {'needed': False}
    
    def _update_position_history(self, symbol: str, insight: Dict):
        """
        Update historical tracking for the position
        """
        if symbol not in self.position_history:
            self.position_history[symbol] = []
        
        self.position_history[symbol].append({
            'timestamp': datetime.now(),
            'price': insight.get('current_price'),
            'pnl_pct': insight.get('pnl_pct'),
            'signals': insight.get('signals', [])
        })
        
        if len(self.position_history[symbol]) > 100:
            self.position_history[symbol] = self.position_history[symbol][-100:]
    
    def generate_exit_signals(
        self,
        monitoring_insights: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Generate exit signals based on monitoring insights
        
        Returns:
            List of exit recommendations
        """
        exit_signals = []
        
        for symbol, insight in monitoring_insights.items():
            if insight.get('recommended_action') == 'SELL':
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': insight.get('action_required', 'Exit recommended'),
                    'urgency': 'high',
                    'timestamp': datetime.now()
                })
            
            elif insight.get('rebalance_action') in ['PARTIAL_SELL', 'REDUCE']:
                exit_signals.append({
                    'symbol': symbol,
                    'action': insight['rebalance_action'],
                    'reason': insight.get('signals', ['Rebalancing needed'])[0],
                    'urgency': 'medium',
                    'timestamp': datetime.now()
                })
        
        if exit_signals:
            log.info(f"Generated {len(exit_signals)} exit signals")
            for signal in exit_signals:
                log.info(f"  {signal['symbol']}: {signal['action']} - {signal['reason']}")
        
        return exit_signals
