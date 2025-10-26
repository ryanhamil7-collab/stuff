import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from src.strategies import RiskManager
from src.strategies.portfolio import Portfolio
from src.utils import log, config

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    log.warning("RL dependencies (gymnasium, stable-baselines3) not available. RL training disabled.")

if RL_AVAILABLE:
    class TradingEnvironment(gym.Env):
        
        def __init__(
            self,
            data: Dict[str, pd.DataFrame],
            initial_capital: float = 100000.0
        ):
            super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        
        self.current_step = 0
        self.max_steps = min(len(df) for df in data.values()) - 1
        
        num_features = 20
        num_symbols = len(self.symbols)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_symbols * num_features + 3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_symbols,),
            dtype=np.float32
        )
        
        self.reset()
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            self.current_step = 50
            self.portfolio.reset()
            self.risk_manager = RiskManager(self.initial_capital)
            
            return self._get_observation(), {}
        
        def _get_observation(self) -> np.ndarray:
            obs = []
            
            for symbol in self.symbols:
                df = self.data[symbol]
                if self.current_step >= len(df):
                    obs.extend([0.0] * 20)
                    continue
                
                row = df.iloc[self.current_step]
                
                features = [
                    row.get('Close', 0) / 100.0,
                    row.get('Volume', 0) / 1e6,
                    row.get('RSI', 50) / 100.0,
                    row.get('MACD', 0) / 10.0,
                    row.get('MACD_Signal', 0) / 10.0,
                    row.get('SMA_20', 0) / 100.0,
                    row.get('SMA_50', 0) / 100.0,
                    row.get('EMA_12', 0) / 100.0,
                    row.get('EMA_26', 0) / 100.0,
                    row.get('BB_Position', 0.5),
                    row.get('ATR', 0) / 10.0,
                    row.get('ADX', 0) / 100.0,
                    row.get('OBV', 0) / 1e6,
                    row.get('VWAP', 0) / 100.0,
                    row.get('Stoch_K', 50) / 100.0,
                    row.get('Stoch_D', 50) / 100.0,
                    row.get('Plus_DI', 0) / 100.0,
                    row.get('Minus_DI', 0) / 100.0,
                    row.get('BB_Width', 0) / 10.0,
                    row.get('Signal', 0)
                ]
                
                obs.extend(features)
            
            total_value = self.portfolio.get_total_value(self._get_current_prices())
            obs.append(total_value / self.initial_capital)
            obs.append(len(self.portfolio.positions) / 10.0)
            obs.append(self.portfolio.cash / self.initial_capital)
            
            return np.array(obs, dtype=np.float32)
        
        def _get_current_prices(self) -> Dict[str, float]:
            prices = {}
            for symbol in self.symbols:
                df = self.data[symbol]
                if self.current_step < len(df):
                    prices[symbol] = df.iloc[self.current_step]['Close']
                else:
                    prices[symbol] = 0.0
            return prices
        
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            current_prices = self._get_current_prices()
            
            for i, symbol in enumerate(self.symbols):
                action_value = float(action[i])
                price = current_prices.get(symbol, 0)
                
                if price == 0:
                    continue
                
                if action_value > 0.3 and not self.portfolio.has_position(symbol):
                    shares = self.risk_manager.calculate_position_size_fixed(price)
                    
                    if shares > 0:
                        equity_series = pd.Series(self.portfolio.equity_curve)
                        is_valid, msg = self.risk_manager.validate_trade(
                            symbol, price, shares,
                            self.portfolio.get_open_positions(),
                            equity_series
                        )
                        
                        if is_valid:
                            self.portfolio.open_position(symbol, shares, price)
                
                elif action_value < -0.3 and self.portfolio.has_position(symbol):
                    self.portfolio.close_position(symbol, price, reason='rl_signal')
            
            self.portfolio.update_positions(current_prices)
            
            reward = self._calculate_reward()
            
            self.current_step += 1
            
            done = self.current_step >= self.max_steps
            truncated = False
            
            obs = self._get_observation()
            
            info = {
                'portfolio_value': self.portfolio.get_total_value(current_prices),
                'num_positions': len(self.portfolio.positions),
                'cash': self.portfolio.cash
            }
            
            return obs, reward, done, truncated, info
        
        def _calculate_reward(self) -> float:
            if len(self.portfolio.equity_curve) < 2:
                return 0.0
            
            equity_series = pd.Series(self.portfolio.equity_curve)
            returns = equity_series.pct_change().dropna()
            
            if len(returns) == 0:
                return 0.0
            
            recent_return = returns.iloc[-1] if len(returns) > 0 else 0.0
            
            sharpe = 0.0
            if len(returns) > 10:
                sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
            
            cumulative_max = equity_series.expanding().max()
            drawdown = (equity_series - cumulative_max) / cumulative_max
            current_drawdown = drawdown.iloc[-1]
            
            reward = (
                recent_return * 100 +
                sharpe * 0.1 -
                abs(current_drawdown) * 10
            )
            
            return float(reward)
else:
    class TradingEnvironment:
        def __init__(self, *args, **kwargs):
            raise ImportError("RL dependencies not available. Install gymnasium and stable-baselines3 to use RL features.")

class DecisionAgent:
    
    def __init__(self):
        self.model = None
        self.env = None
        self.risk_manager = RiskManager()
        log.info("DecisionAgent initialized")
    
    def train_rl_model(
        self,
        data: Dict[str, pd.DataFrame],
        total_timesteps: int = 10000
    ):
        if not RL_AVAILABLE:
            log.error("Cannot train RL model: gymnasium and stable-baselines3 not available")
            return
        
        log.info(f"Training RL model with {total_timesteps} timesteps")
        
        self.env = DummyVecEnv([lambda: TradingEnvironment(data)])
        
        learning_rate = config.get('rl.learning_rate', 0.0003)
        n_steps = config.get('rl.n_steps', 2048)
        batch_size = config.get('rl.batch_size', 64)
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=config.get('rl.n_epochs', 10),
            gamma=config.get('rl.gamma', 0.99),
            gae_lambda=config.get('rl.gae_lambda', 0.95),
            clip_range=config.get('rl.clip_range', 0.2),
            ent_coef=config.get('rl.ent_coef', 0.01),
            vf_coef=config.get('rl.vf_coef', 0.5),
            max_grad_norm=config.get('rl.max_grad_norm', 0.5),
            verbose=1
        )
        
        self.model.learn(total_timesteps=total_timesteps)
        
        log.info("RL model training completed")
    
    def make_decisions(
        self,
        signals: Dict[str, Dict],
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        log.info(f"Making trading decisions for {len(signals)} symbols")
        
        decisions = []
        
        equity_series = pd.Series(portfolio.equity_curve)
        
        for symbol, signal in signals.items():
            action = signal.get('action', 'HOLD')
            price = current_prices.get(symbol, 0)
            
            if price == 0:
                continue
            
            if action == 'BUY' and not portfolio.has_position(symbol):
                shares = self.risk_manager.calculate_position_size_fixed(price)
                
                if shares > 0:
                    is_valid, msg = self.risk_manager.validate_trade(
                        symbol, price, shares,
                        portfolio.get_open_positions(),
                        equity_series
                    )
                    
                    if is_valid:
                        decisions.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'signal_score': signal.get('combined_score', 0),
                            'llm_confidence': signal.get('llm_confidence', 0),
                            'timestamp': datetime.now()
                        })
                        log.info(f"Decision: BUY {shares} shares of {symbol} @ ${price:.2f}")
                    else:
                        log.info(f"Trade validation failed for {symbol}: {msg}")
            
            elif action == 'SELL' and portfolio.has_position(symbol):
                position = portfolio.get_position(symbol)
                
                should_close = False
                reason = 'signal'
                
                if self.risk_manager.check_stop_loss(price, position['entry_price']):
                    should_close = True
                    reason = 'stop_loss'
                elif self.risk_manager.check_take_profit(price, position['entry_price']):
                    should_close = True
                    reason = 'take_profit'
                elif signal.get('combined_score', 0) < -0.5:
                    should_close = True
                    reason = 'strong_sell_signal'
                
                if should_close:
                    decisions.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': position['shares'],
                        'price': price,
                        'reason': reason,
                        'timestamp': datetime.now()
                    })
                    log.info(f"Decision: SELL {position['shares']} shares of {symbol} @ ${price:.2f} (reason: {reason})")
        
        log.info(f"Generated {len(decisions)} trading decisions")
        return decisions
    
    def execute_decisions(
        self,
        decisions: List[Dict],
        portfolio: Portfolio
    ) -> Dict:
        log.info(f"Executing {len(decisions)} trading decisions")
        
        executed = []
        failed = []
        
        for decision in decisions:
            symbol = decision['symbol']
            action = decision['action']
            price = decision['price']
            shares = decision['shares']
            
            try:
                if action == 'BUY':
                    success = portfolio.open_position(symbol, shares, price)
                    if success:
                        executed.append(decision)
                    else:
                        failed.append(decision)
                
                elif action == 'SELL':
                    reason = decision.get('reason', 'signal')
                    success = portfolio.close_position(symbol, price, reason=reason)
                    if success:
                        executed.append(decision)
                    else:
                        failed.append(decision)
                
            except Exception as e:
                log.error(f"Error executing decision for {symbol}: {str(e)}")
                failed.append(decision)
        
        result = {
            'executed': executed,
            'failed': failed,
            'num_executed': len(executed),
            'num_failed': len(failed),
            'timestamp': datetime.now()
        }
        
        log.info(f"Executed {len(executed)} decisions, {len(failed)} failed")
        return result
    
    def run(
        self,
        signals: Dict[str, Dict],
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> Dict:
        log.info("DecisionAgent starting decision making")
        
        decisions = self.make_decisions(signals, portfolio, current_prices)
        
        execution_result = self.execute_decisions(decisions, portfolio)
        
        result = {
            'decisions': decisions,
            'execution': execution_result,
            'timestamp': datetime.now()
        }
        
        log.info("DecisionAgent completed successfully")
        return result
