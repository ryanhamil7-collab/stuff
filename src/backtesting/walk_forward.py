import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from src.utils import log, config
from src.backtesting import BacktestEngine
from scipy.stats import norm

class WalkForwardValidator:
    
    def __init__(self, train_window: int = 365, test_window: int = 90, step_size: int = 90):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        log.info(f"WalkForwardValidator initialized: train={train_window}d, test={test_window}d, step={step_size}d")
    
    def generate_windows(
        self, 
        start_date: str, 
        end_date: str
    ) -> List[Tuple[str, str, str, str]]:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        windows = []
        current_start = start_dt
        
        while True:
            train_end = current_start + timedelta(days=self.train_window)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_window)
            
            if test_end > end_dt:
                break
            
            windows.append((
                current_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            current_start += timedelta(days=self.step_size)
        
        log.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def run_walk_forward(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> Dict:
        log.info("Starting walk-forward validation...")
        
        windows = self.generate_windows(start_date, end_date)
        
        if len(windows) == 0:
            log.error("No valid windows generated for walk-forward validation")
            return {
                'success': False,
                'error': 'Insufficient data for walk-forward validation'
            }
        
        backtest_engine = BacktestEngine()
        
        train_results = []
        test_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            log.info(f"Window {i+1}/{len(windows)}: Train={train_start} to {train_end}, Test={test_start} to {test_end}")
            
            try:
                train_result = backtest_engine.run_backtest(data, train_start, train_end)
                train_results.append({
                    'window': i + 1,
                    'start_date': train_start,
                    'end_date': train_end,
                    'type': 'train',
                    'sharpe_ratio': train_result['sharpe_ratio'],
                    'total_return': train_result['total_return'],
                    'max_drawdown': train_result['max_drawdown'],
                    'win_rate': train_result['win_rate'],
                    'num_trades': train_result['num_trades']
                })
                
                test_result = backtest_engine.run_backtest(data, test_start, test_end)
                test_results.append({
                    'window': i + 1,
                    'start_date': test_start,
                    'end_date': test_end,
                    'type': 'test',
                    'sharpe_ratio': test_result['sharpe_ratio'],
                    'total_return': test_result['total_return'],
                    'max_drawdown': test_result['max_drawdown'],
                    'win_rate': test_result['win_rate'],
                    'num_trades': test_result['num_trades']
                })
                
                log.info(f"  Train Sharpe: {train_result['sharpe_ratio']:.2f}, Test Sharpe: {test_result['sharpe_ratio']:.2f}")
                
                overfitting_threshold = config.get('backtest.validation.overfitting_threshold', 0.7)
                if test_result['sharpe_ratio'] < train_result['sharpe_ratio'] * overfitting_threshold:
                    log.warning(f"  ⚠ Potential overfitting detected in window {i+1}")
                    log.warning(f"  Test Sharpe ({test_result['sharpe_ratio']:.2f}) < Train Sharpe ({train_result['sharpe_ratio']:.2f}) * {overfitting_threshold}")
                
            except Exception as e:
                log.error(f"Error in window {i+1}: {str(e)}")
                continue
        
        if not train_results or not test_results:
            log.error("Walk-forward validation failed - no valid results")
            return {
                'success': False,
                'error': 'No valid results from walk-forward validation'
            }
        
        train_df = pd.DataFrame(train_results)
        test_df = pd.DataFrame(test_results)
        
        avg_train_sharpe = train_df['sharpe_ratio'].mean()
        avg_test_sharpe = test_df['sharpe_ratio'].mean()
        
        avg_train_return = train_df['total_return'].mean()
        avg_test_return = test_df['total_return'].mean()
        
        sharpe_degradation = (avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe if avg_train_sharpe != 0 else 0
        
        log.info("=" * 80)
        log.info("WALK-FORWARD VALIDATION RESULTS")
        log.info("=" * 80)
        log.info(f"Total Windows: {len(windows)}")
        log.info(f"Average Train Sharpe: {avg_train_sharpe:.2f}")
        log.info(f"Average Test Sharpe: {avg_test_sharpe:.2f}")
        log.info(f"Sharpe Degradation: {sharpe_degradation:.2%}")
        log.info(f"Average Train Return: {avg_train_return:.2%}")
        log.info(f"Average Test Return: {avg_test_return:.2%}")
        log.info("=" * 80)
        
        overfitting_threshold = config.get('backtest.validation.overfitting_threshold', 0.7)
        if avg_test_sharpe < avg_train_sharpe * overfitting_threshold:
            log.error(f"⚠ OVERFITTING DETECTED: Test performance significantly worse than training")
            log.error(f"Consider simplifying the model or adding regularization")
        else:
            log.info(f"✓ Model appears robust across different time periods")
        
        return {
            'success': True,
            'num_windows': len(windows),
            'train_results': train_df,
            'test_results': test_df,
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'sharpe_degradation': sharpe_degradation,
            'avg_train_return': avg_train_return,
            'avg_test_return': avg_test_return,
            'overfitting_detected': avg_test_sharpe < avg_train_sharpe * overfitting_threshold
        }
    
    def save_results(self, results: Dict, output_dir: str = 'data'):
        if not results.get('success'):
            log.warning("Cannot save results - validation failed")
            return
        
        train_df = results['train_results']
        test_df = results['test_results']
        
        train_df.to_csv(f'{output_dir}/walk_forward_train.csv', index=False)
        test_df.to_csv(f'{output_dir}/walk_forward_test.csv', index=False)
        
        log.info(f"Walk-forward results saved to {output_dir}/")
