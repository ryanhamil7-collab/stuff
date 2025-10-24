"""
Performance monitoring and metrics tracking for the trading system.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import numpy as np

from .logger import log


class PerformanceMonitor:
    """
    Monitor system performance, resource usage, and trading metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of historical data points to keep
        """
        self.max_history = max_history
        self.metrics = {
            'execution_times': deque(maxlen=max_history),
            'memory_usage': deque(maxlen=max_history),
            'cpu_usage': deque(maxlen=max_history),
            'api_calls': deque(maxlen=max_history),
            'errors': deque(maxlen=max_history),
            'trades': deque(maxlen=max_history)
        }
        self.counters = {
            'total_api_calls': 0,
            'total_errors': 0,
            'total_trades': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        log.info("PerformanceMonitor initialized")
    
    def record_execution_time(self, operation: str, duration: float) -> None:
        """
        Record execution time for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        with self._lock:
            self.metrics['execution_times'].append({
                'operation': operation,
                'duration': duration,
                'timestamp': datetime.now()
            })
    
    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            with self._lock:
                self.metrics['memory_usage'].append({
                    'rss': memory_info.rss / 1024 / 1024,  # MB
                    'vms': memory_info.vms / 1024 / 1024,  # MB
                    'percent': process.memory_percent(),
                    'timestamp': datetime.now()
                })
        except Exception as e:
            log.error(f"Error recording memory usage: {str(e)}")
    
    def record_cpu_usage(self) -> None:
        """Record current CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            with self._lock:
                self.metrics['cpu_usage'].append({
                    'percent': cpu_percent,
                    'timestamp': datetime.now()
                })
        except Exception as e:
            log.error(f"Error recording CPU usage: {str(e)}")
    
    def record_api_call(self, service: str, success: bool, duration: float) -> None:
        """
        Record an API call.
        
        Args:
            service: Name of the API service
            success: Whether the call was successful
            duration: Duration in seconds
        """
        with self._lock:
            self.metrics['api_calls'].append({
                'service': service,
                'success': success,
                'duration': duration,
                'timestamp': datetime.now()
            })
            self.counters['total_api_calls'] += 1
    
    def record_error(self, error_type: str, message: str) -> None:
        """
        Record an error.
        
        Args:
            error_type: Type of error
            message: Error message
        """
        with self._lock:
            self.metrics['errors'].append({
                'type': error_type,
                'message': message,
                'timestamp': datetime.now()
            })
            self.counters['total_errors'] += 1
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        profit: Optional[float] = None
    ) -> None:
        """
        Record a trade execution.
        
        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            quantity: Trade quantity
            price: Execution price
            profit: Realized profit (if applicable)
        """
        with self._lock:
            self.metrics['trades'].append({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'profit': profit,
                'timestamp': datetime.now()
            })
            self.counters['total_trades'] += 1
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.counters['cache_hits'] += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.counters['cache_misses'] += 1
    
    def get_execution_time_stats(self, operation: Optional[str] = None) -> Dict[str, float]:
        """
        Get execution time statistics.
        
        Args:
            operation: Filter by operation name (optional)
            
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            times = [
                entry['duration'] 
                for entry in self.metrics['execution_times']
                if operation is None or entry['operation'] == operation
            ]
        
        if not times:
            return {}
        
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'count': len(times)
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self.metrics['memory_usage']:
                return {}
            
            rss_values = [entry['rss'] for entry in self.metrics['memory_usage']]
            percent_values = [entry['percent'] for entry in self.metrics['memory_usage']]
        
        return {
            'current_rss_mb': rss_values[-1] if rss_values else 0,
            'mean_rss_mb': np.mean(rss_values),
            'max_rss_mb': np.max(rss_values),
            'current_percent': percent_values[-1] if percent_values else 0,
            'mean_percent': np.mean(percent_values),
            'max_percent': np.max(percent_values)
        }
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """
        Get CPU usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self.metrics['cpu_usage']:
                return {}
            
            cpu_values = [entry['percent'] for entry in self.metrics['cpu_usage']]
        
        return {
            'current': cpu_values[-1] if cpu_values else 0,
            'mean': np.mean(cpu_values),
            'max': np.max(cpu_values),
            'min': np.min(cpu_values)
        }
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get API call statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self.metrics['api_calls']:
                return {'total_calls': 0}
            
            calls = list(self.metrics['api_calls'])
            total_calls = len(calls)
            successful_calls = sum(1 for call in calls if call['success'])
            durations = [call['duration'] for call in calls]
            
            by_service = {}
            for call in calls:
                service = call['service']
                if service not in by_service:
                    by_service[service] = {'count': 0, 'success': 0, 'durations': []}
                by_service[service]['count'] += 1
                if call['success']:
                    by_service[service]['success'] += 1
                by_service[service]['durations'].append(call['duration'])
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'mean_duration': np.mean(durations) if durations else 0,
            'by_service': {
                service: {
                    'count': stats['count'],
                    'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0,
                    'mean_duration': np.mean(stats['durations']) if stats['durations'] else 0
                }
                for service, stats in by_service.items()
            }
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self.metrics['errors']:
                return {'total_errors': 0}
            
            errors = list(self.metrics['errors'])
            
            by_type = {}
            for error in errors:
                error_type = error['type']
                if error_type not in by_type:
                    by_type[error_type] = 0
                by_type[error_type] += 1
        
        return {
            'total_errors': len(errors),
            'by_type': by_type,
            'recent_errors': [
                {'type': e['type'], 'message': e['message'][:100], 'timestamp': e['timestamp']}
                for e in list(errors)[-10:]
            ]
        }
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """
        Get trading statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if not self.metrics['trades']:
                return {'total_trades': 0}
            
            trades = list(self.metrics['trades'])
            
            total_trades = len(trades)
            buy_trades = sum(1 for t in trades if t['action'].lower() == 'buy')
            sell_trades = sum(1 for t in trades if t['action'].lower() == 'sell')
            
            profits = [t['profit'] for t in trades if t['profit'] is not None]
            total_profit = sum(profits) if profits else 0
            winning_trades = sum(1 for p in profits if p > 0)
            losing_trades = sum(1 for p in profits if p < 0)
            
            by_symbol = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = {'count': 0, 'volume': 0}
                by_symbol[symbol]['count'] += 1
                by_symbol[symbol]['volume'] += trade['quantity'] * trade['price']
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_profit': total_profit,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(profits) if profits else 0,
            'average_profit': np.mean(profits) if profits else 0,
            'by_symbol': by_symbol
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_requests = self.counters['cache_hits'] + self.counters['cache_misses']
            hit_rate = self.counters['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.counters['cache_hits'],
            'cache_misses': self.counters['cache_misses'],
            'total_requests': total_requests,
            'hit_rate': hit_rate
        }
    
    def get_uptime(self) -> float:
        """
        Get system uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            'uptime_seconds': self.get_uptime(),
            'execution_times': self.get_execution_time_stats(),
            'memory': self.get_memory_stats(),
            'cpu': self.get_cpu_stats(),
            'api_calls': self.get_api_stats(),
            'errors': self.get_error_stats(),
            'trades': self.get_trade_stats(),
            'cache': self.get_cache_stats(),
            'counters': dict(self.counters)
        }
    
    def print_summary(self) -> None:
        """Print performance summary to log."""
        summary = self.get_summary()
        
        log.info("=" * 80)
        log.info("PERFORMANCE SUMMARY")
        log.info("=" * 80)
        log.info(f"Uptime: {summary['uptime_seconds']:.2f} seconds")
        
        if summary['memory']:
            log.info(f"Memory: {summary['memory']['current_rss_mb']:.2f} MB "
                    f"(avg: {summary['memory']['mean_rss_mb']:.2f} MB, "
                    f"max: {summary['memory']['max_rss_mb']:.2f} MB)")
        
        if summary['cpu']:
            log.info(f"CPU: {summary['cpu']['current']:.1f}% "
                    f"(avg: {summary['cpu']['mean']:.1f}%, max: {summary['cpu']['max']:.1f}%)")
        
        if summary['api_calls']['total_calls'] > 0:
            log.info(f"API Calls: {summary['api_calls']['total_calls']} "
                    f"(success rate: {summary['api_calls']['success_rate']:.2%})")
        
        if summary['trades']['total_trades'] > 0:
            log.info(f"Trades: {summary['trades']['total_trades']} "
                    f"(win rate: {summary['trades']['win_rate']:.2%}, "
                    f"profit: ${summary['trades']['total_profit']:.2f})")
        
        if summary['errors']['total_errors'] > 0:
            log.warning(f"Errors: {summary['errors']['total_errors']}")
        
        if summary['cache']['total_requests'] > 0:
            log.info(f"Cache: {summary['cache']['hit_rate']:.2%} hit rate "
                    f"({summary['cache']['cache_hits']} hits, {summary['cache']['cache_misses']} misses)")
        
        log.info("=" * 80)
    
    def reset(self) -> None:
        """Reset all metrics and counters."""
        with self._lock:
            for key in self.metrics:
                self.metrics[key].clear()
            for key in self.counters:
                self.counters[key] = 0
            self.start_time = time.time()
        
        log.info("Performance monitor reset")


_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize performance timer.
        
        Args:
            operation: Name of the operation
            monitor: Performance monitor instance (uses global if None)
        """
        self.operation = operation
        self.monitor = monitor or get_performance_monitor()
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        duration = time.time() - self.start_time
        self.monitor.record_execution_time(self.operation, duration)
        
        if exc_type is not None:
            self.monitor.record_error(
                error_type=exc_type.__name__,
                message=str(exc_val)
            )
