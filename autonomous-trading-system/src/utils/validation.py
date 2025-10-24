import pandas as pd
import numpy as np
from typing import Dict
from src.utils import log

class DataValidator:
    
    @staticmethod
    def check_lookahead_bias(data: Dict[str, pd.DataFrame]) -> bool:
        log.info("Checking for lookahead bias in data...")
        
        has_lookahead = False
        
        for symbol, df in data.items():
            if 'Date' not in df.columns:
                continue
            
            df_sorted = df.sort_values('Date')
            
            for col in df.columns:
                if col in ['Date', 'Symbol']:
                    continue
                
                if df_sorted[col].isna().sum() > 0:
                    first_valid_idx = df_sorted[col].first_valid_index()
                    if first_valid_idx is not None and first_valid_idx > 0:
                        log.warning(f"Potential lookahead in {symbol}.{col}: NaN values at start")
            
            if 'Close' in df.columns and len(df) > 1:
                returns = df['Close'].pct_change()
                
                for col in df.columns:
                    if col in ['Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Volume']:
                        continue
                    
                    if col in df.columns and not df[col].isna().all():
                        corr = df[col].shift(1).corr(returns)
                        
                        if abs(corr) > 0.95:
                            log.error(f"LOOKAHEAD DETECTED: {symbol}.{col} has suspiciously high correlation ({corr:.3f}) with future returns")
                            has_lookahead = True
        
        if has_lookahead:
            log.error("Lookahead bias detected! Fix before proceeding.")
            return False
        else:
            log.info("No obvious lookahead bias detected")
            return True
    
    @staticmethod
    def validate_time_series_split(train_end: str, test_start: str) -> bool:
        from datetime import datetime
        
        train_end_dt = datetime.strptime(train_end, '%Y-%m-%d')
        test_start_dt = datetime.strptime(test_start, '%Y-%m-%d')
        
        if test_start_dt <= train_end_dt:
            log.error(f"Invalid split: test_start ({test_start}) must be after train_end ({train_end})")
            return False
        
        gap_days = (test_start_dt - train_end_dt).days
        if gap_days < 1:
            log.warning(f"No gap between train and test sets - consider adding buffer")
        
        return True
    
    @staticmethod
    def check_data_quality(data: Dict[str, pd.DataFrame]) -> Dict:
        log.info("Checking data quality...")
        
        quality_report = {}
        
        for symbol, df in data.items():
            report = {
                'total_rows': len(df),
                'missing_values': df.isna().sum().to_dict(),
                'duplicate_dates': 0,
                'negative_prices': 0,
                'zero_volume': 0
            }
            
            if 'Date' in df.columns:
                report['duplicate_dates'] = df['Date'].duplicated().sum()
            
            if 'Close' in df.columns:
                report['negative_prices'] = (df['Close'] <= 0).sum()
            
            if 'Volume' in df.columns:
                report['zero_volume'] = (df['Volume'] == 0).sum()
            
            quality_report[symbol] = report
            
            if report['duplicate_dates'] > 0:
                log.warning(f"{symbol}: {report['duplicate_dates']} duplicate dates found")
            
            if report['negative_prices'] > 0:
                log.error(f"{symbol}: {report['negative_prices']} negative prices found")
        
        return quality_report

def validate_no_lookahead(data: Dict[str, pd.DataFrame]) -> bool:
    validator = DataValidator()
    return validator.check_lookahead_bias(data)
