"""
Google Drive Storage Manager

Handles persistent storage of models, training data, and checkpoints
in Google Drive for Colab environments.

Features:
- Automatic Drive mounting
- Model checkpoint management
- Training data caching
- Version control for models
- Automatic sync on save/load
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.utils import log

class DriveStorageManager:
    """
    Manages persistent storage in Google Drive for Colab
    """
    
    def __init__(
        self,
        drive_root: str = "/content/drive/MyDrive/autonomous_trading",
        auto_mount: bool = True
    ):
        """
        Initialize Drive storage manager
        
        Args:
            drive_root: Root directory in Google Drive
            auto_mount: Automatically mount Drive if not mounted
        """
        self.drive_root = Path(drive_root)
        self.is_colab = self._check_colab_environment()
        
        if self.is_colab and auto_mount:
            self._mount_drive()
        
        self._setup_directories()
        
        log.info(f"DriveStorageManager initialized - Root: {self.drive_root}")
    
    def _check_colab_environment(self) -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _mount_drive(self):
        """Mount Google Drive in Colab"""
        if not self.is_colab:
            log.warning("Not in Colab environment, skipping Drive mount")
            return
        
        try:
            from google.colab import drive
            
            if not os.path.exists('/content/drive'):
                log.info("Mounting Google Drive...")
                drive.mount('/content/drive', force_remount=False)
                log.info("✓ Google Drive mounted successfully")
            else:
                log.info("Google Drive already mounted")
                
        except Exception as e:
            log.error(f"Failed to mount Google Drive: {str(e)}")
            raise
    
    def _setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.drive_root,
            self.drive_root / "models" / "rl_checkpoints",
            self.drive_root / "models" / "llm_checkpoints",
            self.drive_root / "data" / "historical",
            self.drive_root / "data" / "cache",
            self.drive_root / "results" / "backtests",
            self.drive_root / "results" / "performance",
            self.drive_root / "config"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        log.info(f"✓ Directory structure created in {self.drive_root}")
    
    def save_rl_model(
        self,
        model,
        symbol: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save RL model checkpoint to Drive
        
        Args:
            model: RL model instance
            symbol: Trading symbol
            metadata: Optional metadata about the model
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rl_trader_{symbol}_{timestamp}.zip"
        checkpoint_path = self.drive_root / "models" / "rl_checkpoints" / filename
        
        try:
            model.save(str(checkpoint_path))
            
            if metadata:
                metadata_path = checkpoint_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self._update_model_registry('rl', symbol, str(checkpoint_path), metadata)
            
            log.info(f"✓ RL model saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            log.error(f"Failed to save RL model: {str(e)}")
            raise
    
    def load_latest_rl_model(
        self,
        symbol: str,
        model_class
    ) -> Optional[Any]:
        """
        Load latest RL model checkpoint for symbol
        
        Args:
            symbol: Trading symbol
            model_class: RL model class (PPO or DQN)
            
        Returns:
            Loaded model or None if not found
        """
        checkpoint_dir = self.drive_root / "models" / "rl_checkpoints"
        
        checkpoints = list(checkpoint_dir.glob(f"rl_trader_{symbol}_*.zip"))
        
        if not checkpoints:
            log.info(f"No RL checkpoints found for {symbol}")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            model = model_class.load(str(latest_checkpoint))
            log.info(f"✓ Loaded RL model: {latest_checkpoint.name}")
            return model
            
        except Exception as e:
            log.error(f"Failed to load RL model: {str(e)}")
            return None
    
    def save_training_data(
        self,
        data: Dict,
        data_type: str,
        symbol: Optional[str] = None
    ):
        """
        Save training data to Drive
        
        Args:
            data: Data to save (dict or DataFrame)
            data_type: Type of data (e.g., 'historical', 'features')
            symbol: Optional symbol for symbol-specific data
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        
        if symbol:
            filename = f"{data_type}_{symbol}_{timestamp}.pkl"
        else:
            filename = f"{data_type}_{timestamp}.pkl"
        
        data_path = self.drive_root / "data" / "cache" / filename
        
        try:
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            
            log.info(f"✓ Training data saved: {data_path}")
            
        except Exception as e:
            log.error(f"Failed to save training data: {str(e)}")
    
    def load_training_data(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        max_age_days: int = 7
    ) -> Optional[Dict]:
        """
        Load cached training data from Drive
        
        Args:
            data_type: Type of data to load
            symbol: Optional symbol for symbol-specific data
            max_age_days: Maximum age of cached data in days
            
        Returns:
            Loaded data or None if not found/expired
        """
        cache_dir = self.drive_root / "data" / "cache"
        
        if symbol:
            pattern = f"{data_type}_{symbol}_*.pkl"
        else:
            pattern = f"{data_type}_*.pkl"
        
        cached_files = list(cache_dir.glob(pattern))
        
        if not cached_files:
            log.info(f"No cached data found for {data_type}")
            return None
        
        latest_file = max(cached_files, key=lambda p: p.stat().st_mtime)
        
        file_age_days = (datetime.now().timestamp() - latest_file.stat().st_mtime) / 86400
        
        if file_age_days > max_age_days:
            log.info(f"Cached data too old ({file_age_days:.1f} days), skipping")
            return None
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            
            log.info(f"✓ Loaded cached data: {latest_file.name} (age: {file_age_days:.1f} days)")
            return data
            
        except Exception as e:
            log.error(f"Failed to load cached data: {str(e)}")
            return None
    
    def save_historical_data(
        self,
        symbol: str,
        data,
        start_date: str,
        end_date: str
    ):
        """
        Save historical market data to Drive
        
        Args:
            symbol: Trading symbol
            data: Historical data (DataFrame)
            start_date: Start date of data
            end_date: End date of data
        """
        filename = f"{symbol}_{start_date}_{end_date}.pkl"
        data_path = self.drive_root / "data" / "historical" / filename
        
        try:
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            
            log.info(f"✓ Historical data saved: {symbol} ({start_date} to {end_date})")
            
        except Exception as e:
            log.error(f"Failed to save historical data: {str(e)}")
    
    def load_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[Any]:
        """
        Load historical market data from Drive
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Historical data or None if not found
        """
        filename = f"{symbol}_{start_date}_{end_date}.pkl"
        data_path = self.drive_root / "data" / "historical" / filename
        
        if not data_path.exists():
            log.info(f"No cached historical data for {symbol}")
            return None
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            log.info(f"✓ Loaded historical data: {symbol} ({start_date} to {end_date})")
            return data
            
        except Exception as e:
            log.error(f"Failed to load historical data: {str(e)}")
            return None
    
    def _update_model_registry(
        self,
        model_type: str,
        symbol: str,
        checkpoint_path: str,
        metadata: Optional[Dict]
    ):
        """
        Update model registry with new checkpoint
        
        Args:
            model_type: Type of model ('rl' or 'llm')
            symbol: Trading symbol
            checkpoint_path: Path to checkpoint
            metadata: Model metadata
        """
        registry_path = self.drive_root / "config" / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        if model_type not in registry:
            registry[model_type] = {}
        
        if symbol not in registry[model_type]:
            registry[model_type][symbol] = []
        
        entry = {
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        registry[model_type][symbol].append(entry)
        
        registry[model_type][symbol] = registry[model_type][symbol][-10:]
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def get_model_history(
        self,
        model_type: str,
        symbol: str
    ) -> List[Dict]:
        """
        Get checkpoint history for a model
        
        Args:
            model_type: Type of model ('rl' or 'llm')
            symbol: Trading symbol
            
        Returns:
            List of checkpoint entries
        """
        registry_path = self.drive_root / "config" / "model_registry.json"
        
        if not registry_path.exists():
            return []
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        return registry.get(model_type, {}).get(symbol, [])
    
    def save_performance_metrics(
        self,
        metrics: Dict,
        date: Optional[str] = None
    ):
        """
        Save performance metrics to Drive
        
        Args:
            metrics: Performance metrics dictionary
            date: Optional date string, defaults to today
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        metrics_path = self.drive_root / "results" / "performance" / f"metrics_{date}.json"
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            log.info(f"✓ Performance metrics saved: {date}")
            
        except Exception as e:
            log.error(f"Failed to save performance metrics: {str(e)}")
    
    def cleanup_old_checkpoints(
        self,
        model_type: str = 'rl',
        keep_latest: int = 5
    ):
        """
        Clean up old checkpoints, keeping only the latest N
        
        Args:
            model_type: Type of model ('rl' or 'llm')
            keep_latest: Number of latest checkpoints to keep per symbol
        """
        checkpoint_dir = self.drive_root / "models" / f"{model_type}_checkpoints"
        
        checkpoints_by_symbol = {}
        for checkpoint in checkpoint_dir.glob("*.zip"):
            parts = checkpoint.stem.split('_')
            if len(parts) >= 3:
                symbol = parts[2]
                if symbol not in checkpoints_by_symbol:
                    checkpoints_by_symbol[symbol] = []
                checkpoints_by_symbol[symbol].append(checkpoint)
        
        for symbol, checkpoints in checkpoints_by_symbol.items():
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for old_checkpoint in checkpoints[keep_latest:]:
                try:
                    old_checkpoint.unlink()
                    
                    metadata_file = old_checkpoint.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    log.info(f"Deleted old checkpoint: {old_checkpoint.name}")
                    
                except Exception as e:
                    log.error(f"Failed to delete checkpoint: {str(e)}")
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage stats
        """
        stats = {
            'rl_checkpoints': len(list((self.drive_root / "models" / "rl_checkpoints").glob("*.zip"))),
            'cached_data': len(list((self.drive_root / "data" / "cache").glob("*.pkl"))),
            'historical_data': len(list((self.drive_root / "data" / "historical").glob("*.pkl"))),
            'performance_logs': len(list((self.drive_root / "results" / "performance").glob("*.json")))
        }
        
        return stats
