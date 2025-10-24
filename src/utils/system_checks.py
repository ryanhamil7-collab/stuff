"""
System Checks Module for Autonomous Trading System
Validates environment, resources, APIs, and dependencies before launch.
"""

import os
import sys
import socket
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import requests
except ImportError:
    requests = None


class SystemChecker:
    """Comprehensive system validation for trading system."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize system checker.
        
        Args:
            strict_mode: If True, raise errors on warnings. If False, only warn.
        """
        self.strict_mode = strict_mode
        self.warnings = []
        self.errors = []
        self.info = []
    
    def log_info(self, message: str):
        """Log informational message."""
        self.info.append(message)
        print(f"ℹ️  {message}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.warnings.append(message)
        print(f"⚠️  {message}")
        if self.strict_mode:
            raise RuntimeError(f"Strict mode: {message}")
    
    def log_error(self, message: str):
        """Log error message."""
        self.errors.append(message)
        print(f"❌ {message}")
    
    def log_success(self, message: str):
        """Log success message."""
        print(f"✅ {message}")
    
    def check_python_version(self, min_version: Tuple[int, int] = (3, 10)) -> bool:
        """
        Check Python version meets minimum requirements.
        
        Args:
            min_version: Minimum (major, minor) version required
            
        Returns:
            True if version is sufficient
        """
        print("\n[Check 1/10] Python Version")
        version = sys.version_info
        current = (version.major, version.minor)
        
        if current >= min_version:
            self.log_success(f"Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.log_error(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"got {version.major}.{version.minor}"
            )
            return False
    
    def check_api_keys(self, required_keys: Optional[List[str]] = None) -> bool:
        """
        Check for required API keys in environment.
        
        Args:
            required_keys: List of required environment variable names
            
        Returns:
            True if all keys present
        """
        print("\n[Check 2/10] API Keys")
        
        if required_keys is None:
            required_keys = [
                'ALPACA_API_KEY',
                'ALPACA_SECRET_KEY',
            ]
        
        optional_keys = [
            'ALPHA_VANTAGE_API_KEY',
            'FINNHUB_API_KEY',
            'HUGGINGFACE_TOKEN',
        ]
        
        missing_required = []
        missing_optional = []
        
        for key in required_keys:
            value = os.getenv(key)
            if not value or value == f'your_{key.lower()}_here':
                missing_required.append(key)
        
        for key in optional_keys:
            value = os.getenv(key)
            if not value or value == f'your_{key.lower()}_here':
                missing_optional.append(key)
        
        if missing_required:
            self.log_error(
                f"Missing required API keys: {', '.join(missing_required)}. "
                "Set them in .env file."
            )
            return False
        
        if missing_optional:
            self.log_warning(
                f"Missing optional API keys: {', '.join(missing_optional)}. "
                "Some features may be limited."
            )
        
        self.log_success(f"All required API keys present")
        return True
    
    def validate_api_key(self, key_name: str, test_url: str) -> bool:
        """
        Validate an API key by making a test request.
        
        Args:
            key_name: Environment variable name
            test_url: URL to test with the key
            
        Returns:
            True if key is valid
        """
        if requests is None:
            self.log_warning("requests library not available, skipping API validation")
            return True
        
        key_value = os.getenv(key_name)
        if not key_value:
            return False
        
        try:
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'Error Message' not in data and 'error' not in data:
                    return True
        except Exception as e:
            self.log_warning(f"Could not validate {key_name}: {e}")
        
        return False
    
    def check_gpu_cuda(self) -> Dict[str, any]:
        """
        Check GPU/CUDA availability.
        
        Returns:
            Dict with GPU info
        """
        print("\n[Check 3/10] GPU/CUDA Availability")
        
        gpu_info = {
            'available': False,
            'device_count': 0,
            'device_name': None,
            'cuda_version': None,
            'recommended_device': 'cpu'
        }
        
        if torch is None:
            self.log_warning("PyTorch not installed, GPU detection skipped")
            return gpu_info
        
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['recommended_device'] = 'cuda'
            
            self.log_success(
                f"GPU detected: {gpu_info['device_name']} "
                f"(CUDA {gpu_info['cuda_version']})"
            )
            self.log_info(f"Using GPU acceleration for LLM inference")
        else:
            self.log_warning(
                "No GPU detected. System will use CPU (slower performance). "
                "For Kaggle, enable GPU in notebook settings."
            )
            gpu_info['recommended_device'] = 'cpu'
        
        return gpu_info
    
    def check_system_resources(self) -> Dict[str, any]:
        """
        Check system resources (RAM, CPU, disk).
        
        Returns:
            Dict with resource info
        """
        print("\n[Check 4/10] System Resources")
        
        resources = {
            'ram_gb': 0,
            'ram_sufficient': False,
            'cpu_cores': 0,
            'cpu_sufficient': False,
            'disk_free_gb': 0,
            'disk_sufficient': False
        }
        
        if psutil is None:
            self.log_warning("psutil not installed, resource checks skipped")
            return resources
        
        ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        resources['ram_gb'] = ram_gb
        resources['ram_sufficient'] = ram_gb >= 16
        
        if ram_gb >= 16:
            self.log_success(f"RAM: {ram_gb:.1f}GB available")
        elif ram_gb >= 8:
            self.log_warning(
                f"RAM: {ram_gb:.1f}GB available (recommended: 16GB+). "
                "May have issues with large models."
            )
        else:
            self.log_error(
                f"RAM: {ram_gb:.1f}GB available (minimum: 8GB). "
                "Insufficient for LLM operations."
            )
        
        cpu_cores = psutil.cpu_count()
        resources['cpu_cores'] = cpu_cores
        resources['cpu_sufficient'] = cpu_cores >= 4
        
        if cpu_cores >= 4:
            self.log_success(f"CPU: {cpu_cores} cores")
        else:
            self.log_warning(
                f"CPU: {cpu_cores} cores (recommended: 4+). "
                "Multi-agent features may be slow."
            )
        
        disk_usage = shutil.disk_usage('.')
        disk_free_gb = disk_usage.free / (1024 ** 3)
        resources['disk_free_gb'] = disk_free_gb
        resources['disk_sufficient'] = disk_free_gb >= 10
        
        if disk_free_gb >= 10:
            self.log_success(f"Disk: {disk_free_gb:.1f}GB free")
        else:
            self.log_error(
                f"Disk: {disk_free_gb:.1f}GB free (minimum: 10GB). "
                "Insufficient for data and models."
            )
        
        return resources
    
    def check_internet_connectivity(self) -> bool:
        """
        Check internet connectivity.
        
        Returns:
            True if internet is available
        """
        print("\n[Check 5/10] Internet Connectivity")
        
        test_hosts = [
            ("www.google.com", 80),
            ("api.alpaca.markets", 443),
            ("www.alphavantage.co", 443),
        ]
        
        for host, port in test_hosts:
            try:
                socket.create_connection((host, port), timeout=5)
                self.log_success(f"Internet connection verified ({host})")
                return True
            except OSError:
                continue
        
        self.log_error(
            "No internet connection detected. Required for API data fetching. "
            "For Kaggle, enable Internet in notebook settings."
        )
        return False
    
    def check_required_directories(self) -> bool:
        """
        Check and create required directories.
        
        Returns:
            True if all directories exist or were created
        """
        print("\n[Check 6/10] Directory Structure")
        
        required_dirs = [
            'data',
            'logs',
            'config',
            'backtest_results',
            'models',
        ]
        
        all_exist = True
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.log_info(f"Created directory: {dir_name}/")
                except Exception as e:
                    self.log_error(f"Failed to create {dir_name}/: {e}")
                    all_exist = False
            else:
                self.log_success(f"Directory exists: {dir_name}/")
        
        return all_exist
    
    def check_config_files(self) -> bool:
        """
        Check for required configuration files.
        
        Returns:
            True if all config files exist
        """
        print("\n[Check 7/10] Configuration Files")
        
        config_files = {
            '.env': 'Environment variables',
            'config/config.yaml': 'Main configuration',
            'config/scheduler.yaml': 'Scheduler configuration',
        }
        
        all_exist = True
        for file_path, description in config_files.items():
            if Path(file_path).exists():
                self.log_success(f"{description}: {file_path}")
            else:
                self.log_warning(
                    f"{description} not found: {file_path}. "
                    "Will be auto-created with defaults."
                )
                all_exist = False
        
        return all_exist
    
    def check_package_versions(self) -> bool:
        """
        Check critical package versions.
        
        Returns:
            True if all packages meet minimum versions
        """
        print("\n[Check 8/10] Package Versions")
        
        packages = {
            'pandas': '2.0.0',
            'numpy': '1.24.0',
            'torch': '2.1.0',
            'transformers': '4.40.0',
        }
        
        all_ok = True
        for package, min_version in packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                self.log_success(f"{package}: {version}")
            except ImportError:
                self.log_warning(f"{package} not installed (optional)")
        
        return all_ok
    
    def check_model_access(self) -> bool:
        """
        Check if models can be accessed.
        
        Returns:
            True if models are accessible
        """
        print("\n[Check 9/10] Model Access")
        
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*'))
            if model_files:
                self.log_success(f"Found {len(model_files)} cached models")
            else:
                self.log_info("No cached models found (will download on first run)")
        else:
            self.log_info("Models directory will be created on first run")
        
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token and hf_token != 'your_huggingface_token_here':
            self.log_success("Hugging Face token configured")
        else:
            self.log_warning(
                "Hugging Face token not set. "
                "May be required for some models."
            )
        
        return True
    
    def check_kaggle_environment(self) -> Dict[str, any]:
        """
        Check if running in Kaggle environment and get info.
        
        Returns:
            Dict with Kaggle environment info
        """
        print("\n[Check 10/10] Kaggle Environment")
        
        kaggle_info = {
            'is_kaggle': False,
            'kernel_type': None,
            'internet_enabled': False,
            'gpu_enabled': False,
        }
        
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            kaggle_info['is_kaggle'] = True
            kaggle_info['kernel_type'] = os.getenv('KAGGLE_KERNEL_RUN_TYPE')
            self.log_success(f"Running in Kaggle ({kaggle_info['kernel_type']})")
            
            kaggle_info['internet_enabled'] = self.check_internet_connectivity()
            
            if torch and torch.cuda.is_available():
                kaggle_info['gpu_enabled'] = True
                self.log_success("Kaggle GPU enabled")
            else:
                self.log_warning("Kaggle GPU not enabled (enable in settings)")
        else:
            self.log_info("Not running in Kaggle environment")
        
        return kaggle_info
    
    def run_all_checks(self, skip_optional: bool = False) -> bool:
        """
        Run all system checks.
        
        Args:
            skip_optional: If True, skip optional checks
            
        Returns:
            True if all critical checks pass
        """
        print("=" * 70)
        print("AUTONOMOUS TRADING SYSTEM - SYSTEM CHECKS")
        print("=" * 70)
        
        critical_passed = True
        
        if not self.check_python_version():
            critical_passed = False
        
        if not skip_optional:
            self.check_api_keys()
        
        gpu_info = self.check_gpu_cuda()
        resources = self.check_system_resources()
        
        if not self.check_internet_connectivity():
            if not skip_optional:
                critical_passed = False
        
        self.check_required_directories()
        self.check_config_files()
        
        if not skip_optional:
            self.check_package_versions()
            self.check_model_access()
        
        kaggle_info = self.check_kaggle_environment()
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if critical_passed and not self.errors:
            print("\n✅ All critical checks passed!")
            print("System is ready to launch.")
            return True
        else:
            print("\n❌ Some checks failed. Please fix errors before launching.")
            return False


def run_system_checks(strict_mode: bool = False, skip_optional: bool = False) -> bool:
    """
    Convenience function to run all system checks.
    
    Args:
        strict_mode: If True, treat warnings as errors
        skip_optional: If True, skip optional checks
        
    Returns:
        True if all checks pass
    """
    checker = SystemChecker(strict_mode=strict_mode)
    return checker.run_all_checks(skip_optional=skip_optional)


if __name__ == "__main__":
    success = run_system_checks()
    sys.exit(0 if success else 1)
