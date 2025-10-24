#!/usr/bin/env python3
"""
Autonomous Trading System - Python Installer
Cross-platform installation script for Kaggle/Colab and local environments.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


class Installer:
    """Automated installer with error handling."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.errors = []
        self.warnings = []
    
    def print_header(self, text: str) -> None:
        """Print formatted header."""
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)
    
    def print_step(self, step: int, total: int, text: str) -> None:
        """Print step information."""
        print(f"\n[{step}/{total}] {text}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.errors.append(
                f"Python 3.10+ required. Current: {version.major}.{version.minor}.{version.micro}"
            )
            return False
        
        if version.major == 3 and version.minor > 12:
            self.warnings.append(
                f"Python {version.major}.{version.minor} is newer than tested. May have issues."
            )
        
        print("✓ Python version OK")
        return True
    
    def is_kaggle_environment(self) -> bool:
        """Check if running in Kaggle environment."""
        return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version."""
        try:
            print("Upgrading pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
                check=True,
                capture_output=True
            )
            print("✓ pip upgraded")
            return True
        except subprocess.CalledProcessError as e:
            self.warnings.append(f"Failed to upgrade pip: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        requirements_file = self.base_dir / "requirements.txt"
        
        if not requirements_file.exists():
            self.errors.append("requirements.txt not found!")
            return False
        
        print("Installing dependencies (this may take a few minutes)...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=False
            )
            print("✓ Dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print("⚠ Some packages failed, trying with fallbacks...")
            return self.install_with_fallbacks()
    
    def install_with_fallbacks(self) -> bool:
        """Install dependencies with fallback options."""
        core_packages = [
            "numpy>=1.24.0,<2.0.0",
            "pandas>=2.0.0",
            "yfinance>=0.2.28",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "apscheduler>=3.10.0",
            "flask>=3.0.0",
            "loguru>=0.7.0",
        ]
        
        print("Installing core packages...")
        for package in core_packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "--quiet"],
                    check=True,
                    capture_output=True
                )
                print(f"✓ {package.split('>=')[0]}")
            except subprocess.CalledProcessError:
                self.warnings.append(f"Failed to install {package}")
        
        print("\nInstalling pandas-ta...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pandas-ta>=0.4.71b0", "--quiet"],
                check=True,
                capture_output=True
            )
            print("✓ pandas-ta installed")
        except subprocess.CalledProcessError:
            print("⚠ pandas-ta >=0.4.71b0 failed, trying any version...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "pandas-ta", "--quiet"],
                    check=True,
                    capture_output=True
                )
                print("✓ pandas-ta installed (latest)")
            except subprocess.CalledProcessError:
                self.warnings.append("pandas-ta installation failed (optional)")
        
        print("\nChecking for CUDA (for auto-gptq)...")
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA detected, attempting auto-gptq installation...")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "auto-gptq==0.7.1", 
                         "--extra-index-url", "https://huggingface.github.io/autogptq-index/whl/cu118/", 
                         "--quiet"],
                        check=True,
                        capture_output=True,
                        timeout=300
                    )
                    print("✓ auto-gptq installed")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    self.warnings.append("auto-gptq installation failed (optional, requires CUDA)")
            else:
                print("⚠ No CUDA detected, skipping auto-gptq (optional)")
        except ImportError:
            print("⚠ torch not installed yet, skipping auto-gptq check")
        
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        directories = [
            "data",
            "logs",
            "backtest_results",
            "config"
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory}/")
        
        return True
    
    def setup_config_files(self) -> bool:
        """Setup configuration files."""
        env_file = self.base_dir / ".env"
        env_example = self.base_dir / ".env.example"
        
        if not env_file.exists():
            if env_example.exists():
                shutil.copy(env_example, env_file)
                print("✓ Created .env from .env.example")
            else:
                default_env = """ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
TRADING_MODE=paper
INITIAL_CAPITAL=1000
LOG_LEVEL=INFO
KAGGLE_MODE=true
KEEP_ALIVE=true
"""
                env_file.write_text(default_env)
                print("✓ Created default .env")
            
            self.warnings.append("Please update .env with your API keys!")
        else:
            print("✓ .env already exists")
        
        return True
    
    def verify_installation(self) -> bool:
        """Verify that key modules can be imported."""
        required_modules = [
            'pandas',
            'numpy',
            'yfinance',
            'yaml',
            'dotenv',
            'flask',
            'apscheduler',
        ]
        
        failed_modules = []
        
        for module in required_modules:
            try:
                if module == 'yaml':
                    __import__('yaml')
                elif module == 'dotenv':
                    __import__('dotenv')
                else:
                    __import__(module)
                print(f"✓ {module}")
            except ImportError:
                print(f"✗ {module}")
                failed_modules.append(module)
        
        try:
            import pandas_ta
            print("✓ pandas_ta (optional)")
        except ImportError:
            print("⚠ pandas_ta not available (optional)")
        
        if failed_modules:
            self.errors.append(f"Failed to import: {', '.join(failed_modules)}")
            return False
        
        return True
    
    def print_summary(self) -> None:
        """Print installation summary."""
        self.print_header("Installation Summary")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠ WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\n✓ Installation completed successfully!")
            print("\nNext steps:")
            print("1. Update .env with your Alpaca API keys")
            print("2. Run the system:")
            print("   python3 launcher.py")
            print("\nFor Kaggle/Colab, see docs/KAGGLE_SETUP.md")
        else:
            print("\n❌ Installation failed. Please fix the errors above.")
    
    def run(self) -> bool:
        """Run the installation process."""
        self.print_header("Autonomous Trading System - Installer")
        
        if self.is_kaggle_environment():
            print("Kaggle environment detected")
        
        self.print_step(1, 6, "Checking Python version...")
        if not self.check_python_version():
            self.print_summary()
            return False
        
        self.print_step(2, 6, "Upgrading pip...")
        self.upgrade_pip()
        
        self.print_step(3, 6, "Installing dependencies...")
        if not self.install_dependencies():
            self.print_summary()
            return False
        
        self.print_step(4, 6, "Creating directories...")
        self.create_directories()
        
        self.print_step(5, 6, "Setting up configuration...")
        self.setup_config_files()
        
        self.print_step(6, 6, "Verifying installation...")
        if not self.verify_installation():
            self.print_summary()
            return False
        
        self.print_summary()
        return len(self.errors) == 0


def main():
    """Main entry point."""
    installer = Installer()
    success = installer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
