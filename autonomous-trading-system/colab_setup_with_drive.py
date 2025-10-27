"""
Google Colab Setup Script with Drive Integration

This script sets up the autonomous trading system in Colab with
persistent storage in Google Drive.

Features:
- Automatic Drive mounting
- Model checkpoint persistence
- Training data caching
- Configuration management
"""

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup Colab environment with Drive integration"""
    
    print("=" * 80)
    print("AUTONOMOUS TRADING SYSTEM - COLAB SETUP WITH DRIVE")
    print("=" * 80)
    
    print("\n[1/6] Mounting Google Drive...")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive', force_remount=False)
            print("✓ Google Drive mounted successfully")
        else:
            print("✓ Google Drive already mounted")
    except Exception as e:
        print(f"⚠ Warning: Could not mount Drive: {e}")
        print("  Continuing without Drive persistence...")
    
    print("\n[2/6] Setting up repository...")
    repo_path = Path('/content/stuff/autonomous-trading-system')
    
    if not repo_path.exists():
        print("Cloning repository...")
        os.system('git clone https://github.com/ryanhamil7-collab/stuff.git /content/stuff')
    else:
        print("Repository already exists, pulling latest changes...")
        os.chdir(repo_path)
        os.system('git pull origin devin/1761207403-autonomous-trading-system')
    
    os.chdir(repo_path)
    print(f"✓ Working directory: {os.getcwd()}")
    
    print("\n[3/6] Checking out latest branch...")
    os.system('git checkout devin/1761207403-autonomous-trading-system')
    print("✓ Branch checked out")
    
    print("\n[4/6] Installing dependencies...")
    print("This may take 5-10 minutes...")
    result = os.system('bash install_colab.sh')
    if result == 0:
        print("✓ Dependencies installed successfully")
    else:
        print("⚠ Warning: Some dependencies may have failed to install")
    
    print("\n[5/6] Setting up API credentials...")
    
    env_file = repo_path / '.env'
    if env_file.exists():
        print("✓ .env file already exists")
    else:
        print("\nCreating .env file...")
        print("You can update these values in the .env file after setup")
        
        env_content = """# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here

HUGGINGFACE_TOKEN=your_huggingface_token_here
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✓ .env file created")
        print("\n⚠ IMPORTANT: Update .env with your actual API keys!")
    
    print("\n[6/6] Initializing Drive storage...")
    try:
        sys.path.insert(0, str(repo_path))
        from src.utils.drive_storage import DriveStorageManager
        
        storage = DriveStorageManager()
        stats = storage.get_storage_stats()
        
        print("✓ Drive storage initialized")
        print(f"\nStorage Statistics:")
        print(f"  RL Checkpoints: {stats['rl_checkpoints']}")
        print(f"  Cached Data: {stats['cached_data']}")
        print(f"  Historical Data: {stats['historical_data']}")
        print(f"  Performance Logs: {stats['performance_logs']}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Drive storage: {e}")
        print("  System will work but without persistence")
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Run: python3 colab_live_trading.py")
    print("\nDrive Integration:")
    print("- Models will be saved to: /content/drive/MyDrive/autonomous_trading/models/")
    print("- Data will be cached in: /content/drive/MyDrive/autonomous_trading/data/")
    print("- Survives runtime disconnects!")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    setup_colab_environment()
