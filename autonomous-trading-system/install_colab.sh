#!/bin/bash

echo "=================================="
echo "Installing Trading System for Colab"
echo "=================================="

echo "Step 1: Removing conflicting packages..."
pip uninstall -y urllib3 requests websockets yfinance alpaca-trade-api 2>/dev/null

echo "Step 2: Installing core dependencies..."
pip install -q urllib3==1.26.18
pip install -q requests==2.31.0
pip install -q websockets==12.0

echo "Step 3: Installing market data APIs..."
pip install -q yfinance==0.2.40
pip install -q alpaca-trade-api==3.2.0
pip install -q alpaca-py==0.21.0
pip install -q alpha-vantage==2.3.1
pip install -q finnhub-python==2.4.25

echo "Step 4: Installing data processing libraries..."
pip install -q pandas==2.2.2
pip install -q numpy==1.26.4
pip install -q scipy==1.13.0

echo "Step 5: Installing technical analysis..."
pip install -q pandas-ta==0.3.14b0

echo "Step 6: Installing ML libraries..."
pip install -q scikit-learn==1.5.0
pip install -q xgboost==2.0.3

echo "Step 7: Installing PyTorch..."
pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Step 8: Installing transformers..."
pip install -q transformers==4.40.0
pip install -q accelerate==0.28.0
pip install -q bitsandbytes==0.43.0

echo "Step 9: Installing RL libraries..."
pip install -q stable-baselines3==2.2.1
pip install -q gymnasium==0.29.1

echo "Step 10: Installing utilities..."
pip install -q python-dotenv==1.0.1
pip install -q pyyaml==6.0.1
pip install -q loguru==0.7.2
pip install -q tenacity==8.2.3
pip install -q pydantic==2.5.0
pip install -q APScheduler==3.10.4
pip install -q beautifulsoup4==4.12.3
pip install -q pytz==2024.1

echo "Step 11: Downloading LLM models..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Mixtral-8x7B-Instruct (4-bit)...')
model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.float16
)
print('✓ Mixtral-8x7B downloaded and cached')
"

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Verifying installations..."
python3 -c "import urllib3; print(f'✓ urllib3: {urllib3.__version__}')"
python3 -c "import websockets; print(f'✓ websockets: {websockets.__version__}')"
python3 -c "import yfinance; print('✓ yfinance: OK')"
python3 -c "import alpaca_trade_api; print('✓ alpaca-trade-api: OK')"
python3 -c "from alpaca.data.historical import StockHistoricalDataClient; print('✓ alpaca-py: OK')"
python3 -c "import torch; print(f'✓ torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python3 -c "import transformers; print(f'✓ transformers: {transformers.__version__}')"
python3 -c "import apscheduler; print(f'✓ APScheduler: {apscheduler.__version__}')"
echo ""
echo "Ready to launch trading system!"
