#!/bin/bash


set -e  # Exit on error

echo "=========================================="
echo "  Autonomous Trading System Setup"
echo "=========================================="
echo ""

if [ ! -d "autonomous-trading-system" ]; then
    echo "❌ Error: autonomous-trading-system directory not found"
    echo "   Please run this script from the 'stuff' repository root"
    exit 1
fi

cd autonomous-trading-system
echo "✓ Navigated to autonomous-trading-system/"
echo ""

echo "Checking Python version..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found. Please install Python 3.10+"
    exit 1
fi

echo "✓ Found Python $PYTHON_VERSION"
echo ""

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

if [ ! -f ".env" ]; then
    echo "Setting up environment file..."
    cp .env.example .env
    echo "✓ .env file created from .env.example"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your API keys before running"
    echo "   (Optional: Alpha Vantage, Finnhub, Hugging Face)"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

echo "Creating necessary directories..."
mkdir -p logs data/trades data/models backups
echo "✓ Directories created"
echo ""

echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Run the system in set-and-forget mode:"
echo "   python launcher.py --set-and-forget --capital 100000"
echo ""
echo "3. Or run a quick backtest:"
echo "   python main.py backtest --symbols AAPL MSFT GOOGL"
echo ""
echo "4. Or launch the dashboard:"
echo "   python main.py dashboard"
echo ""
echo "For full documentation, see:"
echo "   README.md"
echo "   docs/ADVANCED_FEATURES.md"
echo ""
echo "=========================================="
echo ""

read -p "Launch system now in set-and-forget mode? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Launching autonomous trading system..."
    echo "Press Ctrl+C to stop"
    echo ""
    python launcher.py --set-and-forget --capital 100000
fi
