#!/bin/bash

set -e

echo "=========================================="
echo "Autonomous Trading System Setup"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required!${NC}"
    echo -e "${RED}Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version OK${NC}"

if [ -z "$KAGGLE_KERNEL_RUN_TYPE" ]; then
    echo -e "\n${YELLOW}[2/6] Creating virtual environment...${NC}"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment already exists${NC}"
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "\n${YELLOW}[2/6] Kaggle environment detected - skipping venv${NC}"
fi

echo -e "\n${YELLOW}[3/6] Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

echo -e "\n${YELLOW}[4/6] Installing dependencies...${NC}"
echo "This may take a few minutes..."

if [ -f "requirements.txt" ]; then
    echo "Installing core dependencies..."
    python3 -m pip install -r requirements.txt --quiet || {
        echo -e "${YELLOW}Some packages failed, trying with fallbacks...${NC}"
        
        echo "Installing pandas-ta..."
        python3 -m pip install pandas-ta==0.3.14b0 --quiet || {
            echo -e "${YELLOW}pandas-ta 0.3.14b0 failed, trying alternative...${NC}"
            python3 -m pip install pandas-ta --quiet || echo -e "${YELLOW}pandas-ta installation failed (optional)${NC}"
        }
        
        python3 -m pip install numpy pandas yfinance pyyaml python-dotenv apscheduler flask loguru --quiet
    }
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[5/6] Creating directories...${NC}"
mkdir -p data logs backtest_results config
echo -e "${GREEN}✓ Directories created${NC}"

echo -e "\n${YELLOW}[6/6] Setting up configuration...${NC}"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
        echo -e "${YELLOW}⚠ Please update .env with your API keys!${NC}"
    else
        echo -e "${YELLOW}⚠ .env.example not found, creating default .env${NC}"
        cat > .env << EOF
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
TRADING_MODE=paper
INITIAL_CAPITAL=1000
LOG_LEVEL=INFO
KAGGLE_MODE=true
KEEP_ALIVE=true
EOF
        echo -e "${GREEN}✓ Created default .env${NC}"
        echo -e "${YELLOW}⚠ Please update .env with your API keys!${NC}"
    fi
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 -c "import pandas, numpy, yfinance, yaml, dotenv, flask, apscheduler; print('✓ All core modules imported successfully')" || {
    echo -e "${RED}Error: Some modules failed to import${NC}"
    exit 1
}

echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Update .env with your Alpaca API keys"
echo "2. Run the system:"
echo "   python3 launcher.py"
echo ""
echo "For Kaggle/Colab, see docs/KAGGLE_SETUP.md"
echo ""
