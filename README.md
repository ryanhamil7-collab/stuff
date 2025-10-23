# 🤖 Autonomous Trading System

A comprehensive, autonomous stock trading system powered by fine-tuned LLMs, reinforcement learning, and multi-agent architecture. The system operates on autopilot to maximize profitability through intelligent trading decisions, continuous self-improvement, and proactive symbol discovery.

**⚠️ PAPER TRADING ONLY - No Real Money at Risk**

## 🌟 Features

### Core Capabilities
- **Multi-Agent Architecture**: Specialized agents for data collection, analysis, decision-making, and optimization
- **LLM-Powered Decision Making**: Uses Mistral-7B or Llama-3-8B for intelligent trading decisions
- **Reinforcement Learning**: PPO-based policy optimization for adaptive trading strategies
- **Alpha Mining**: Generates and evaluates formulaic alpha factors with RankIC, ICIR metrics
- **Sentiment Analysis**: FinBERT-powered news sentiment analysis
- **Symbol Discovery**: Automated discovery of profitable trading opportunities using clustering and ML
- **Genetic Algorithms**: Evolutionary strategy optimization and parameter tuning
- **Risk Management**: Stop-loss, position sizing (Kelly Criterion), diversification, circuit breakers
- **Comprehensive Backtesting**: Historical testing with Monte Carlo simulations
- **Real-time Dashboard**: Streamlit-based visualization and monitoring

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, ADX, Stochastic Oscillator
- OBV, VWAP
- Market regime detection (bull/bear/sideways)

### Advanced Features
- **Chain-of-Alpha**: Dual-chain LLM for alpha mining and optimization
- **Multi-objective Optimization**: Balancing returns and risk
- **Concept Drift Detection**: Adapting to market regime shifts
- **RAG-Enhanced Prompting**: Knowledge base with SQLite/FAISS
- **LoRA Fine-tuning**: Efficient LLM adaptation to trading domain

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for faster LLM inference)
- 16GB+ RAM recommended
- API keys for:
  - Alpha Vantage (optional)
  - Finnhub (optional)
  - Hugging Face (for LLM models)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd autonomous-trading-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the System

#### 1. Backtest Mode
Run historical backtesting on selected symbols:

```bash
python main.py backtest
```

With custom symbols and date range:
```bash
python main.py backtest --symbols AAPL MSFT GOOGL NVDA TSLA --start-date 2020-01-01 --end-date 2024-12-31
```

#### 2. Autopilot Mode
Run the system continuously with automated trading:

```bash
python main.py autopilot
```

The system will:
- Monitor markets every 5 minutes during trading hours
- Analyze symbols using multi-agent architecture
- Execute simulated trades based on LLM + RL decisions
- Continuously adapt and improve strategies

#### 3. Dashboard Mode
Launch the interactive web dashboard:

```bash
python main.py dashboard
```

Then open http://localhost:8501 in your browser.

#### 4. Single Cycle Mode
Run one trading cycle for testing:

```bash
python main.py single
```

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                   Autonomous Trading System                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  Data   │          │Analysis │          │Decision │
   │  Agent  │─────────▶│  Agent  │─────────▶│  Agent  │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        │                     │                     │
   ┌────▼─────────────────────▼─────────────────────▼────┐
   │              Optimization Agent                      │
   │         (Self-improvement & Strategy Evolution)      │
   └──────────────────────────────────────────────────────┘
```

### Component Overview

1. **Data Agent**
   - Fetches market data from yfinance, Alpha Vantage, Finnhub
   - Calculates technical indicators
   - Processes and cleans data
   - Discovers new trading symbols

2. **Analysis Agent**
   - LLM-based market analysis and forecasting
   - Alpha factor mining and evaluation
   - Sentiment analysis on news
   - Market regime detection

3. **Decision Agent**
   - RL-optimized trading decisions (PPO)
   - Risk-adjusted position sizing
   - Trade execution with validation
   - Stop-loss and take-profit management

4. **Optimization Agent**
   - Genetic algorithm for strategy evolution
   - Performance feedback loop
   - LLM fine-tuning with LoRA
   - Knowledge base updates

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, annualized return, daily returns
- **Risk-Adjusted**: Sharpe ratio (target >1.5), Sortino ratio, Calmar ratio
- **Risk**: Maximum drawdown, volatility, VaR
- **Trading**: Win rate, profit factor, average win/loss
- **Alpha**: RankIC, ICIR, turnover, diversity

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Trading Parameters
```yaml
trading:
  initial_capital: 100000.0
  max_position_size: 0.10
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
  max_positions: 10
```

### LLM Configuration
```yaml
llm:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  quantization: "4bit"
  temperature: 0.7
```

### Reinforcement Learning
```yaml
rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
  target_sharpe: 1.5
```

### Risk Management
```yaml
risk:
  max_drawdown: 0.20
  position_sizing: "kelly"
  kelly_fraction: 0.25
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
docker build -t autonomous-trading-system .
docker run -p 8501:8501 -v $(pwd)/data:/app/data autonomous-trading-system
```

## 📁 Project Structure

```
autonomous-trading-system/
├── src/
│   ├── agents/              # Multi-agent system
│   │   ├── data_agent.py
│   │   ├── analysis_agent.py
│   │   ├── decision_agent.py
│   │   └── symbol_discovery.py
│   ├── models/              # ML/LLM models
│   │   ├── llm_trader.py
│   │   └── sentiment_analyzer.py
│   ├── strategies/          # Trading strategies
│   │   ├── alpha_mining.py
│   │   ├── risk_management.py
│   │   └── portfolio.py
│   ├── data_pipeline/       # Data fetching & processing
│   │   └── data_fetcher.py
│   ├── backtesting/         # Backtesting engine
│   │   └── backtest_engine.py
│   ├── dashboard/           # Streamlit dashboard
│   │   └── app.py
│   ├── utils/               # Utilities
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── indicators.py
│   └── autopilot.py         # Autopilot daemon
├── config/
│   └── config.yaml          # Configuration file
├── data/                    # Data storage
├── logs/                    # Log files
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## 📈 Example Results

Sample backtest results (2020-2024):

```
Initial Capital: $100,000.00
Final Capital: $156,234.50
Total Return: 56.23%
Sharpe Ratio: 1.87
Sortino Ratio: 2.34
Max Drawdown: -12.45%
Win Rate: 58.3%
Total Trades: 247
Profit Factor: 1.92
```

## 🔒 Safety Features

- **Paper Trading Only**: All trades are simulated
- **Circuit Breakers**: Automatic shutdown on excessive losses
- **Position Limits**: Maximum position size and number of positions
- **Risk Controls**: Stop-loss, take-profit, correlation checks
- **Daily Loss Limits**: Maximum daily loss threshold
- **Trading Hours**: Only operates during market hours

## 🤝 Contributing

This is a demonstration project. For production use:

1. Implement proper API authentication
2. Add comprehensive error handling
3. Implement database persistence
4. Add monitoring and alerting
5. Conduct thorough backtesting
6. Implement paper trading validation
7. Add regulatory compliance checks

## ⚖️ Legal Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- This system performs PAPER TRADING ONLY with simulated capital
- No real money is at risk
- Not financial advice
- Not suitable for live trading without extensive modifications
- Use at your own risk
- Past performance does not guarantee future results
- The authors assume no liability for financial losses

## 📚 References

This system draws inspiration from:

- Chain-of-Alpha: Dual-chain LLM for alpha mining
- ElliottAgents: Multi-agent LLM for forecasting
- FLAG-Trader: LLM fused with RL via PPO
- Stock-Evol-Instruct: LLM-guided RL with dynamic instructions
- FinBERT: Financial sentiment analysis
- QuantEvolve: Hypothesis generation framework

## 📝 License

MIT License - See LICENSE file for details

## 🙋 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the configuration in `config/config.yaml`

## 🎯 Roadmap

- [ ] Add more data sources (Polygon, IEX Cloud)
- [ ] Implement options trading strategies
- [ ] Add cryptocurrency support
- [ ] Enhance LLM fine-tuning pipeline
- [ ] Add more genetic algorithm operators
- [ ] Implement ensemble models
- [ ] Add real-time news scraping
- [ ] Enhance dashboard with more visualizations
- [ ] Add mobile app support
- [ ] Implement distributed backtesting

---

**Built with ❤️ for algorithmic trading research**
