import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backtesting import BacktestEngine
from src.agents import DataAgent
from src.utils import config

st.set_page_config(
    page_title="Autonomous Trading System",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 Autonomous Trading System Dashboard")
st.markdown("**PAPER TRADING ONLY - No Real Money at Risk**")

@st.cache_data
def load_sample_data():
    data_agent = DataAgent()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    with st.spinner("Loading market data..."):
        data = data_agent.collect_market_data(symbols)
        processed_data = data_agent.process_data(data)
    
    return processed_data

def plot_equity_curve(equity_df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_df['Date'],
        y=equity_df['Equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00CC96', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_drawdown(equity_df):
    equity_series = equity_df['Equity']
    cumulative_max = equity_series.expanding().max()
    drawdown = (equity_series - cumulative_max) / cumulative_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_df['Date'],
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#EF553B', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_returns_distribution(equity_df):
    returns = equity_df['Equity'].pct_change().dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        marker=dict(color='#636EFA')
    ))
    
    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark'
    )
    
    return fig

def plot_positions(positions_df):
    if positions_df.empty:
        return None
    
    fig = px.bar(
        positions_df,
        x='symbol',
        y='unrealized_pnl',
        color='unrealized_pnl',
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Open Positions P&L'
    )
    
    fig.update_layout(template='plotly_dark')
    
    return fig

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Backtest", "💼 Portfolio", "⚙️ Settings"])

with tab1:
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Active", "Paper Trading")
    
    with col2:
        st.metric("Initial Capital", "$100,000", "")
    
    with col3:
        st.metric("Trading Mode", "Autopilot", "Multi-Agent")
    
    with col4:
        st.metric("Symbols Tracked", "20", "")
    
    st.markdown("---")
    
    st.subheader("System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Multi-Agent System:**
        - 🔍 Data Agent: Market data collection & processing
        - 🧠 Analysis Agent: LLM-based forecasting & alpha mining
        - 🎯 Decision Agent: RL-optimized trading decisions
        - 🔄 Optimization Agent: Continuous self-improvement
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - 📊 Technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - 📰 Sentiment analysis with FinBERT
        - 🧬 Genetic algorithm for strategy evolution
        - 🎲 Monte Carlo simulation for risk assessment
        """)

with tab2:
    st.header("Backtesting Engine")
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes"):
            try:
                data = load_sample_data()
                
                if not data:
                    st.error("No data available for backtesting")
                else:
                    backtest_engine = BacktestEngine()
                    result = backtest_engine.run_backtest(data)
                    
                    st.success("Backtest completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Return",
                            f"{result['total_return']:.2%}",
                            delta=f"${result['final_capital'] - result['initial_capital']:,.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{result['sharpe_ratio']:.2f}",
                            delta="Target: 1.5"
                        )
                    
                    with col3:
                        st.metric(
                            "Max Drawdown",
                            f"{result['max_drawdown']:.2%}",
                            delta=None
                        )
                    
                    with col4:
                        st.metric(
                            "Win Rate",
                            f"{result['win_rate']:.2%}",
                            delta=f"{result['num_trades']} trades"
                        )
                    
                    st.markdown("---")
                    
                    equity_df = result['equity_curve']
                    
                    st.plotly_chart(plot_equity_curve(equity_df), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(plot_drawdown(equity_df), use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(plot_returns_distribution(equity_df), use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("Trade History")
                    trade_history = result['trade_history']
                    if not trade_history.empty:
                        st.dataframe(trade_history.tail(20), use_container_width=True)
                    else:
                        st.info("No trades executed")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

with tab3:
    st.header("Portfolio Management")
    
    st.info("This tab would show real-time portfolio status in live mode")
    
    st.subheader("Current Positions")
    st.markdown("No open positions (demo mode)")
    
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Value", "$100,000", "0.00%")
    
    with col2:
        st.metric("Cash Available", "$100,000", "")
    
    with col3:
        st.metric("Open Positions", "0", "")

with tab4:
    st.header("System Configuration")
    
    st.subheader("Trading Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Initial Capital ($)", value=100000, step=10000)
        st.slider("Max Position Size (%)", 1, 20, 10)
        st.slider("Stop Loss (%)", 1, 10, 2)
    
    with col2:
        st.slider("Take Profit (%)", 1, 20, 5)
        st.number_input("Max Positions", value=10, step=1)
        st.selectbox("Position Sizing", ["Fixed", "Kelly Criterion", "Volatility-Based"])
    
    st.markdown("---")
    
    st.subheader("LLM Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Model Name", value="mistralai/Mistral-7B-Instruct-v0.2")
        st.slider("Temperature", 0.0, 1.0, 0.7)
    
    with col2:
        st.selectbox("Quantization", ["4bit", "8bit", "None"])
        st.slider("Top P", 0.0, 1.0, 0.9)
    
    st.markdown("---")
    
    st.subheader("Safety Settings")
    
    st.checkbox("Paper Trading Only", value=True, disabled=True)
    st.checkbox("Circuit Breaker", value=True)
    st.slider("Max Daily Loss (%)", 1, 20, 5)
    
    if st.button("Save Configuration"):
        st.success("Configuration saved!")

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

st.sidebar.subheader("Quick Stats")
st.sidebar.metric("System Status", "🟢 Online")
st.sidebar.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

st.sidebar.markdown("---")

st.sidebar.subheader("About")
st.sidebar.info("""
This is an autonomous trading system powered by:
- LLM-based decision making
- Reinforcement learning (PPO)
- Alpha mining & genetic algorithms
- Multi-agent architecture

**DISCLAIMER:** Paper trading only. No real money at risk.
""")
