from __future__ import annotations

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    add_analysis_columns,
    calculate_annualized_volatility,
    calculate_average_daily_return,
    calculate_benchmark_correlation,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    classify_risk,
    download_market_data,
    normalize_performance,
)


st.set_page_config(
    page_title="Financial Market Volatility & Risk Dashboard",
    page_icon="📈",
    layout="wide",
)


BENCHMARKS = {
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Dow Jones Industrial Average": "^DJI",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Cache Yahoo Finance requests for one hour."""
    return download_market_data(ticker, start_date, end_date)


def format_percent(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def format_price(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def build_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            mode="lines",
            name="20-Day MA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            mode="lines",
            name="50-Day MA",
        )
    )

    fig.update_layout(
        title=f"{ticker} Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Adjusted Price (USD)",
        hovermode="x unified",
        legend_title_text="Series",
    )
    return fig


def build_daily_returns_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=data.index,
            y=data["Daily Return"] * 100,
            name="Daily Return",
        )
    )
    fig.update_layout(
        title=f"{ticker} Daily Returns",
        xaxis_title="Date",
        yaxis_title="Daily Return (%)",
        hovermode="x unified",
    )
    return fig


def build_rolling_volatility_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=data.index,
            y=data["Rolling 30D Volatility"] * 100,
            mode="lines",
            name="30-Day Rolling Volatility",
        )
    )
    fig.update_layout(
        title=f"{ticker} Rolling 30-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        hovermode="x unified",
    )
    return fig


def build_drawdown_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=data.index,
            y=data["Drawdown"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title=f"{ticker} Drawdown From Running Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    return fig


def build_benchmark_chart(
    stock_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    ticker: str,
    benchmark_name: str,
) -> go.Figure:
    stock_norm = normalize_performance(stock_data["Close"])
    benchmark_norm = normalize_performance(benchmark_data["Close"])

    comparison = pd.concat(
        [
            stock_norm.rename(ticker),
            benchmark_norm.rename(benchmark_name),
        ],
        axis=1,
    ).dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison.index,
            y=comparison[ticker],
            mode="lines",
            name=ticker,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison.index,
            y=comparison[benchmark_name],
            mode="lines",
            name=benchmark_name,
        )
    )

    fig.update_layout(
        title=f"Growth of $100: {ticker} vs {benchmark_name}",
        xaxis_title="Date",
        yaxis_title="Value of $100",
        hovermode="x unified",
    )
    return fig


def build_var_histogram(
    daily_returns: pd.Series,
    var_95: float,
    ticker: str,
):
    clean = daily_returns.dropna() * 100
    var_threshold = -var_95 * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(clean, bins=40, alpha=0.8)
    ax.axvline(
        var_threshold,
        linestyle="--",
        linewidth=2,
        label=f"95% VaR threshold: {var_threshold:.2f}%",
    )
    ax.set_title(f"{ticker} Daily Return Distribution")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


st.title("📈 Financial Market Volatility & Risk Dashboard")
st.write(
    "Analyze a stock's historical performance, volatility, downside risk, and "
    "relationship to a major market benchmark using live Yahoo Finance data."
)

with st.sidebar:
    st.header("Analysis Controls")

    ticker = st.text_input(
        "Stock ticker",
        value="AAPL",
        help="Examples: AAPL, NVDA, DIS, NFLX, MSFT",
    ).strip().upper()

    today = date.today()
    default_start = today - timedelta(days=365 * 2)

    start_date = st.date_input(
        "Start date",
        value=default_start,
        max_value=today,
    )
    end_date = st.date_input(
        "End date",
        value=today,
        max_value=today,
    )

    benchmark_name = st.selectbox(
        "Benchmark",
        options=list(BENCHMARKS.keys()),
        index=0,
    )

    with st.expander("Advanced setting"):
        risk_free_rate_pct = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.25,
            help="Used only for the Sharpe ratio calculation.",
        )

    analyze = st.button("Analyze Market Risk", type="primary", width="stretch")

st.caption(
    "Educational portfolio project only. The risk metrics and classifications "
    "shown here are not financial advice."
)

if not analyze:
    st.info(
        "Choose a ticker, date range, and benchmark in the sidebar, then select "
        "**Analyze Market Risk**."
    )
    st.stop()

if not ticker:
    st.error("Enter a stock ticker before running the analysis.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()

benchmark_ticker = BENCHMARKS[benchmark_name]
risk_free_rate = risk_free_rate_pct / 100

with st.spinner("Downloading market data and calculating risk metrics..."):
    try:
        raw_stock = load_data(ticker, start_date, end_date)
        raw_benchmark = load_data(benchmark_ticker, start_date, end_date)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(
            "The market-data request failed unexpectedly. Please check the "
            "ticker and try again."
        )
        st.exception(exc)
        st.stop()

if len(raw_stock) < 60:
    st.error(
        "This date range does not contain enough trading days for the 50-day "
        "moving average and rolling-risk analysis. Choose a longer date range."
    )
    st.stop()

stock = add_analysis_columns(raw_stock)
benchmark = add_analysis_columns(raw_benchmark)

daily_returns = stock["Daily Return"]
benchmark_returns = benchmark["Daily Return"]

total_return = calculate_total_return(stock["Close"])
average_daily_return = calculate_average_daily_return(daily_returns)
annualized_volatility = calculate_annualized_volatility(daily_returns)
sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
max_drawdown = calculate_max_drawdown(stock["Close"])
var_95 = calculate_historical_var(daily_returns, confidence_level=0.95)
benchmark_corr = calculate_benchmark_correlation(
    daily_returns,
    benchmark_returns,
)
risk_label, risk_reason = classify_risk(annualized_volatility)

starting_price = float(stock["Close"].dropna().iloc[0])
ending_price = float(stock["Close"].dropna().iloc[-1])

st.subheader(f"{ticker} Market Overview")

row1 = st.columns(4)
row1[0].metric("Starting Price", format_price(starting_price))
row1[1].metric("Ending Price", format_price(ending_price))
row1[2].metric("Total Return", format_percent(total_return))
row1[3].metric(
    "Average Daily Return",
    format_percent(average_daily_return, decimals=3),
)

row2 = st.columns(4)
row2[0].metric("Annualized Volatility", format_percent(annualized_volatility))
row2[1].metric("Sharpe Ratio", format_number(sharpe_ratio))
row2[2].metric("Maximum Drawdown", format_percent(max_drawdown))
row2[3].metric("95% Historical VaR", format_percent(var_95))

st.markdown(f"### Risk Classification: **{risk_label}**")
st.write(
    f"{risk_reason} This is a simplified educational classification based on "
    "annualized volatility, not a recommendation to buy or sell a security."
)

overview_tab, risk_tab, benchmark_tab, data_tab = st.tabs(
    ["Performance", "Risk Analysis", "Benchmark Comparison", "Data & Export"]
)

with overview_tab:
    st.plotly_chart(
        build_price_chart(stock, ticker),
        width="stretch",
        key="price_chart",
    )

    st.plotly_chart(
        build_daily_returns_chart(stock, ticker),
        width="stretch",
        key="daily_returns_chart",
    )

    with st.expander("How to read this section"):
        st.write(
            "**Moving averages** smooth daily price changes and make longer-term "
            "trends easier to see. **Daily returns** measure the percentage "
            "change from one trading day to the next."
        )

with risk_tab:
    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            build_rolling_volatility_chart(stock, ticker),
            width="stretch",
            key="rolling_vol_chart",
        )

    with right:
        st.plotly_chart(
            build_drawdown_chart(stock, ticker),
            width="stretch",
            key="drawdown_chart",
        )

    st.markdown("#### Historical Value at Risk")
    var_fig = build_var_histogram(daily_returns, var_95, ticker)
    st.pyplot(var_fig, width="stretch")
    plt.close(var_fig)

    st.write(
        f"The 95% historical one-day VaR is **{format_percent(var_95)}**. "
        "Using the historical return distribution, this means approximately "
        "5% of observed trading days had losses worse than this threshold."
    )

with benchmark_tab:
    st.metric(
        f"Return Correlation With {benchmark_name}",
        format_number(benchmark_corr, decimals=3),
    )

    st.plotly_chart(
        build_benchmark_chart(
            stock,
            benchmark,
            ticker,
            benchmark_name,
        ),
        width="stretch",
        key="benchmark_chart",
    )

    if pd.notna(benchmark_corr):
        if benchmark_corr >= 0.7:
            relationship = "strong positive"
        elif benchmark_corr >= 0.3:
            relationship = "moderate positive"
        elif benchmark_corr > -0.3:
            relationship = "weak"
        elif benchmark_corr > -0.7:
            relationship = "moderate negative"
        else:
            relationship = "strong negative"

        st.write(
            f"Over the selected period, {ticker} had a **{relationship}** "
            f"daily-return relationship with {benchmark_name} "
            f"(correlation = {benchmark_corr:.3f})."
        )

with data_tab:
    display_columns = [
        column
        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Daily Return",
            "MA20",
            "MA50",
            "Rolling 30D Volatility",
            "Cumulative Return",
            "Drawdown",
        ]
        if column in stock.columns
    ]

    export_data = stock[display_columns].copy()
    export_data.index.name = "Date"

    st.dataframe(
        export_data.tail(100),
        width="stretch",
    )

    csv_data = export_data.to_csv().encode("utf-8")

    st.download_button(
        label=f"Download {ticker} Analysis CSV",
        data=csv_data,
        file_name=f"{ticker.lower()}_risk_analysis.csv",
        mime="text/csv",
        width="stretch",
    )

with st.expander("Metric Guide"):
    st.markdown(
        """
- **Total Return:** Percentage change from the first closing price to the last.
- **Average Daily Return:** Mean percentage return across trading days.
- **Annualized Volatility:** Daily return variability scaled to a 252-trading-day year.
- **Sharpe Ratio:** Annualized excess return per unit of volatility using the risk-free rate you selected.
- **Maximum Drawdown:** Largest decline from a previous peak during the selected period.
- **95% Historical VaR:** A historical estimate of a one-day loss threshold exceeded on roughly 5% of observed days.
- **30-Day Rolling Volatility:** How annualized volatility changes through time using a moving 30-day window.
- **Benchmark Correlation:** How closely the stock's daily returns move with the selected market benchmark.
"""
    )

st.divider()
st.caption(
    "Data source: Yahoo Finance through the yfinance Python package. "
    "This application is for educational and portfolio purposes only."
)
