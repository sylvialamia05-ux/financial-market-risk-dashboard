from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


def _to_timestamp(value) -> pd.Timestamp:
    """Convert a date-like value into a pandas Timestamp."""
    return pd.Timestamp(value)


def download_market_data(
    ticker: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Download daily historical market data from Yahoo Finance.

    The end date is treated as inclusive for the user, even though Yahoo
    Finance treats the end date as exclusive.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date)

    if start >= end:
        raise ValueError("The start date must be earlier than the end date.")

    inclusive_end = end + pd.Timedelta(days=1)

    try:
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=inclusive_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance could not return data for {ticker}. "
            "Please try again in a moment."
        ) from exc

    if data is None or data.empty:
        raise ValueError(
            f"No market data was found for '{ticker}'. "
            "Check the ticker symbol and date range."
        )

    # yfinance can return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        # Keep the price-field level (Open, High, Low, Close, Volume).
        if "Close" in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)

    data = data.copy()
    data.index = pd.to_datetime(data.index)

    # Remove timezone information so stock and benchmark data align cleanly.
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    required = {"Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Downloaded data for {ticker} is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    return data.sort_index()


def calculate_daily_returns(close_prices: pd.Series) -> pd.Series:
    """Calculate simple daily percentage returns."""
    return close_prices.pct_change()


def calculate_total_return(close_prices: pd.Series) -> float:
    """Calculate total return over the selected period."""
    clean = close_prices.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[0] - 1)


def calculate_average_daily_return(daily_returns: pd.Series) -> float:
    """Calculate the average daily return."""
    clean = daily_returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def calculate_annualized_volatility(daily_returns: pd.Series) -> float:
    """
    Annualize the standard deviation of daily returns using 252 trading days.
    """
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.04,
) -> float:
    """
    Calculate the annualized Sharpe ratio.

    annual_risk_free_rate should be supplied as a decimal, e.g. 0.04 for 4%.
    """
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return np.nan

    volatility = clean.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return np.nan

    daily_risk_free = (1 + annual_risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess_daily_return = clean.mean() - daily_risk_free

    return float((excess_daily_return / volatility) * np.sqrt(TRADING_DAYS))


def calculate_drawdown(close_prices: pd.Series) -> pd.Series:
    """Calculate drawdown from the running peak."""
    clean = close_prices.astype(float)
    running_peak = clean.cummax()
    return clean / running_peak - 1


def calculate_max_drawdown(close_prices: pd.Series) -> float:
    """Return the worst drawdown during the selected period."""
    drawdown = calculate_drawdown(close_prices).dropna()
    if drawdown.empty:
        return np.nan
    return float(drawdown.min())


def calculate_historical_var(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate one-day Historical Value at Risk.

    The returned value is a positive loss magnitude. For example, 0.025 means
    the historical 95% VaR is approximately a 2.5% one-day loss.
    """
    clean = daily_returns.dropna()
    if clean.empty:
        return np.nan

    percentile = 1 - confidence_level
    return float(max(0.0, -clean.quantile(percentile)))


def calculate_benchmark_correlation(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate correlation between aligned daily stock and benchmark returns."""
    aligned = pd.concat(
        [
            stock_returns.rename("Stock"),
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    return float(aligned["Stock"].corr(aligned["Benchmark"]))


def classify_risk(annualized_volatility: float) -> Tuple[str, str]:
    """
    Classify volatility into a simple educational risk category.

    Thresholds are intentionally simple for this portfolio project and should
    not be interpreted as investment advice.
    """
    if np.isnan(annualized_volatility):
        return "Unavailable", "Not enough data to classify risk."

    if annualized_volatility < 0.20:
        return "Low Risk", "Annualized volatility is below 20%."
    if annualized_volatility < 0.35:
        return "Moderate Risk", "Annualized volatility is between 20% and 35%."
    return "High Risk", "Annualized volatility is 35% or higher."


def add_analysis_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated return, trend, volatility, cumulative-return, and drawdown
    columns to a downloaded market-data DataFrame.
    """
    result = data.copy()

    result["Daily Return"] = calculate_daily_returns(result["Close"])
    result["MA20"] = result["Close"].rolling(window=20).mean()
    result["MA50"] = result["Close"].rolling(window=50).mean()
    result["Rolling 30D Volatility"] = (
        result["Daily Return"].rolling(window=30).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )
    result["Cumulative Return"] = (1 + result["Daily Return"].fillna(0)).cumprod() - 1
    result["Drawdown"] = calculate_drawdown(result["Close"])

    return result


def normalize_performance(close_prices: pd.Series) -> pd.Series:
    """
    Convert prices into growth-of-$100 values for an apples-to-apples comparison.
    """
    clean = close_prices.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)

    return (clean / clean.iloc[0]) * 100
