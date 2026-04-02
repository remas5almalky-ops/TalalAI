"""Data fetcher for Saudi stock market using yfinance."""

import json
import os
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from services.cache import cache
from config import HISTORICAL_PERIOD, HISTORICAL_INTERVAL, CACHE_TTL_SECONDS, SHARIA_ONLY

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_stock_list() -> list[dict]:
    """Load the list of Saudi stocks from JSON."""
    cached = cache.get("stock_list")
    if cached is not None:
        return cached

    with open(DATA_DIR / "saudi_stocks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    stocks = data["stocks"]
    cache.set("stock_list", stocks, ttl=3600)
    return stocks


def get_sectors() -> list[dict]:
    """Get the list of sectors."""
    with open(DATA_DIR / "saudi_stocks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sectors"]


def fetch_stock_data(ticker: str, period: str = None) -> pd.DataFrame:
    """Fetch historical OHLCV data for a stock.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    period = period or HISTORICAL_PERIOD
    cache_key = f"data_{ticker}_{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=HISTORICAL_INTERVAL)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Clean up columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        cache.set(cache_key, df, ttl=CACHE_TTL_SECONDS)
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_multiple_stocks(tickers: list[str], period: str = None) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple stocks."""
    results = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            results[ticker] = df
    return results


def get_stock_info(ticker: str) -> dict:
    """Get basic stock info from yfinance."""
    cache_key = f"info_{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "volume": info.get("volume") or info.get("regularMarketVolume", 0),
            "change_pct": info.get("regularMarketChangePercent", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "year_high": info.get("fiftyTwoWeekHigh", 0),
            "year_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
        }
        cache.set(cache_key, result, ttl=CACHE_TTL_SECONDS)
        return result
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {}


def get_all_tickers() -> list[str]:
    """Get all Tadawul ticker symbols (respects Sharia filter)."""
    stocks = load_stock_list()
    if SHARIA_ONLY:
        stocks = [s for s in stocks if s.get("sharia", False)]
    return [s["ticker"] for s in stocks]


def is_sharia_compliant(ticker: str) -> bool:
    """Check if a stock is Sharia-compliant."""
    stocks = load_stock_list()
    for s in stocks:
        if s["ticker"] == ticker:
            return s.get("sharia", False)
    return False


def get_sharia_note(ticker: str) -> str:
    """Get Sharia compliance note for a stock."""
    stocks = load_stock_list()
    for s in stocks:
        if s["ticker"] == ticker:
            return s.get("sharia_note", "")
    return ""


def get_stock_meta(ticker: str) -> dict:
    """Get stock metadata (name, sector) from our JSON file."""
    stocks = load_stock_list()
    for s in stocks:
        if s["ticker"] == ticker:
            return s
    return {"ticker": ticker, "name": ticker, "name_ar": ticker, "sector": "Unknown", "sector_ar": "غير معروف"}
