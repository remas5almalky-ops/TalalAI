"""Technical analysis engine using the `ta` library."""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from models.stock import TechnicalSignals


def analyze(df: pd.DataFrame) -> TechnicalSignals:
    """Run full technical analysis on OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns

    Returns:
        TechnicalSignals with all computed indicators
    """
    if df.empty or len(df) < 30:
        return TechnicalSignals()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    current_price = float(close.iloc[-1])

    signals = TechnicalSignals()
    signals.price = current_price

    # === RSI ===
    rsi_indicator = RSIIndicator(close=close, window=14)
    rsi_values = rsi_indicator.rsi()
    signals.rsi = float(rsi_values.iloc[-1]) if not rsi_values.empty else 50.0

    if signals.rsi > 70:
        signals.rsi_signal = "Overbought"
    elif signals.rsi < 30:
        signals.rsi_signal = "Oversold"
    else:
        signals.rsi_signal = "Neutral"

    # === MACD ===
    macd_indicator = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    signals.macd = float(macd_indicator.macd().iloc[-1] or 0)
    signals.macd_signal = float(macd_indicator.macd_signal().iloc[-1] or 0)
    signals.macd_histogram = float(macd_indicator.macd_diff().iloc[-1] or 0)

    # Check crossover (compare last 2 values)
    macd_hist = macd_indicator.macd_diff()
    if len(macd_hist) >= 2:
        prev_hist = float(macd_hist.iloc[-2] or 0)
        curr_hist = float(macd_hist.iloc[-1] or 0)
        if prev_hist < 0 and curr_hist > 0:
            signals.macd_crossover = "Bullish"
        elif prev_hist > 0 and curr_hist < 0:
            signals.macd_crossover = "Bearish"
        else:
            signals.macd_crossover = "Bullish" if curr_hist > 0 else "Bearish"

    # === Bollinger Bands ===
    bb = BollingerBands(close=close, window=20, window_dev=2)
    signals.bb_upper = float(bb.bollinger_hband().iloc[-1] or 0)
    signals.bb_middle = float(bb.bollinger_mavg().iloc[-1] or 0)
    signals.bb_lower = float(bb.bollinger_lband().iloc[-1] or 0)

    if signals.bb_upper > signals.bb_lower:
        bb_range = signals.bb_upper - signals.bb_lower
        bb_pos = (current_price - signals.bb_lower) / bb_range
        if bb_pos > 0.8:
            signals.bb_position = "Upper"
        elif bb_pos < 0.2:
            signals.bb_position = "Lower"
        else:
            signals.bb_position = "Middle"

    # === Moving Averages ===
    signals.sma_20 = float(SMAIndicator(close=close, window=20).sma_indicator().iloc[-1] or 0)
    signals.sma_50 = float(SMAIndicator(close=close, window=min(50, len(close))).sma_indicator().iloc[-1] or 0)

    if len(close) >= 200:
        signals.sma_200 = float(SMAIndicator(close=close, window=200).sma_indicator().iloc[-1] or 0)
    else:
        signals.sma_200 = float(SMAIndicator(close=close, window=len(close)).sma_indicator().iloc[-1] or 0)

    signals.ema_12 = float(EMAIndicator(close=close, window=12).ema_indicator().iloc[-1] or 0)
    signals.ema_26 = float(EMAIndicator(close=close, window=min(26, len(close))).ema_indicator().iloc[-1] or 0)

    # MA Signal
    if signals.sma_50 > signals.sma_200 and current_price > signals.sma_50:
        signals.ma_signal = "Golden Cross"
    elif signals.sma_50 < signals.sma_200 and current_price < signals.sma_50:
        signals.ma_signal = "Death Cross"
    else:
        signals.ma_signal = "Neutral"

    # === ATR ===
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    signals.atr = float(atr.average_true_range().iloc[-1] or 0)

    # === Volume Analysis ===
    vol_sma_20 = volume.rolling(window=20).mean()
    if not vol_sma_20.empty and vol_sma_20.iloc[-1] > 0:
        signals.volume_ratio = float(volume.iloc[-1] / vol_sma_20.iloc[-1])
    else:
        signals.volume_ratio = 1.0

    if signals.volume_ratio > 1.5:
        signals.volume_signal = "High"
    elif signals.volume_ratio < 0.5:
        signals.volume_signal = "Low"
    else:
        signals.volume_signal = "Normal"

    # === Stochastic ===
    if len(close) >= 14:
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        denom = high_14 - low_14
        denom = denom.replace(0, np.nan)
        stoch_k = ((close - low_14) / denom) * 100
        signals.stoch_k = float(stoch_k.iloc[-1] or 50)
        signals.stoch_d = float(stoch_k.rolling(window=3).mean().iloc[-1] or 50)

    # === Trend Detection ===
    signals.trend = _detect_trend(signals, current_price)

    # === 52-week High/Low ===
    year_data = close.tail(252) if len(close) >= 252 else close
    signals.high_52w = float(year_data.max())
    signals.low_52w = float(year_data.min())

    # === Support / Resistance ===
    signals.support, signals.resistance = _find_support_resistance(df)

    return signals


def _detect_trend(signals: TechnicalSignals, price: float) -> str:
    """Detect overall trend from indicator signals."""
    bullish_count = 0
    bearish_count = 0

    # Price vs MAs
    if price > signals.sma_50:
        bullish_count += 1
    else:
        bearish_count += 1

    if signals.sma_50 > signals.sma_200:
        bullish_count += 1
    else:
        bearish_count += 1

    # MACD
    if signals.macd_histogram > 0:
        bullish_count += 1
    else:
        bearish_count += 1

    # RSI
    if signals.rsi > 55:
        bullish_count += 1
    elif signals.rsi < 45:
        bearish_count += 1

    # EMA
    if signals.ema_12 > signals.ema_26:
        bullish_count += 1
    else:
        bearish_count += 1

    if bullish_count >= 4:
        return "Uptrend"
    elif bearish_count >= 4:
        return "Downtrend"
    return "Sideways"


def _find_support_resistance(df: pd.DataFrame) -> tuple[float, float]:
    """Find nearest support and resistance levels using pivot points."""
    if len(df) < 20:
        price = float(df["Close"].iloc[-1])
        return price * 0.97, price * 1.03

    recent = df.tail(60)
    close = float(recent["Close"].iloc[-1])
    highs = recent["High"].values
    lows = recent["Low"].values

    # Find local minima as support candidates
    supports = []
    resistances = []

    for i in range(2, len(recent) - 2):
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            if lows[i] < close:
                supports.append(float(lows[i]))
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            if highs[i] > close:
                resistances.append(float(highs[i]))

    support = max(supports) if supports else close * 0.97
    resistance = min(resistances) if resistances else close * 1.03

    return support, resistance
