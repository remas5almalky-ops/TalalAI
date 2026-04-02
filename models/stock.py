"""Data models for the stock recommendation system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StockInfo:
    """Basic stock information."""
    ticker: str
    name: str
    name_ar: str
    sector: str
    sector_ar: str
    current_price: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    market_cap: float = 0.0
    sharia: bool = True
    sharia_note: str = ""


@dataclass
class TechnicalSignals:
    """Technical analysis indicator values."""
    # RSI
    rsi: float = 50.0
    rsi_signal: str = "Neutral"  # Overbought / Oversold / Neutral

    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "Neutral"  # Bullish / Bearish / Neutral

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: str = "Middle"  # Upper / Middle / Lower

    # Moving Averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ma_signal: str = "Neutral"  # Golden Cross / Death Cross / Neutral

    # ATR
    atr: float = 0.0

    # Volume
    volume_ratio: float = 1.0  # current / 20-day avg
    volume_signal: str = "Normal"  # High / Normal / Low

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # Trend
    trend: str = "Sideways"  # Uptrend / Downtrend / Sideways

    # Price context
    price: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0

    # Support / Resistance
    support: float = 0.0
    resistance: float = 0.0


@dataclass
class Recommendation:
    """Full stock recommendation."""
    stock: StockInfo
    signals: TechnicalSignals

    # Recommendation
    action: str = "Hold"  # Strong Buy / Buy / Hold / Avoid
    confidence: float = 50.0
    opportunity_score: float = 50.0
    predicted_move: float = 0.0  # Expected % price move

    # Price targets
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0

    # Explanation
    explanation: str = ""
    factors: list = field(default_factory=list)

    # Investment type suitability
    short_term: bool = False
    long_term: bool = False

    # Score breakdown
    score_breakdown: dict = field(default_factory=dict)
