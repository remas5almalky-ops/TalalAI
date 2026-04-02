"""Short-term swing trade analyzer (3-7 day horizon).

This is NOT a momentum scorer. It answers one question:
"Is RIGHT NOW a good entry point for a short-term trade?"

It penalizes stocks that already ran up, are near resistance, or are overextended.
It rewards stocks near support, in pullback, or showing early breakout signals.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from services import data_fetcher, technical_analysis
from services.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class SwingSignal:
    """Result of short-term swing analysis."""
    ticker: str = ""
    name: str = ""
    name_ar: str = ""
    sector: str = ""
    sharia: bool = True
    sharia_note: str = ""

    # Core verdict
    action: str = "Wait"           # Buy / Wait / Avoid
    confidence: float = 0.0        # 0-100
    timing_score: float = 0.0      # 0-100 (is NOW a good time?)

    # Price data
    current_price: float = 0.0
    change_pct: float = 0.0

    # Trade plan
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    expected_profit_pct: float = 0.0
    risk_pct: float = 0.0
    risk_reward: float = 0.0

    # Timing factors
    entry_quality: str = ""        # "Near Support" / "Pullback" / "Early Breakout" / "Extended" / "Overpriced"
    timing_reasons: list = field(default_factory=list)
    warning_flags: list = field(default_factory=list)

    # Key indicators
    rsi: float = 50.0
    stoch_k: float = 50.0
    macd_crossover: str = "Neutral"
    trend: str = "Sideways"
    volume_ratio: float = 1.0
    bb_position: str = "Middle"
    support: float = 0.0
    resistance: float = 0.0
    atr: float = 0.0

    # Recent price action
    gain_last_3d: float = 0.0      # % gain in last 3 days
    gain_last_5d: float = 0.0      # % gain in last 5 days
    dist_from_support_pct: float = 0.0
    dist_from_resistance_pct: float = 0.0

    # Explanation
    explanation: str = ""


def analyze_swing(ticker: str) -> SwingSignal | None:
    """Analyze a stock for short-term swing trade entry timing."""
    cache_key = f"swing_{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    meta = data_fetcher.get_stock_meta(ticker)
    df = data_fetcher.fetch_stock_data(ticker)

    if df.empty or len(df) < 30:
        return None

    signals = technical_analysis.analyze(df)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = float(close.iloc[-1])

    result = SwingSignal(
        ticker=ticker,
        name=meta["name"],
        name_ar=meta["name_ar"],
        sector=meta["sector"],
        sharia=meta.get("sharia", False),
        sharia_note=meta.get("sharia_note", ""),
        current_price=price,
        rsi=signals.rsi,
        stoch_k=signals.stoch_k,
        macd_crossover=signals.macd_crossover,
        trend=signals.trend,
        volume_ratio=signals.volume_ratio,
        bb_position=signals.bb_position,
        support=signals.support,
        resistance=signals.resistance,
        atr=signals.atr,
    )

    # --- Change % ---
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        result.change_pct = round((price - prev) / prev * 100, 2) if prev > 0 else 0

    # --- Recent price action (critical for timing) ---
    if len(close) >= 4:
        p3 = float(close.iloc[-4])
        result.gain_last_3d = round((price - p3) / p3 * 100, 2) if p3 > 0 else 0
    if len(close) >= 6:
        p5 = float(close.iloc[-6])
        result.gain_last_5d = round((price - p5) / p5 * 100, 2) if p5 > 0 else 0

    # --- Distance from support/resistance ---
    if signals.support > 0:
        result.dist_from_support_pct = round((price - signals.support) / price * 100, 2)
    if signals.resistance > price:
        result.dist_from_resistance_pct = round((signals.resistance - price) / price * 100, 2)

    # ═══════════════════════════════════════════════════════════
    # TIMING ANALYSIS: Is now a good entry?
    # ═══════════════════════════════════════════════════════════

    timing_score = 50  # Start neutral
    reasons = []
    warnings = []

    # ── 1. ALREADY EXTENDED CHECK (penalty) ──
    # If stock already gained >4% in 3 days, you're late
    if result.gain_last_3d > 6:
        timing_score -= 30
        warnings.append(f"Stock already up {result.gain_last_3d:.1f}% in 3 days - you are LATE")
    elif result.gain_last_3d > 4:
        timing_score -= 20
        warnings.append(f"Stock ran up {result.gain_last_3d:.1f}% in 3 days - entry risky")
    elif result.gain_last_3d > 2.5:
        timing_score -= 10
        warnings.append(f"Stock up {result.gain_last_3d:.1f}% recently - partial move done")

    # If stock gained >8% in 5 days, strongly penalize
    if result.gain_last_5d > 8:
        timing_score -= 25
        warnings.append(f"Stock surged {result.gain_last_5d:.1f}% in 5 days - extremely overextended")
    elif result.gain_last_5d > 5:
        timing_score -= 15
        warnings.append(f"Stock up {result.gain_last_5d:.1f}% in 5 days - extended move")

    # ── 2. NEAR RESISTANCE CHECK (penalty) ──
    if result.dist_from_resistance_pct < 1.0 and result.dist_from_resistance_pct > 0:
        timing_score -= 20
        warnings.append(f"Price only {result.dist_from_resistance_pct:.1f}% from resistance - limited upside")
    elif result.dist_from_resistance_pct < 2.0:
        timing_score -= 10
        warnings.append(f"Approaching resistance ({result.dist_from_resistance_pct:.1f}% away)")

    # ── 3. RSI OVERBOUGHT CHECK (penalty) ──
    if signals.rsi > 75:
        timing_score -= 25
        warnings.append(f"RSI overbought at {signals.rsi:.0f} - high pullback risk")
    elif signals.rsi > 70:
        timing_score -= 15
        warnings.append(f"RSI elevated at {signals.rsi:.0f} - limited short-term upside")

    # ── 4. STOCHASTIC OVERBOUGHT ──
    if signals.stoch_k > 85:
        timing_score -= 15
        warnings.append(f"Stochastic overbought ({signals.stoch_k:.0f}) - reversal likely")
    elif signals.stoch_k > 80:
        timing_score -= 8

    # ── 5. BOLLINGER UPPER WITHOUT BREAKOUT ──
    if signals.bb_position == "Upper" and signals.trend != "Uptrend":
        timing_score -= 15
        warnings.append("At upper Bollinger Band without uptrend - likely to pull back")

    # ═══════════════════════════════════════════════════════════
    # POSITIVE ENTRY SIGNALS (reward)
    # ═══════════════════════════════════════════════════════════

    # ── 6. NEAR SUPPORT (best entry) ──
    if result.dist_from_support_pct < 1.5:
        timing_score += 25
        reasons.append(f"Price at support level ({result.dist_from_support_pct:.1f}% above) - ideal entry zone")
        result.entry_quality = "Near Support"
    elif result.dist_from_support_pct < 3.0:
        timing_score += 15
        reasons.append(f"Price near support ({result.dist_from_support_pct:.1f}% above)")
        result.entry_quality = "Near Support"

    # ── 7. PULLBACK IN UPTREND (great entry) ──
    is_pullback = (signals.trend == "Uptrend" and
                   result.gain_last_3d < 0 and
                   signals.rsi < 55 and
                   price > signals.sma_50)
    if is_pullback:
        timing_score += 25
        reasons.append("Healthy pullback in uptrend - classic buy-the-dip setup")
        result.entry_quality = "Pullback"

    # Also: price dipped to SMA20 in uptrend
    if (signals.trend == "Uptrend" and
            signals.sma_20 > 0 and
            abs(price - signals.sma_20) / price < 0.01):
        timing_score += 15
        reasons.append("Price testing SMA20 support in uptrend - bounce expected")
        if not result.entry_quality:
            result.entry_quality = "Pullback"

    # ── 8. RSI OVERSOLD BOUNCE ──
    if signals.rsi < 30:
        timing_score += 25
        reasons.append(f"RSI oversold at {signals.rsi:.0f} - strong bounce potential")
        if not result.entry_quality:
            result.entry_quality = "Oversold Bounce"
    elif signals.rsi < 40 and signals.trend != "Downtrend":
        timing_score += 12
        reasons.append(f"RSI low at {signals.rsi:.0f} - room to move up")

    # ── 9. STOCHASTIC OVERSOLD CROSSOVER ──
    if signals.stoch_k < 25:
        timing_score += 15
        reasons.append(f"Stochastic oversold ({signals.stoch_k:.0f}) - reversal imminent")
    elif signals.stoch_k < 35 and signals.stoch_k > signals.stoch_d:
        timing_score += 10
        reasons.append("Stochastic turning up from low levels")

    # ── 10. FRESH MACD BULLISH CROSSOVER ──
    if signals.macd_crossover == "Bullish":
        # Check if it's truly fresh (histogram just turned positive)
        timing_score += 18
        reasons.append("Fresh bullish MACD crossover - new momentum starting")
        if not result.entry_quality:
            result.entry_quality = "Early Breakout"

    # ── 11. BOLLINGER LOWER BAND BOUNCE ──
    if signals.bb_position == "Lower":
        timing_score += 18
        reasons.append("Price at lower Bollinger Band - mean reversion bounce likely")
        if not result.entry_quality:
            result.entry_quality = "Bollinger Bounce"

    # ── 12. VOLUME CONFIRMATION ──
    if signals.volume_ratio > 1.5 and result.gain_last_3d <= 2:
        timing_score += 10
        reasons.append(f"Volume surge ({signals.volume_ratio:.1f}x avg) without big price move yet - accumulation")
    elif signals.volume_ratio > 2.0 and result.gain_last_3d > 0:
        timing_score += 5
        reasons.append(f"Strong volume ({signals.volume_ratio:.1f}x) confirming move")

    # ── 13. EARLY BREAKOUT above resistance ──
    if (signals.resistance > 0 and price > signals.resistance and
            result.gain_last_3d < 3 and signals.volume_ratio > 1.2):
        timing_score += 20
        reasons.append("Breaking above resistance with volume - early breakout entry")
        result.entry_quality = "Early Breakout"

    # ── 14. DOWNTREND PENALTY ──
    if signals.trend == "Downtrend":
        timing_score -= 15
        warnings.append("Stock in downtrend - counter-trend trades are risky")

    # ── 15. LOW VOLUME WARNING ──
    if signals.volume_ratio < 0.5:
        timing_score -= 8
        warnings.append("Very low volume - move may lack conviction")

    # ═══════════════════════════════════════════════════════════
    # FINAL SCORING & VERDICT
    # ═══════════════════════════════════════════════════════════

    timing_score = max(0, min(100, timing_score))
    result.timing_score = round(timing_score, 1)
    result.timing_reasons = reasons
    result.warning_flags = warnings

    # --- Entry quality fallback ---
    if not result.entry_quality:
        if timing_score >= 60:
            result.entry_quality = "Acceptable"
        elif timing_score >= 40:
            result.entry_quality = "Neutral"
        else:
            result.entry_quality = "Extended"

    # --- Verdict ---
    has_critical_warnings = any("LATE" in w or "extremely" in w.lower() or "overbought" in w.lower() for w in warnings)

    if timing_score >= 65 and not has_critical_warnings:
        result.action = "Buy"
    elif timing_score >= 45 and not has_critical_warnings:
        result.action = "Wait"
    else:
        result.action = "Avoid"

    # Override: if stock is clearly overbought + extended, force Avoid
    if signals.rsi > 75 and result.gain_last_5d > 5:
        result.action = "Avoid"
    # Override: if near resistance with no room, force Wait
    if result.dist_from_resistance_pct < 1.5 and result.action == "Buy":
        result.action = "Wait"
        warnings.append("Downgraded to Wait: too close to resistance for short-term trade")

    # --- Confidence ---
    if result.action == "Buy":
        result.confidence = round(min(95, timing_score + len(reasons) * 3), 1)
    elif result.action == "Wait":
        result.confidence = round(timing_score, 1)
    else:
        result.confidence = round(max(10, 100 - timing_score), 1)

    # --- Trade plan (only meaningful for Buy) ---
    atr = signals.atr if signals.atr > 0 else price * 0.015

    if result.action == "Buy":
        result.entry_price = round(price, 2)
        # Short-term target: 1.5-2x ATR (3-7 day horizon)
        result.target_price = round(price + atr * 2.0, 2)
        result.stop_loss = round(price - atr * 1.2, 2)

        # Use support as tighter stop if close
        if signals.support > 0 and signals.support < price:
            tight_stop = round(signals.support * 0.995, 2)
            if tight_stop > result.stop_loss:
                result.stop_loss = tight_stop

        # Cap target at resistance if close
        if signals.resistance > price and signals.resistance < result.target_price:
            result.target_price = round(signals.resistance * 0.99, 2)

        # Ensure target > entry
        if result.target_price <= result.entry_price:
            result.target_price = round(result.entry_price + atr * 1.5, 2)

    elif result.action == "Wait":
        # Show what the ideal entry would be
        result.entry_price = round(max(signals.support, price - atr * 0.8), 2)
        result.target_price = round(result.entry_price + atr * 2.0, 2)
        result.stop_loss = round(result.entry_price - atr * 1.2, 2)
    else:
        result.entry_price = 0
        result.target_price = 0
        result.stop_loss = 0

    # --- Profit / Risk / R:R ---
    if result.entry_price > 0 and result.target_price > result.entry_price:
        result.expected_profit_pct = round((result.target_price - result.entry_price) / result.entry_price * 100, 1)
    if result.entry_price > 0 and result.stop_loss > 0 and result.stop_loss < result.entry_price:
        result.risk_pct = round((result.entry_price - result.stop_loss) / result.entry_price * 100, 1)
    if result.risk_pct > 0:
        result.risk_reward = round(result.expected_profit_pct / result.risk_pct, 1)

    # --- Explanation ---
    result.explanation = _build_explanation(result)

    cache.set(cache_key, result, ttl=900)
    return result


def _build_explanation(s: SwingSignal) -> str:
    """Build a clear explanation for the swing trade recommendation."""
    name = s.name

    if s.action == "Buy":
        top = " + ".join(s.timing_reasons[:3]) if s.timing_reasons else "favorable setup"
        text = (f"BUY NOW: {name} - {top}. "
                f"Entry at {s.entry_price:.2f} SAR with target {s.target_price:.2f} "
                f"(+{s.expected_profit_pct:.1f}%) and stop loss at {s.stop_loss:.2f} SAR "
                f"(-{s.risk_pct:.1f}%). Risk/Reward {s.risk_reward:.1f}:1. "
                f"Entry quality: {s.entry_quality}.")
        if s.warning_flags:
            text += f" Note: {s.warning_flags[0]}"
        return text

    elif s.action == "Wait":
        if s.warning_flags:
            issues = s.warning_flags[0]
        else:
            issues = "timing is not optimal"
        text = (f"WAIT: {name} - {issues}. "
                f"The stock has potential but current price is not an ideal entry point. ")
        if s.timing_reasons:
            text += f"Positive: {s.timing_reasons[0]}. "
        if s.entry_price > 0:
            text += (f"Better entry around {s.entry_price:.2f} SAR. "
                     f"Set alert and wait for pullback or clearer signal.")
        return text

    else:
        if s.warning_flags:
            issues = " | ".join(s.warning_flags[:2])
        else:
            issues = "weak setup"
        text = (f"AVOID: {name} - {issues}. "
                f"This is NOT a good time to enter. The stock is overextended or showing "
                f"bearish signals. Wait for a significant pullback or trend reversal before considering entry.")
        return text
