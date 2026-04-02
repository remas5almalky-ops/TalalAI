"""High-profit opportunity scanner: finds trades with 10%+ profit potential.

Scans all Sharia-compliant stocks for specific high-probability setups
that can realistically deliver 10%+ returns in the short-to-medium term.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from services import data_fetcher, technical_analysis
from services.cache import cache
from config import SHARIA_ONLY

logger = logging.getLogger(__name__)

MIN_PROFIT_PCT = 10.0  # Minimum target profit


@dataclass
class Opportunity:
    """A high-profit trading opportunity."""
    ticker: str = ""
    name: str = ""
    name_ar: str = ""
    sector: str = ""
    sharia: bool = True

    # Verdict
    action: str = "Pass"           # Buy / Watch / Pass
    setup_type: str = ""           # Type of setup detected
    probability: str = ""          # High / Medium / Low
    confidence: float = 0.0

    # Price
    current_price: float = 0.0
    change_pct: float = 0.0

    # Trade plan
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    profit_pct: float = 0.0
    risk_pct: float = 0.0
    risk_reward: float = 0.0

    # Context
    atr_pct: float = 0.0          # Daily volatility %
    rsi: float = 50.0
    trend: str = "Sideways"
    volume_ratio: float = 1.0
    upside_to_52w_high: float = 0.0
    gain_last_5d: float = 0.0

    # Scoring
    setup_score: float = 0.0       # 0-100 quality of setup
    timing_score: float = 0.0      # 0-100 quality of entry timing

    # Explanation
    reasons: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    explanation: str = ""
    timeframe: str = ""            # Expected hold period


def scan_opportunities() -> list[Opportunity]:
    """Scan all stocks for 10%+ profit opportunities."""
    cached = cache.get("opportunities_10pct")
    if cached is not None:
        return cached

    stocks = data_fetcher.load_stock_list()
    if SHARIA_ONLY:
        stocks = [s for s in stocks if s.get("sharia", False)]

    all_data = data_fetcher.fetch_multiple_stocks([s["ticker"] for s in stocks])

    opportunities = []
    for stock_meta in stocks:
        ticker = stock_meta["ticker"]
        df = all_data.get(ticker)
        if df is None or df.empty or len(df) < 30:
            continue

        try:
            opp = _analyze_opportunity(ticker, stock_meta, df)
            if opp and opp.action != "Pass":
                opportunities.append(opp)
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")

    # Sort by: action (Buy first), then setup_score descending
    action_order = {"Buy": 0, "Watch": 1, "Pass": 2}
    opportunities.sort(key=lambda o: (action_order.get(o.action, 2), -o.setup_score))

    cache.set("opportunities_10pct", opportunities, ttl=900)
    return opportunities


def _analyze_opportunity(ticker: str, meta: dict, df: pd.DataFrame) -> Opportunity | None:
    """Analyze a single stock for 10%+ opportunity."""
    signals = technical_analysis.analyze(df)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = float(close.iloc[-1])

    if price <= 0:
        return None

    atr = signals.atr if signals.atr > 0 else price * 0.02
    atr_pct = atr / price * 100

    # Skip stocks with very low volatility (can't reach 10%)
    if atr_pct < 1.5:
        return None

    opp = Opportunity(
        ticker=ticker,
        name=meta["name"],
        name_ar=meta["name_ar"],
        sector=meta["sector"],
        sharia=meta.get("sharia", False),
        current_price=price,
        atr_pct=round(atr_pct, 2),
        rsi=signals.rsi,
        trend=signals.trend,
        volume_ratio=signals.volume_ratio,
    )

    # Change %
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        opp.change_pct = round((price - prev) / prev * 100, 2) if prev > 0 else 0

    # 5-day gain
    if len(close) >= 6:
        p5 = float(close.iloc[-6])
        opp.gain_last_5d = round((price - p5) / p5 * 100, 2) if p5 > 0 else 0

    # Upside to 52-week high
    if signals.high_52w > price:
        opp.upside_to_52w_high = round((signals.high_52w - price) / price * 100, 1)

    # ═══════════════════════════════════════════════
    # DETECT HIGH-PROFIT SETUPS
    # ═══════════════════════════════════════════════

    setups_found = []

    # ── SETUP 1: DEEP OVERSOLD RECOVERY ──
    # Stock is deeply oversold with large upside to prior levels
    if signals.rsi < 35 and opp.upside_to_52w_high > 15:
        score = 70
        if signals.rsi < 25:
            score += 15
        if signals.stoch_k < 20:
            score += 10
        if signals.macd_crossover == "Bullish":
            score += 10
        if signals.trend != "Downtrend":
            score += 5

        # Target: recovery to middle of 52w range
        mid_52w = (signals.high_52w + signals.low_52w) / 2
        target = max(mid_52w, price * 1.10)
        profit = (target - price) / price * 100

        if profit >= MIN_PROFIT_PCT:
            setups_found.append({
                "type": "Deep Oversold Recovery",
                "score": min(100, score),
                "target": round(target, 2),
                "stop": round(max(signals.support * 0.98, price - atr * 2.5), 2),
                "profit": round(profit, 1),
                "timeframe": "2-4 weeks",
                "reasons": [
                    f"RSI deeply oversold at {signals.rsi:.0f}",
                    f"{opp.upside_to_52w_high:.0f}% below 52-week high - massive recovery potential",
                    f"Target: recovery to mid-range at {target:.2f} SAR",
                ],
                "probability": "High" if score >= 75 else "Medium",
            })

    # ── SETUP 2: BREAKOUT FROM CONSOLIDATION ──
    # Price breaking out of tight range with volume
    recent_20 = df.tail(20)
    range_20d = (float(recent_20["High"].max()) - float(recent_20["Low"].min()))
    range_pct = range_20d / price * 100 if price > 0 else 0

    # Tight consolidation (range < 10%) then breaking up
    if range_pct < 12 and signals.bb_position == "Upper" and signals.volume_ratio > 1.2:
        prior_high = float(df.tail(60)["High"].max())
        target = max(prior_high, price * 1.10)
        profit = (target - price) / price * 100
        score = 65

        if signals.macd_crossover == "Bullish":
            score += 15
        if signals.volume_ratio > 1.5:
            score += 10
        if signals.trend == "Uptrend":
            score += 10

        if profit >= MIN_PROFIT_PCT:
            setups_found.append({
                "type": "Breakout from Consolidation",
                "score": min(100, score),
                "target": round(target, 2),
                "stop": round(float(recent_20["Low"].min()) * 0.99, 2),
                "profit": round(profit, 1),
                "timeframe": "1-3 weeks",
                "reasons": [
                    f"Tight consolidation ({range_pct:.1f}% range) breaking upward",
                    f"Volume confirmation at {signals.volume_ratio:.1f}x average",
                    "Breakouts from tight ranges often produce explosive moves",
                ],
                "probability": "Medium" if score < 75 else "High",
            })

    # ── SETUP 3: VOLATILE UPTREND PULLBACK ──
    # High-ATR stock in uptrend pulling back (buy the dip for 10%+)
    if (atr_pct > 2.5 and signals.trend == "Uptrend" and
            opp.gain_last_5d < 0 and signals.rsi < 55 and
            price > signals.sma_50):
        # Target: prior swing high or 3x ATR
        prior_swing_high = float(df.tail(30)["High"].max())
        target = max(prior_swing_high, price + atr * 3.5)
        profit = (target - price) / price * 100
        score = 70

        if signals.rsi < 40:
            score += 10
        if signals.macd_crossover == "Bullish":
            score += 10
        if abs(price - signals.sma_20) / price < 0.015:
            score += 10
            setups_found[0]["reasons"].append("Testing SMA20 support") if setups_found else None

        if profit >= MIN_PROFIT_PCT:
            setups_found.append({
                "type": "Uptrend Pullback (High Volatility)",
                "score": min(100, score),
                "target": round(target, 2),
                "stop": round(max(signals.support * 0.99, price - atr * 1.5), 2),
                "profit": round(profit, 1),
                "timeframe": "1-2 weeks",
                "reasons": [
                    f"High-volatility stock (ATR {atr_pct:.1f}%) pulling back in uptrend",
                    f"Price dipped {abs(opp.gain_last_5d):.1f}% in 5 days - buying opportunity",
                    f"Target: prior swing high at {target:.2f} SAR (+{profit:.1f}%)",
                ],
                "probability": "High" if score >= 75 else "Medium",
            })

    # ── SETUP 4: RECOVERY TO 52-WEEK HIGH ──
    # Stock with strong uptrend and large gap to 52w high
    if (opp.upside_to_52w_high >= 12 and signals.trend == "Uptrend" and
            signals.rsi < 65 and signals.ema_12 > signals.ema_26):
        target = round(signals.high_52w * 0.98, 2)  # Just below 52w high
        profit = (target - price) / price * 100
        score = 60

        if signals.macd_crossover == "Bullish":
            score += 15
        if signals.volume_ratio > 1.2:
            score += 10
        if signals.ma_signal == "Golden Cross":
            score += 10

        if profit >= MIN_PROFIT_PCT:
            setups_found.append({
                "type": "Recovery to 52-Week High",
                "score": min(100, score),
                "target": target,
                "stop": round(max(signals.support * 0.99, price - atr * 2.0), 2),
                "profit": round(profit, 1),
                "timeframe": "2-6 weeks",
                "reasons": [
                    f"Strong uptrend with {opp.upside_to_52w_high:.0f}% gap to 52-week high",
                    "Momentum indicators aligned bullish",
                    f"Target: approach 52-week high at {signals.high_52w:.2f} SAR",
                ],
                "probability": "Medium",
            })

    # ── SETUP 5: EXTREME VOLATILITY BOUNCE ──
    # Stock crashed hard, now bouncing with volume
    if (atr_pct > 3.5 and opp.gain_last_5d < -5 and
            signals.rsi < 40 and signals.volume_ratio > 1.3):
        target = round(price * 1.12, 2)
        profit = 12.0
        score = 65

        if signals.stoch_k < 20:
            score += 15
        if signals.macd_crossover == "Bullish":
            score += 15

        setups_found.append({
            "type": "Volatility Bounce (Reversal)",
            "score": min(100, score),
            "target": target,
            "stop": round(price - atr * 1.5, 2),
            "profit": profit,
            "timeframe": "3-7 days",
            "reasons": [
                f"Dropped {abs(opp.gain_last_5d):.1f}% in 5 days - oversold bounce setup",
                f"High volatility stock (ATR {atr_pct:.1f}%) can recover quickly",
                f"Volume at {signals.volume_ratio:.1f}x average - buying pressure detected",
            ],
            "probability": "Medium",
        })

    # ── SETUP 6: GOLDEN CROSS + UNDERVALUED ──
    if (signals.ma_signal == "Golden Cross" and opp.upside_to_52w_high > 15 and
            signals.rsi < 60):
        target = round(price * 1.12, 2)
        profit = 12.0
        score = 72

        if signals.macd_crossover == "Bullish":
            score += 10
        if signals.volume_ratio > 1.0:
            score += 5

        setups_found.append({
            "type": "Golden Cross + Recovery Potential",
            "score": min(100, score),
            "target": target,
            "stop": round(signals.sma_200 * 0.98, 2),
            "profit": profit,
            "timeframe": "2-4 weeks",
            "reasons": [
                "Golden Cross (SMA50 > SMA200) - major bullish signal",
                f"Still {opp.upside_to_52w_high:.0f}% below 52-week high",
                "Long-term trend has just turned bullish",
            ],
            "probability": "High",
        })

    # ═══════════════════════════════════════════════
    # SELECT BEST SETUP
    # ═══════════════════════════════════════════════

    if not setups_found:
        return None

    # Pick the highest-scoring setup
    best = max(setups_found, key=lambda s: s["score"])

    opp.setup_type = best["type"]
    opp.setup_score = best["score"]
    opp.target_price = best["target"]
    opp.stop_loss = best["stop"]
    opp.profit_pct = best["profit"]
    opp.timeframe = best["timeframe"]
    opp.reasons = best["reasons"]
    opp.probability = best["probability"]
    opp.entry_price = round(price, 2)

    # Risk calculation
    if opp.stop_loss < price and opp.stop_loss > 0:
        opp.risk_pct = round((price - opp.stop_loss) / price * 100, 1)
    else:
        opp.risk_pct = round(atr_pct * 2, 1)
        opp.stop_loss = round(price * (1 - opp.risk_pct / 100), 2)

    if opp.risk_pct > 0:
        opp.risk_reward = round(opp.profit_pct / opp.risk_pct, 1)

    # ── TIMING CHECK ── Is NOW the right time?
    timing = 50
    timing_penalties = []

    if opp.gain_last_5d > 6:
        timing -= 25
        timing_penalties.append("Already moved significantly - may be chasing")
    elif opp.gain_last_5d > 3:
        timing -= 10

    if signals.rsi > 70:
        timing -= 20
        timing_penalties.append("RSI overbought - wait for pullback")

    if signals.stoch_k > 85:
        timing -= 15
        timing_penalties.append("Stochastic overbought")

    if signals.bb_position == "Lower" or opp.gain_last_5d < -2:
        timing += 15
    if signals.macd_crossover == "Bullish":
        timing += 15
    if signals.rsi < 40:
        timing += 15
    if signals.volume_ratio > 1.3 and opp.gain_last_5d <= 2:
        timing += 10

    opp.timing_score = max(0, min(100, timing))
    opp.risks = timing_penalties

    # ── FINAL VERDICT ──
    if opp.setup_score >= 65 and opp.timing_score >= 50 and opp.risk_reward >= 1.5:
        opp.action = "Buy"
        opp.confidence = round(min(95, (opp.setup_score * 0.6 + opp.timing_score * 0.4)), 1)
    elif opp.setup_score >= 55 and opp.profit_pct >= 10:
        opp.action = "Watch"
        opp.confidence = round(min(80, opp.setup_score * 0.7), 1)
        if timing_penalties:
            opp.risks.insert(0, "Timing not ideal - wait for better entry")
    else:
        return None  # Not worth showing

    # ── EXPLANATION ──
    opp.explanation = _build_explanation(opp)

    return opp


def _build_explanation(o: Opportunity) -> str:
    """Build explanation for the opportunity."""
    reasons_str = " | ".join(o.reasons[:2])

    if o.action == "Buy":
        return (f"BUY: {o.name} ({o.setup_type}) - {reasons_str}. "
                f"Entry at {o.entry_price:.2f} SAR with {o.profit_pct:.0f}% upside target "
                f"({o.target_price:.2f} SAR). Stop loss at {o.stop_loss:.2f} SAR "
                f"(risk {o.risk_pct:.1f}%). Risk/Reward {o.risk_reward:.1f}:1. "
                f"Timeframe: {o.timeframe}. Probability: {o.probability}.")
    else:
        return (f"WATCH: {o.name} ({o.setup_type}) - {reasons_str}. "
                f"Setup has {o.profit_pct:.0f}% potential but timing is not ideal. "
                f"{'Wait for: ' + o.risks[0] if o.risks else 'Monitor for better entry.'} "
                f"Target: {o.target_price:.2f} SAR. Timeframe: {o.timeframe}.")
