"""Scoring engine: hybrid momentum + value model for buy recommendations."""

from models.stock import StockInfo, TechnicalSignals, Recommendation
from config import (
    SCORING_WEIGHTS,
    STRONG_BUY_THRESHOLD,
    BUY_THRESHOLD,
    HOLD_THRESHOLD,
    TARGET_ATR_MULTIPLIER,
    STOP_LOSS_ATR_MULTIPLIER,
)


def score_stock(stock: StockInfo, signals: TechnicalSignals) -> Recommendation:
    """Generate a full recommendation for a stock based on technical signals."""
    breakdown = {}
    factors = []

    # === MOMENTUM SCORE (25%) - Primary driver ===
    momentum_score, momentum_factors = _score_momentum(signals)
    breakdown["momentum"] = momentum_score
    factors.extend(momentum_factors)

    # === MACD SCORE (15%) ===
    macd_score, macd_factors = _score_macd(signals)
    breakdown["macd"] = macd_score
    factors.extend(macd_factors)

    # === RSI SCORE (10%) - Momentum zone scoring ===
    rsi_score, rsi_factors = _score_rsi(signals)
    breakdown["rsi"] = rsi_score
    factors.extend(rsi_factors)

    # === TREND SCORE (15%) ===
    trend_score, trend_factors = _score_trend(signals)
    breakdown["trend"] = trend_score
    factors.extend(trend_factors)

    # === VOLUME SCORE (10%) ===
    volume_score, volume_factors = _score_volume(signals)
    breakdown["volume"] = volume_score
    factors.extend(volume_factors)

    # === BREAKOUT SCORE (10%) - Bollinger + 52w position ===
    breakout_score, breakout_factors = _score_breakout(signals)
    breakdown["breakout"] = breakout_score
    factors.extend(breakout_factors)

    # === VOLATILITY SCORE (5%) ===
    volatility_score = _score_volatility(signals)
    breakdown["volatility"] = volatility_score

    # === SUPPORT/RESISTANCE SCORE (10%) ===
    sr_score, sr_factors = _score_support_resistance(signals)
    breakdown["support_resistance"] = sr_score
    factors.extend(sr_factors)

    # === Calculate Weighted Opportunity Score ===
    opportunity_score = sum(
        breakdown[key] * SCORING_WEIGHTS[key]
        for key in SCORING_WEIGHTS
    )
    opportunity_score = round(min(100, max(0, opportunity_score)), 1)

    # === Bonus: compound signal boost ===
    # When multiple strong signals align, boost the score
    bullish_signals = 0
    if signals.macd_crossover == "Bullish":
        bullish_signals += 1
    if signals.trend == "Uptrend":
        bullish_signals += 1
    if signals.volume_ratio > 1.2:
        bullish_signals += 1
    if 40 <= signals.rsi <= 65:
        bullish_signals += 1
    if signals.ema_12 > signals.ema_26:
        bullish_signals += 1

    if bullish_signals >= 4:
        opportunity_score = min(100, opportunity_score + 8)
        factors.insert(0, "Multiple bullish signals converging (strong alignment)")
    elif bullish_signals >= 3:
        opportunity_score = min(100, opportunity_score + 4)

    opportunity_score = round(opportunity_score, 1)

    # === Determine Recommendation ===
    if opportunity_score >= STRONG_BUY_THRESHOLD:
        action = "Strong Buy"
    elif opportunity_score >= BUY_THRESHOLD:
        action = "Buy"
    elif opportunity_score >= HOLD_THRESHOLD:
        action = "Hold"
    else:
        action = "Avoid"

    # === Confidence: based on signal agreement ===
    confidence = round(min(99, opportunity_score + (bullish_signals * 2)), 1)

    # === Predicted Move (%) ===
    predicted_move = _predict_move(signals, opportunity_score)

    # === Price Targets ===
    entry_price = signals.price
    atr = signals.atr if signals.atr > 0 else signals.price * 0.02

    if action in ("Strong Buy", "Buy"):
        target_price = round(entry_price + (atr * TARGET_ATR_MULTIPLIER), 2)
        stop_loss = round(entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER), 2)
    elif action == "Hold":
        target_price = round(entry_price + (atr * 1.5), 2)
        stop_loss = round(entry_price - (atr * 1.0), 2)
    else:
        target_price = round(entry_price + (atr * 0.5), 2)
        stop_loss = round(entry_price - (atr * 2.0), 2)

    # Use resistance as target if better
    if signals.resistance > entry_price:
        target_price = round(max(target_price, signals.resistance), 2)

    # Use support as stop loss if reasonable
    if 0 < signals.support < entry_price:
        stop_loss = round(max(stop_loss, signals.support * 0.99), 2)

    # === Generate Explanation ===
    explanation = _generate_explanation(stock, signals, action, factors, predicted_move)

    # === Investment type suitability ===
    short_term = (signals.macd_crossover == "Bullish" and signals.volume_ratio > 1.0) or signals.rsi < 35
    long_term = signals.trend == "Uptrend" and (signals.ma_signal == "Golden Cross" or signals.sma_50 > signals.sma_200)

    return Recommendation(
        stock=stock,
        signals=signals,
        action=action,
        confidence=confidence,
        opportunity_score=opportunity_score,
        predicted_move=predicted_move,
        entry_price=round(entry_price, 2),
        target_price=target_price,
        stop_loss=stop_loss,
        explanation=explanation,
        factors=factors[:6],
        short_term=short_term,
        long_term=long_term,
        score_breakdown=breakdown,
    )


# ── Scoring Functions ─────────────────────────────────────────────


def _score_momentum(signals: TechnicalSignals) -> tuple[float, list]:
    """Score price momentum: rewards stocks that are moving up with strength."""
    score = 50
    factors = []

    # EMA crossover (fast above slow = bullish momentum)
    if signals.ema_12 > signals.ema_26:
        ema_spread = ((signals.ema_12 - signals.ema_26) / signals.ema_26) * 100
        if ema_spread > 2:
            score += 30
            factors.append(f"Strong bullish momentum (EMA spread +{ema_spread:.1f}%)")
        elif ema_spread > 0.5:
            score += 20
            factors.append("Positive momentum with EMA12 above EMA26")
        else:
            score += 10
    else:
        ema_spread = ((signals.ema_26 - signals.ema_12) / signals.ema_26) * 100
        if ema_spread > 2:
            score -= 20
        else:
            score -= 10

    # Price above SMA20 (short-term momentum)
    if signals.price > signals.sma_20 > 0:
        score += 10
    elif signals.sma_20 > 0:
        score -= 5

    # Price above SMA50 (medium-term momentum)
    if signals.price > signals.sma_50 > 0:
        score += 10
    elif signals.sma_50 > 0:
        score -= 10

    return max(0, min(100, score)), factors


def _score_macd(signals: TechnicalSignals) -> tuple[float, list]:
    """Score MACD: bullish crossover and increasing histogram = strong buy signal."""
    score = 50
    factors = []

    if signals.macd_crossover == "Bullish":
        score += 30
        if signals.macd_histogram > 0:
            score += 10
            factors.append("Fresh bullish MACD crossover with rising histogram")
        else:
            factors.append("Bullish MACD crossover detected")
    elif signals.macd_crossover == "Bearish":
        score -= 20
        factors.append("Bearish MACD signal - momentum fading")

    # Histogram strength
    if signals.macd_histogram > 0:
        score += 5
    else:
        score -= 5

    return max(0, min(100, score)), factors


def _score_rsi(signals: TechnicalSignals) -> tuple[float, list]:
    """Score RSI: momentum-zone scoring.

    Healthy buy zone: 40-65 (not overbought, has room to run)
    Oversold bounce: <30 (contrarian buy)
    Overbought risk: >75 (potential pullback)
    """
    rsi = signals.rsi
    factors = []

    if rsi <= 25:
        score = 85
        factors.append(f"RSI deeply oversold at {rsi:.0f} - strong reversal potential")
    elif rsi <= 35:
        score = 80
        factors.append(f"RSI oversold at {rsi:.0f} - buying opportunity")
    elif rsi <= 45:
        score = 75
        factors.append(f"RSI at {rsi:.0f} - approaching bullish momentum zone")
    elif rsi <= 55:
        score = 80  # Sweet spot - healthy momentum
        factors.append(f"RSI in optimal momentum zone ({rsi:.0f})")
    elif rsi <= 65:
        score = 70  # Still good - momentum building
    elif rsi <= 75:
        score = 50  # Getting warm but still OK in strong trends
    elif rsi <= 80:
        score = 30
        factors.append(f"RSI overbought at {rsi:.0f} - caution advised")
    else:
        score = 15
        factors.append(f"RSI extremely overbought ({rsi:.0f}) - high pullback risk")

    return score, factors


def _score_trend(signals: TechnicalSignals) -> tuple[float, list]:
    """Score trend: uptrend is a BUY signal, not just neutral."""
    score = 50
    factors = []

    if signals.trend == "Uptrend":
        score += 30
        factors.append("Confirmed uptrend - price above key moving averages")
    elif signals.trend == "Downtrend":
        score -= 25
        factors.append("Downtrend active - wait for reversal")

    if signals.ma_signal == "Golden Cross":
        score += 15
        factors.append("Golden Cross (SMA50 > SMA200) - major bullish signal")
    elif signals.ma_signal == "Death Cross":
        score -= 15

    return max(0, min(100, score)), factors


def _score_volume(signals: TechnicalSignals) -> tuple[float, list]:
    """Score volume: above-average volume confirms price moves."""
    ratio = signals.volume_ratio
    factors = []

    if ratio > 2.0:
        score = 90
        factors.append(f"Exceptional volume ({ratio:.1f}x avg) - strong market interest")
    elif ratio > 1.5:
        score = 80
        factors.append(f"High volume ({ratio:.1f}x avg) - move confirmed by volume")
    elif ratio > 1.1:
        score = 65
    elif ratio > 0.8:
        score = 50
    elif ratio > 0.5:
        score = 35
    else:
        score = 20
        factors.append("Very low volume - weak conviction in current price")

    return score, factors


def _score_breakout(signals: TechnicalSignals) -> tuple[float, list]:
    """Score breakout potential: Bollinger + 52-week range.

    Key insight: in an uptrend, touching upper Bollinger = strength (breakout),
    not weakness. Near 52w high in uptrend = momentum, not risk.
    """
    score = 50
    factors = []

    # Bollinger position - context-dependent
    if signals.bb_position == "Upper" and signals.trend == "Uptrend":
        score += 20
        factors.append("Breaking above upper Bollinger Band in uptrend (breakout)")
    elif signals.bb_position == "Upper" and signals.trend != "Uptrend":
        score -= 5  # mild concern without trend support
    elif signals.bb_position == "Lower" and signals.trend == "Downtrend":
        score -= 10  # falling in downtrend, not a bounce
    elif signals.bb_position == "Lower" and signals.trend != "Downtrend":
        score += 15
        factors.append("Price near lower Bollinger Band - potential bounce entry")
    elif signals.bb_position == "Middle":
        score += 5  # neutral

    # 52-week range: momentum-aware
    if signals.high_52w > signals.low_52w:
        range_52w = signals.high_52w - signals.low_52w
        position = (signals.price - signals.low_52w) / range_52w

        if position > 0.85 and signals.trend == "Uptrend":
            score += 15
            factors.append("Trading near 52-week high with strong uptrend (breakout candidate)")
        elif position > 0.7 and signals.trend == "Uptrend":
            score += 10
        elif position < 0.3 and signals.trend != "Downtrend":
            score += 10
            factors.append("Near 52-week low with improving technicals (value opportunity)")
        elif position < 0.3:
            score -= 5

    return max(0, min(100, score)), factors


def _score_volatility(signals: TechnicalSignals) -> float:
    """Score volatility: moderate is preferred for trading."""
    if signals.price <= 0:
        return 50

    atr_pct = (signals.atr / signals.price) * 100

    if atr_pct < 0.5:
        return 35
    elif atr_pct < 1.5:
        return 65
    elif atr_pct < 2.5:
        return 75  # Good trading volatility
    elif atr_pct < 4:
        return 55
    elif atr_pct < 6:
        return 35
    else:
        return 20


def _score_support_resistance(signals: TechnicalSignals) -> tuple[float, list]:
    """Score risk/reward based on distance to support and resistance."""
    if signals.support <= 0 or signals.price <= 0:
        return 50, []

    factors = []
    distance_to_support = (signals.price - signals.support) / signals.price
    distance_to_resistance = (signals.resistance - signals.price) / signals.price if signals.resistance > signals.price else 0

    # Risk/reward ratio
    if distance_to_resistance > 0 and distance_to_support > 0:
        rr_ratio = distance_to_resistance / distance_to_support
        if rr_ratio > 3:
            score = 90
            factors.append(f"Excellent risk/reward ratio ({rr_ratio:.1f}:1)")
        elif rr_ratio > 2:
            score = 75
            factors.append(f"Good risk/reward ratio ({rr_ratio:.1f}:1)")
        elif rr_ratio > 1:
            score = 60
        else:
            score = 40
    elif distance_to_support < 0.02:
        score = 70
        factors.append("Price at support level - low downside risk")
    else:
        score = 50

    return max(0, min(100, score)), factors


# ── Prediction & Explanation ──────────────────────────────────────


def _predict_move(signals: TechnicalSignals, score: float) -> float:
    """Predict expected % price move based on signals and score."""
    if signals.price <= 0 or signals.atr <= 0:
        return 0.0

    atr_pct = (signals.atr / signals.price) * 100

    # Base prediction from ATR
    if score >= 75:
        move = atr_pct * 2.5
    elif score >= 58:
        move = atr_pct * 1.5
    elif score >= 40:
        move = atr_pct * 0.5
    else:
        move = -atr_pct * 1.0

    # Trend multiplier
    if signals.trend == "Uptrend":
        move *= 1.3
    elif signals.trend == "Downtrend":
        move *= 0.5

    # MACD boost
    if signals.macd_crossover == "Bullish":
        move *= 1.2

    return round(move, 1)


def _generate_explanation(stock: StockInfo, signals: TechnicalSignals,
                          action: str, factors: list, predicted_move: float) -> str:
    """Generate a human-readable explanation with prediction."""
    name = stock.name

    if not factors:
        if action in ("Strong Buy", "Buy"):
            return f"{name} shows favorable technical signals pointing to upside potential of ~{predicted_move}%."
        elif action == "Avoid":
            return f"{name} shows weak technical signals. Wait for clearer buying opportunity."
        return f"{name} is consolidating. Monitor for a breakout or trend change before entering."

    # Build explanation from top factors
    top_factors = factors[:3]
    parts = " + ".join(top_factors)

    if action == "Strong Buy":
        verdict = (f"STRONG BUY: {name} shows exceptional setup with {parts}. "
                   f"Expected upside ~{predicted_move}% based on current momentum. "
                   f"The stock is displaying strong bullish convergence across multiple indicators.")
    elif action == "Buy":
        verdict = (f"BUY: {name} - {parts}. "
                   f"Technical analysis points to ~{predicted_move}% upside potential. "
                   f"Good entry opportunity at current levels.")
    elif action == "Hold":
        verdict = (f"{name}: {parts}. "
                   f"The stock shows mixed signals. Hold existing positions and watch for "
                   f"stronger confirmation before adding.")
    else:
        verdict = (f"AVOID: {name} - {parts}. "
                   f"Wait for better conditions before entering.")

    return verdict
