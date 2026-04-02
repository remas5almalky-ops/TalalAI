"""Bottom fishing scanner: finds beaten-down stocks with recovery potential.

Identifies stocks trading near their lows that show signs of stabilization
or early recovery - ideal for patient investors (1-6 month horizon).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from services import data_fetcher, technical_analysis
from services.cache import cache
from config import SHARIA_ONLY

logger = logging.getLogger(__name__)


@dataclass
class BottomStock:
    """A bottom-fishing candidate."""
    ticker: str = ""
    name: str = ""
    name_ar: str = ""
    sector: str = ""
    sharia: bool = True

    # Verdict
    rating: str = ""               # Gold / Silver / Bronze
    status: str = ""               # "Bottoming Out" / "Early Recovery" / "Still Falling"
    confidence: float = 0.0

    # Price context
    current_price: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    drop_from_high_pct: float = 0.0  # How far it fell
    upside_to_high_pct: float = 0.0  # Potential recovery
    position_52w: float = 0.0       # 0=at low, 100=at high

    # Recovery signals
    recovery_score: float = 0.0     # 0-100
    signals_bullish: list = field(default_factory=list)
    signals_bearish: list = field(default_factory=list)

    # Key indicators
    rsi: float = 50.0
    trend: str = "Sideways"
    macd_crossover: str = "Neutral"
    volume_ratio: float = 1.0
    sma_200: float = 0.0
    dist_from_sma200_pct: float = 0.0
    change_30d: float = 0.0

    # Trade plan
    entry_price: float = 0.0
    target_price: float = 0.0      # Conservative target
    target_price_full: float = 0.0  # Full recovery target
    stop_loss: float = 0.0
    profit_pct: float = 0.0        # To conservative target
    full_recovery_pct: float = 0.0 # To 52w high
    risk_pct: float = 0.0

    # Explanation
    explanation: str = ""
    buy_reason: str = ""           # One-line summary

    # Honest verdict (separate from rating)
    honest_verdict: str = ""       # "اشتري" / "انتظر" / "ابتعد"
    honest_verdict_en: str = ""    # "BUY" / "WAIT" / "STAY AWAY"
    honest_reason_ar: str = ""     # Arabic explanation
    honest_reason_en: str = ""     # English explanation
    honest_score: int = 0          # Raw honesty score
    honest_checklist: list = field(default_factory=list)  # What passed/failed


def scan_bottoms() -> list[BottomStock]:
    """Scan for bottom-fishing opportunities."""
    cached = cache.get("bottom_stocks")
    if cached is not None:
        return cached

    stocks = data_fetcher.load_stock_list()
    if SHARIA_ONLY:
        stocks = [s for s in stocks if s.get("sharia", False)]

    all_data = data_fetcher.fetch_multiple_stocks([s["ticker"] for s in stocks])

    candidates = []
    for stock_meta in stocks:
        ticker = stock_meta["ticker"]
        df = all_data.get(ticker)
        if df is None or df.empty or len(df) < 30:
            continue

        try:
            result = _analyze_bottom(ticker, stock_meta, df)
            if result:
                candidates.append(result)
        except Exception as e:
            logger.error(f"Error scanning bottom {ticker}: {e}")

    # Sort by honest verdict (Buy first), then by honest_score descending
    verdict_order = {"اشتري": 0, "تجميع": 1, "انتظر": 2, "تجنب": 3, "ابتعد!": 4}
    candidates.sort(key=lambda b: (verdict_order.get(b.honest_verdict, 5), -b.honest_score))

    cache.set("bottom_stocks", candidates, ttl=900)
    return candidates


def _analyze_bottom(ticker: str, meta: dict, df: pd.DataFrame) -> BottomStock | None:
    """Analyze if a stock is a bottom-fishing candidate."""
    signals = technical_analysis.analyze(df)
    close = df["Close"]
    price = float(close.iloc[-1])

    if price <= 0 or signals.high_52w <= signals.low_52w:
        return None

    # ── Basic calculations ──
    rng = signals.high_52w - signals.low_52w
    position_52w = (price - signals.low_52w) / rng * 100
    drop_from_high = (signals.high_52w - price) / signals.high_52w * 100
    upside_to_high = (signals.high_52w - price) / price * 100

    # Filter: must have dropped at least 15% from 52w high
    if drop_from_high < 15:
        return None

    # 30-day change
    p30 = float(close.iloc[-min(30, len(close))]) if len(close) >= 2 else price
    change_30d = (price - p30) / p30 * 100 if p30 > 0 else 0

    # Distance from SMA200
    dist_sma200 = ((price - signals.sma_200) / signals.sma_200 * 100) if signals.sma_200 > 0 else 0

    atr = signals.atr if signals.atr > 0 else price * 0.02

    b = BottomStock(
        ticker=ticker,
        name=meta["name"],
        name_ar=meta["name_ar"],
        sector=meta["sector"],
        sharia=meta.get("sharia", False),
        current_price=price,
        high_52w=signals.high_52w,
        low_52w=signals.low_52w,
        drop_from_high_pct=round(drop_from_high, 1),
        upside_to_high_pct=round(upside_to_high, 1),
        position_52w=round(position_52w, 1),
        rsi=signals.rsi,
        trend=signals.trend,
        macd_crossover=signals.macd_crossover,
        volume_ratio=signals.volume_ratio,
        sma_200=signals.sma_200,
        dist_from_sma200_pct=round(dist_sma200, 1),
        change_30d=round(change_30d, 1),
    )

    # ═══════════════════════════════════════════
    # RECOVERY SCORING
    # ═══════════════════════════════════════════

    score = 50
    bullish = []
    bearish = []

    # ── 1. How deep is the discount? (bigger drop = more upside) ──
    if drop_from_high > 40:
        score += 20
        bullish.append(f"Massive {drop_from_high:.0f}% drop from high - deep value territory")
    elif drop_from_high > 30:
        score += 15
        bullish.append(f"Major {drop_from_high:.0f}% correction - significant recovery potential")
    elif drop_from_high > 20:
        score += 10
        bullish.append(f"Solid {drop_from_high:.0f}% pullback from highs")
    else:
        score += 5

    # ── 2. Is it stabilizing? (not still crashing) ──
    # Look at last 10 days volatility vs prior
    if len(close) >= 20:
        recent_10 = close.tail(10)
        prior_10 = close.iloc[-20:-10]
        recent_range = (float(recent_10.max()) - float(recent_10.min())) / price * 100
        prior_range = (float(prior_10.max()) - float(prior_10.min())) / price * 100

        if recent_range < prior_range * 0.7:
            score += 12
            bullish.append("Price stabilizing - volatility contracting (base forming)")
        elif recent_range < prior_range:
            score += 6
            bullish.append("Showing signs of stabilization")

    # ── 3. RSI recovery from oversold ──
    if signals.rsi < 30:
        score += 15
        bullish.append(f"RSI deeply oversold ({signals.rsi:.0f}) - ready for bounce")
    elif signals.rsi < 40:
        score += 10
        bullish.append(f"RSI low at {signals.rsi:.0f} - selling pressure exhausting")
    elif signals.rsi < 50:
        score += 5
        bullish.append(f"RSI neutral ({signals.rsi:.0f}) - no longer in freefall")
    elif signals.rsi > 60:
        score += 3
        bullish.append("Momentum turning positive")

    # ── 4. MACD turning bullish ──
    if signals.macd_crossover == "Bullish":
        score += 12
        bullish.append("MACD bullish crossover - momentum shifting to buyers")
    elif signals.macd_histogram > 0:
        score += 5
        bullish.append("MACD histogram positive - selling pressure easing")
    else:
        score -= 5
        bearish.append("MACD still bearish - no momentum reversal yet")

    # ── 5. Volume analysis ──
    if signals.volume_ratio > 1.5 and change_30d > -5:
        score += 10
        bullish.append(f"Volume picking up ({signals.volume_ratio:.1f}x avg) - smart money accumulating")
    elif signals.volume_ratio > 1.0:
        score += 3

    # ── 6. Price vs SMA200 ──
    if dist_sma200 < -20:
        score += 10
        bullish.append(f"Trading {abs(dist_sma200):.0f}% below SMA200 - extreme discount")
    elif dist_sma200 < -10:
        score += 7
        bullish.append(f"{abs(dist_sma200):.0f}% below SMA200 - undervalued vs long-term avg")
    elif dist_sma200 < -3:
        score += 3
    elif dist_sma200 > 5:
        score -= 5  # Already recovered above SMA200

    # ── 7. Near 52-week low (deeper = more potential) ──
    if position_52w < 15:
        score += 12
        bullish.append(f"Trading near 52-week low ({position_52w:.0f}th percentile)")
    elif position_52w < 25:
        score += 8
        bullish.append(f"In lower quarter of 52-week range")
    elif position_52w < 40:
        score += 4

    # ── 8. Recent 30-day trend ──
    if change_30d < -15:
        score -= 8
        bearish.append(f"Still falling hard ({change_30d:+.1f}% in 30 days) - may not have bottomed")
    elif change_30d < -8:
        score -= 3
        bearish.append(f"Continued decline ({change_30d:+.1f}% in 30d) - wait for stabilization")
    elif -3 < change_30d < 3:
        score += 8
        bullish.append("Price flat over 30 days - forming a base")
    elif change_30d > 3 and change_30d < 15:
        score += 10
        bullish.append(f"Early recovery: up {change_30d:+.1f}% in 30 days from bottom")
    elif change_30d > 15:
        score -= 3
        bearish.append(f"Already bounced {change_30d:+.1f}% - may have missed the bottom")

    # ── 9. Stochastic ──
    if signals.stoch_k < 20:
        score += 8
        bullish.append("Stochastic oversold - reversal signal")
    elif signals.stoch_k < 35:
        score += 4

    # ── 10. Support holding ──
    dist_support = (price - signals.support) / price * 100 if signals.support > 0 else 10
    if dist_support < 2:
        score += 8
        bullish.append("Price sitting at support - holding above key level")

    # ═══════════════════════════════════════════
    # DETERMINE STATUS & RATING
    # ═══════════════════════════════════════════

    score = max(0, min(100, score))
    b.recovery_score = round(score, 1)
    b.signals_bullish = bullish
    b.signals_bearish = bearish

    # Status
    if change_30d < -10:
        b.status = "Still Falling"
    elif -3 < change_30d < 5 and signals.rsi < 50:
        b.status = "Bottoming Out"
    elif change_30d > 3 and signals.macd_crossover == "Bullish":
        b.status = "Early Recovery"
    elif signals.trend == "Uptrend":
        b.status = "Early Recovery"
    else:
        b.status = "Bottoming Out"

    # Rating
    if score >= 75 and b.status != "Still Falling":
        b.rating = "Gold"
        b.confidence = round(min(90, score), 1)
    elif score >= 60:
        b.rating = "Silver"
        b.confidence = round(min(75, score), 1)
    elif score >= 45:
        b.rating = "Bronze"
        b.confidence = round(min(60, score), 1)
    else:
        return None  # Not worth showing

    # ═══════════════════════════════════════════
    # TRADE PLAN (long-term patient investor)
    # ═══════════════════════════════════════════

    b.entry_price = round(price, 2)

    # Conservative target: halfway recovery to 52w high
    mid_recovery = price + (signals.high_52w - price) * 0.5
    b.target_price = round(mid_recovery, 2)
    b.profit_pct = round((b.target_price - price) / price * 100, 1)

    # Full recovery target
    b.target_price_full = round(signals.high_52w * 0.97, 2)
    b.full_recovery_pct = round((b.target_price_full - price) / price * 100, 1)

    # Stop loss: below recent low or support
    recent_low = float(df.tail(20)["Low"].min())
    b.stop_loss = round(min(recent_low * 0.97, price - atr * 2.5), 2)
    if b.stop_loss <= 0:
        b.stop_loss = round(price * 0.88, 2)

    b.risk_pct = round((price - b.stop_loss) / price * 100, 1)

    # ── Explanation ──
    b.explanation = _build_explanation(b)
    if bullish:
        b.buy_reason = bullish[0]

    # ═══════════════════════════════════════════
    # HONEST VERDICT - The real advice
    # ═══════════════════════════════════════════
    _compute_honest_verdict(b, signals, change_30d)

    return b


def _build_explanation(b: BottomStock) -> str:
    """Build explanation for the bottom-fishing candidate."""
    status_text = {
        "Still Falling": "still declining but nearing potential support",
        "Bottoming Out": "showing signs of forming a bottom",
        "Early Recovery": "showing early signs of recovery from its lows",
    }

    text = (f"{b.name} has dropped {b.drop_from_high_pct:.0f}% from its 52-week high "
            f"and is {status_text.get(b.status, 'near its lows')}. ")

    if b.signals_bullish:
        text += f"{b.signals_bullish[0]}. "

    text += (f"Conservative target at {b.target_price:.2f} SAR (+{b.profit_pct:.0f}%), "
             f"full recovery to {b.target_price_full:.2f} SAR (+{b.full_recovery_pct:.0f}%). "
             f"Patience required - this is a 1-6 month play.")

    if b.status == "Still Falling":
        text += " Consider waiting for stabilization before entering."

    return text


def _compute_honest_verdict(b: BottomStock, signals, change_30d: float):
    """Compute brutally honest verdict with checklist.

    The 4 Golden Rules for bottom fishing:
    1. MACD must be bullish (momentum turning)
    2. Trend must NOT be Downtrend (stopped falling)
    3. RSI between 35-60 (not extreme)
    4. 30-day price stable or recovering (not crashing)
    """
    checklist = []
    h_score = 0

    # Rule 1: MACD Bullish?
    if b.macd_crossover == "Bullish":
        h_score += 3
        checklist.append({"rule": "MACD صاعد (الزخم تحول)", "rule_en": "MACD Bullish (momentum turning)", "pass": True})
    else:
        h_score -= 2
        checklist.append({"rule": "MACD لسه سلبي", "rule_en": "MACD still bearish", "pass": False})

    # Rule 2: Trend not Downtrend?
    if b.trend == "Uptrend":
        h_score += 3
        checklist.append({"rule": "الترند صاعد (Uptrend)", "rule_en": "Uptrend confirmed", "pass": True})
    elif b.trend == "Sideways":
        h_score += 1
        checklist.append({"rule": "الترند جانبي - وقف النزول", "rule_en": "Sideways - stopped falling", "pass": True})
    else:
        h_score -= 2
        checklist.append({"rule": "الترند نازل (Downtrend) - خطر!", "rule_en": "Downtrend active - DANGER", "pass": False})

    # Rule 3: RSI healthy zone?
    if 35 <= b.rsi <= 60:
        h_score += 2
        checklist.append({"rule": f"RSI مثالي ({b.rsi:.0f}) - منطقة صحية", "rule_en": f"RSI healthy at {b.rsi:.0f}", "pass": True})
    elif b.rsi < 35:
        h_score += 1
        checklist.append({"rule": f"RSI منخفض جداً ({b.rsi:.0f}) - ممكن يرتد بس خطر", "rule_en": f"RSI very low ({b.rsi:.0f}) - may bounce but risky", "pass": True})
    elif b.rsi > 65:
        h_score -= 1
        checklist.append({"rule": f"RSI مرتفع ({b.rsi:.0f}) - فاتك جزء من الحركة", "rule_en": f"RSI elevated ({b.rsi:.0f}) - missed part of the move", "pass": False})
    else:
        h_score += 2
        checklist.append({"rule": f"RSI مقبول ({b.rsi:.0f})", "rule_en": f"RSI acceptable ({b.rsi:.0f})", "pass": True})

    # Rule 4: 30-day price action?
    if change_30d > 15:
        h_score -= 1
        checklist.append({"rule": f"ارتفع {change_30d:+.1f}% بالشهر - فاتك القاع!", "rule_en": f"Already bounced {change_30d:+.1f}% - missed the bottom", "pass": False})
    elif change_30d > 3:
        h_score += 2
        checklist.append({"rule": f"بداية تعافي ({change_30d:+.1f}% بالشهر)", "rule_en": f"Early recovery ({change_30d:+.1f}% in 30d)", "pass": True})
    elif change_30d > -3:
        h_score += 1
        checklist.append({"rule": "السعر مستقر - يكوّن قاع", "rule_en": "Price stable - forming base", "pass": True})
    elif change_30d > -8:
        h_score -= 1
        checklist.append({"rule": f"لسه ينزل ({change_30d:+.1f}% بالشهر)", "rule_en": f"Still declining ({change_30d:+.1f}% in 30d)", "pass": False})
    else:
        h_score -= 3
        checklist.append({"rule": f"ينهار {change_30d:+.1f}% بالشهر - سكين ساقط!", "rule_en": f"Crashing {change_30d:+.1f}% - falling knife!", "pass": False})

    # Bonus: below SMA200?
    if b.dist_from_sma200_pct < -15:
        h_score += 1
        checklist.append({"rule": f"تحت SMA200 بـ{abs(b.dist_from_sma200_pct):.0f}% - خصم كبير", "rule_en": f"{abs(b.dist_from_sma200_pct):.0f}% below SMA200 - deep discount", "pass": True})
    elif b.dist_from_sma200_pct < -5:
        h_score += 1
        checklist.append({"rule": f"تحت SMA200 بـ{abs(b.dist_from_sma200_pct):.0f}%", "rule_en": f"{abs(b.dist_from_sma200_pct):.0f}% below SMA200", "pass": True})
    elif b.dist_from_sma200_pct > 5:
        h_score -= 1
        checklist.append({"rule": f"فوق SMA200 بـ{b.dist_from_sma200_pct:.0f}% - مو رخيص", "rule_en": f"{b.dist_from_sma200_pct:.0f}% above SMA200 - not cheap", "pass": False})

    # Bonus: big upside?
    if b.upside_to_high_pct > 30:
        h_score += 1
        checklist.append({"rule": f"فرصة ارتفاع +{b.upside_to_high_pct:.0f}% للقمة", "rule_en": f"+{b.upside_to_high_pct:.0f}% upside to 52w high", "pass": True})

    b.honest_score = h_score
    b.honest_checklist = checklist

    # ── Final Verdict ──
    passed = sum(1 for c in checklist if c["pass"])
    failed = sum(1 for c in checklist if not c["pass"])

    if h_score >= 6:
        b.honest_verdict = "اشتري"
        b.honest_verdict_en = "BUY"
        b.honest_reason_ar = "كل المؤشرات إيجابية - فرصة حقيقية للتعافي"
        b.honest_reason_en = "All indicators positive - real recovery opportunity"
    elif h_score >= 4:
        b.honest_verdict = "تجميع"
        b.honest_verdict_en = "ACCUMULATE"
        b.honest_reason_ar = "إشارات مختلطة - ادخل بجزء وانتظر تأكيد"
        b.honest_reason_en = "Mixed signals - enter partial position and wait for confirmation"
    elif h_score >= 2:
        b.honest_verdict = "انتظر"
        b.honest_verdict_en = "WAIT"
        b.honest_reason_ar = "فيه إمكانية بس التوقيت مو مثالي - راقب السهم"
        b.honest_reason_en = "Has potential but timing not ideal - monitor the stock"
    elif h_score >= 0:
        b.honest_verdict = "تجنب"
        b.honest_verdict_en = "AVOID"
        b.honest_reason_ar = "إشارات سلبية أكثر من الإيجابية - لا تدخل الآن"
        b.honest_reason_en = "More negative signals than positive - don't enter now"
    else:
        b.honest_verdict = "ابتعد!"
        b.honest_verdict_en = "STAY AWAY"
        b.honest_reason_ar = "سكين ساقط! مو لأنه رخيص يعني بيرتفع"
        b.honest_reason_en = "Falling knife! Cheap doesn't mean it will recover"
