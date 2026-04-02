"""Stock screener: orchestrates fetch -> analyze -> score -> rank -> filter."""

import logging
from typing import Optional

from models.stock import StockInfo, Recommendation
from services import data_fetcher, technical_analysis, scoring_engine
from services.cache import cache
from config import TOP_PICKS_COUNT, MAX_PER_SECTOR, SHARIA_ONLY

logger = logging.getLogger(__name__)


def get_top_picks(count: int = TOP_PICKS_COUNT) -> list[Recommendation]:
    """Get top recommended stocks for today."""
    cached = cache.get("top_picks")
    if cached is not None:
        return cached[:count]

    all_recs = analyze_all_stocks()
    # Sort by opportunity score descending
    all_recs.sort(key=lambda r: r.opportunity_score, reverse=True)

    # Diversify: max N per sector
    top_picks = []
    sector_count = {}
    for rec in all_recs:
        sector = rec.stock.sector
        if sector_count.get(sector, 0) < MAX_PER_SECTOR:
            top_picks.append(rec)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(top_picks) >= count:
            break

    cache.set("top_picks", top_picks, ttl=900)
    return top_picks


def analyze_all_stocks() -> list[Recommendation]:
    """Analyze all stocks and return recommendations."""
    cached = cache.get("all_recommendations")
    if cached is not None:
        return cached

    stocks = data_fetcher.load_stock_list()
    if SHARIA_ONLY:
        stocks = [s for s in stocks if s.get("sharia", False)]
    tickers = [s["ticker"] for s in stocks]

    # Fetch all data
    data = data_fetcher.fetch_multiple_stocks(tickers)

    recommendations = []
    for stock_meta in stocks:
        ticker = stock_meta["ticker"]
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        try:
            # Build StockInfo
            stock_info = StockInfo(
                ticker=ticker,
                name=stock_meta["name"],
                name_ar=stock_meta["name_ar"],
                sector=stock_meta["sector"],
                sector_ar=stock_meta["sector_ar"],
                current_price=float(df["Close"].iloc[-1]),
                volume=int(df["Volume"].iloc[-1]),
                sharia=stock_meta.get("sharia", False),
                sharia_note=stock_meta.get("sharia_note", ""),
            )

            # Calculate change percentage
            if len(df) >= 2:
                prev_close = float(df["Close"].iloc[-2])
                if prev_close > 0:
                    stock_info.change_pct = round(
                        ((stock_info.current_price - prev_close) / prev_close) * 100, 2
                    )

            # Technical analysis
            signals = technical_analysis.analyze(df)

            # Score and recommend
            rec = scoring_engine.score_stock(stock_info, signals)
            recommendations.append(rec)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            continue

    cache.set("all_recommendations", recommendations, ttl=900)
    return recommendations


def analyze_single_stock(ticker: str) -> Optional[Recommendation]:
    """Analyze a single stock and return its recommendation."""
    # Check if already in cache
    all_recs = cache.get("all_recommendations")
    if all_recs:
        for rec in all_recs:
            if rec.stock.ticker == ticker:
                return rec

    # Fetch and analyze individually
    meta = data_fetcher.get_stock_meta(ticker)
    df = data_fetcher.fetch_stock_data(ticker)

    if df.empty:
        return None

    stock_info = StockInfo(
        ticker=ticker,
        name=meta["name"],
        name_ar=meta["name_ar"],
        sector=meta["sector"],
        sector_ar=meta["sector_ar"],
        current_price=float(df["Close"].iloc[-1]),
        volume=int(df["Volume"].iloc[-1]),
        sharia=meta.get("sharia", False),
        sharia_note=meta.get("sharia_note", ""),
    )

    if len(df) >= 2:
        prev_close = float(df["Close"].iloc[-2])
        if prev_close > 0:
            stock_info.change_pct = round(
                ((stock_info.current_price - prev_close) / prev_close) * 100, 2
            )

    signals = technical_analysis.analyze(df)
    return scoring_engine.score_stock(stock_info, signals)


def filter_stocks(
    sector: Optional[str] = None,
    investment_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    action: Optional[str] = None,
) -> list[Recommendation]:
    """Filter stocks based on criteria."""
    all_recs = analyze_all_stocks()
    filtered = []

    for rec in all_recs:
        # Sector filter
        if sector and rec.stock.sector != sector:
            continue

        # Price range filter
        if min_price is not None and rec.stock.current_price < min_price:
            continue
        if max_price is not None and rec.stock.current_price > max_price:
            continue

        # Investment type filter
        if investment_type == "short" and not rec.short_term:
            continue
        if investment_type == "long" and not rec.long_term:
            continue

        # Action filter
        if action == "Buy" and rec.action not in ("Buy", "Strong Buy"):
            continue
        elif action and action != "Buy" and rec.action != action:
            continue

        filtered.append(rec)

    filtered.sort(key=lambda r: r.opportunity_score, reverse=True)
    return filtered


def search_stocks(query: str) -> list[Recommendation]:
    """Search stocks by name or ticker."""
    all_recs = analyze_all_stocks()
    query_lower = query.lower().strip()

    results = []
    for rec in all_recs:
        if (query_lower in rec.stock.ticker.lower() or
            query_lower in rec.stock.name.lower() or
            query_lower in rec.stock.name_ar):
            results.append(rec)

    results.sort(key=lambda r: r.opportunity_score, reverse=True)
    return results


def get_market_summary() -> dict:
    """Get a summary of overall market conditions."""
    all_recs = analyze_all_stocks()
    if not all_recs:
        return {"total": 0, "buy": 0, "hold": 0, "avoid": 0, "avg_score": 0, "mood": "Unknown"}

    buy_count = sum(1 for r in all_recs if r.action == "Buy")
    hold_count = sum(1 for r in all_recs if r.action == "Hold")
    avoid_count = sum(1 for r in all_recs if r.action == "Avoid")
    avg_score = round(sum(r.opportunity_score for r in all_recs) / len(all_recs), 1)

    # Market mood
    buy_ratio = buy_count / len(all_recs) if all_recs else 0
    if buy_ratio > 0.5:
        mood = "Bullish"
    elif buy_ratio > 0.3:
        mood = "Moderately Bullish"
    elif avoid_count / len(all_recs) > 0.5:
        mood = "Bearish"
    else:
        mood = "Neutral"

    # Sector breakdown
    sectors = {}
    for rec in all_recs:
        s = rec.stock.sector
        if s not in sectors:
            sectors[s] = {"buy": 0, "hold": 0, "avoid": 0, "avg_score": 0, "scores": []}
        sectors[s][rec.action.lower()] += 1
        sectors[s]["scores"].append(rec.opportunity_score)

    for s in sectors:
        scores = sectors[s].pop("scores")
        sectors[s]["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0

    return {
        "total": len(all_recs),
        "buy": buy_count,
        "hold": hold_count,
        "avoid": avoid_count,
        "avg_score": avg_score,
        "mood": mood,
        "sectors": sectors,
    }
