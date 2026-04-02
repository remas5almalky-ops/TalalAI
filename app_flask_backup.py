"""Saudi Stock Recommendation System - Flask Application."""

import logging
from flask import Flask, render_template, request, jsonify

from services import stock_screener, data_fetcher, fundamental_analysis, swing_analyzer, opportunity_scanner, bottom_scanner
from services.cache import cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def dashboard():
    """Main dashboard with top stock picks."""
    try:
        top_picks = stock_screener.get_top_picks()
        summary = stock_screener.get_market_summary()
        sectors = data_fetcher.get_sectors()
        return render_template("dashboard.html", top_picks=top_picks, summary=summary, sectors=sectors)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template("dashboard.html", top_picks=[], summary={}, sectors=[], error=str(e))


@app.route("/stock/<ticker>")
def stock_detail(ticker):
    """Individual stock analysis page."""
    try:
        rec = stock_screener.analyze_single_stock(ticker)
        if rec is None:
            return render_template("stock_detail.html", rec=None, error=f"No data found for {ticker}")

        # Get historical price data for chart
        df = data_fetcher.fetch_stock_data(ticker)
        chart_data = []
        if not df.empty:
            for date, row in df.tail(90).iterrows():
                chart_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })

        return render_template("stock_detail.html", rec=rec, chart_data=chart_data)
    except Exception as e:
        logger.error(f"Stock detail error for {ticker}: {e}")
        return render_template("stock_detail.html", rec=None, error=str(e))


@app.route("/search")
def search_page():
    """Search and filter page."""
    sectors = data_fetcher.get_sectors()
    return render_template("search.html", sectors=sectors)


@app.route("/api/top-picks")
def api_top_picks():
    """API: Get top stock picks as HTML partial."""
    top_picks = stock_screener.get_top_picks()
    return render_template("partials/stock_cards.html", recommendations=top_picks)


@app.route("/api/search")
def api_search():
    """API: Search and filter stocks, returns HTML partial."""
    query = request.args.get("q", "").strip()
    sector = request.args.get("sector", "").strip() or None
    investment_type = request.args.get("type", "").strip() or None
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    action = request.args.get("action", "").strip() or None

    if query:
        results = stock_screener.search_stocks(query)
        # Apply additional filters to search results
        if sector:
            results = [r for r in results if r.stock.sector == sector]
        if min_price is not None:
            results = [r for r in results if r.stock.current_price >= min_price]
        if max_price is not None:
            results = [r for r in results if r.stock.current_price <= max_price]
    else:
        results = stock_screener.filter_stocks(
            sector=sector,
            investment_type=investment_type,
            min_price=min_price,
            max_price=max_price,
            action=action,
        )

    return render_template("partials/stock_table.html", recommendations=results)


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    """API: Get stock analysis as JSON."""
    rec = stock_screener.analyze_single_stock(ticker)
    if rec is None:
        return jsonify({"error": f"No data for {ticker}"}), 404

    return jsonify({
        "ticker": rec.stock.ticker,
        "name": rec.stock.name,
        "action": rec.action,
        "confidence": rec.confidence,
        "opportunity_score": rec.opportunity_score,
        "entry_price": rec.entry_price,
        "target_price": rec.target_price,
        "stop_loss": rec.stop_loss,
        "trend": rec.signals.trend,
        "explanation": rec.explanation,
    })


@app.route("/report/<ticker>")
def stock_report(ticker):
    """Comprehensive deep analysis report for a stock."""
    try:
        rec = stock_screener.analyze_single_stock(ticker)
        deep = fundamental_analysis.get_deep_analysis(ticker)

        if rec is None or deep is None:
            return render_template("report.html", rec=None, deep=None,
                                   error=f"Could not generate report for {ticker}")

        # Chart data
        df = data_fetcher.fetch_stock_data(ticker)
        chart_data = []
        if not df.empty:
            for date, row in df.tail(120).iterrows():
                chart_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })

        return render_template("report.html", rec=rec, deep=deep,
                               chart_data=chart_data, ticker=ticker)
    except Exception as e:
        logger.error(f"Report error for {ticker}: {e}")
        return render_template("report.html", rec=None, deep=None, error=str(e))


@app.route("/bottom-fishing")
def bottom_fishing_page():
    """Bottom fishing: beaten-down stocks with recovery potential."""
    try:
        candidates = bottom_scanner.scan_bottoms()
        gold = [c for c in candidates if c.rating == "Gold"]
        silver = [c for c in candidates if c.rating == "Silver"]
        bronze = [c for c in candidates if c.rating == "Bronze"]
        return render_template("bottom_fishing.html",
                               gold=gold, silver=silver, bronze=bronze,
                               total=len(candidates))
    except Exception as e:
        logger.error(f"Bottom fishing error: {e}")
        return render_template("bottom_fishing.html",
                               gold=[], silver=[], bronze=[], total=0, error=str(e))


@app.route("/opportunities")
def opportunities_page():
    """High-profit 10%+ opportunity scanner page."""
    try:
        opps = opportunity_scanner.scan_opportunities()
        buy_opps = [o for o in opps if o.action == "Buy"]
        watch_opps = [o for o in opps if o.action == "Watch"]
        return render_template("opportunities.html",
                               buy_opps=buy_opps, watch_opps=watch_opps,
                               total=len(opps))
    except Exception as e:
        logger.error(f"Opportunities error: {e}")
        return render_template("opportunities.html",
                               buy_opps=[], watch_opps=[], total=0, error=str(e))


@app.route("/swing")
def swing_page():
    """Short-term swing trade analyzer page."""
    return render_template("swing.html")


@app.route("/api/swing")
def api_swing():
    """API: Swing trade analysis for a stock. Returns HTML partial."""
    query = request.args.get("q", "").strip()
    if not query:
        return "<p class='text-gray-400 text-sm text-center py-8'>Enter a stock name or ticker above</p>"

    ticker = _resolve_ticker(query)
    if not ticker:
        return render_template("partials/lookup_not_found.html", query=query)

    result = swing_analyzer.analyze_swing(ticker)
    if result is None:
        return render_template("partials/lookup_not_found.html", query=query)

    # Sparkline data
    df = data_fetcher.fetch_stock_data(ticker)
    chart_data = []
    if not df.empty:
        for date, row in df.tail(20).iterrows():
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
            })

    return render_template("partials/swing_result.html", s=result, chart_data=chart_data)


@app.route("/api/suggest")
def api_suggest():
    """API: Autocomplete suggestions from local stock list. No yfinance call needed."""
    query = request.args.get("q", "").strip().lower()
    if len(query) < 1:
        return jsonify([])

    stocks = data_fetcher.load_stock_list()
    matches = []
    for s in stocks:
        ticker_clean = s["ticker"].replace(".SR", "")
        if (query in s["ticker"].lower()
                or query in ticker_clean.lower()
                or query in s["name"].lower()
                or query in s["name_ar"]):
            matches.append({
                "ticker": s["ticker"],
                "name": s["name"],
                "name_ar": s["name_ar"],
                "sector": s["sector"],
            })
        if len(matches) >= 8:
            break

    return jsonify(matches)


@app.route("/api/lookup")
def api_lookup():
    """API: Instant recommendation for a single stock. Returns rich HTML partial."""
    query = request.args.get("q", "").strip()
    if not query:
        return "<p class='text-gray-400 text-sm text-center py-8'>Enter a stock name or ticker above</p>"

    # Resolve query to a ticker
    ticker = _resolve_ticker(query)
    if not ticker:
        return render_template("partials/lookup_not_found.html", query=query)

    rec = stock_screener.analyze_single_stock(ticker)
    if rec is None:
        return render_template("partials/lookup_not_found.html", query=query)

    # Get mini chart data (last 30 days for sparkline)
    df = data_fetcher.fetch_stock_data(ticker)
    chart_data = []
    if not df.empty:
        for date, row in df.tail(30).iterrows():
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
            })

    return render_template("partials/lookup_result.html", rec=rec, chart_data=chart_data)


def _resolve_ticker(query: str) -> str | None:
    """Resolve a user query (name, ticker, partial) into a full .SR ticker."""
    query_lower = query.lower().strip()
    stocks = data_fetcher.load_stock_list()

    # Exact ticker match (with or without .SR)
    for s in stocks:
        if s["ticker"].lower() == query_lower or s["ticker"].lower() == query_lower + ".sr":
            return s["ticker"]

    # Exact name match
    for s in stocks:
        if s["name"].lower() == query_lower or s["name_ar"] == query:
            return s["ticker"]

    # Partial match - prefer ticker prefix, then name contains
    for s in stocks:
        ticker_num = s["ticker"].replace(".SR", "")
        if ticker_num == query_lower:
            return s["ticker"]

    for s in stocks:
        if query_lower in s["name"].lower() or query_lower in s["name_ar"]:
            return s["ticker"]

    for s in stocks:
        if query_lower in s["ticker"].lower():
            return s["ticker"]

    return None


@app.route("/api/refresh")
def api_refresh():
    """Clear cache and trigger fresh analysis."""
    cache.clear()
    return jsonify({"status": "Cache cleared. Data will refresh on next request."})


@app.route("/api/summary")
def api_summary():
    """API: Market summary as JSON."""
    summary = stock_screener.get_market_summary()
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
