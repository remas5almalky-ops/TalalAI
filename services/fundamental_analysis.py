"""Fundamental analysis service - deep company analysis with financials, competitors, executives."""

import logging
from typing import Optional

import yfinance as yf

from services.cache import cache

logger = logging.getLogger(__name__)


# ── Known competitor mappings for Tadawul stocks ──────────────────

COMPETITOR_MAP = {
    "2222.SR": {
        "peers": [
            {"ticker": "SHEL", "name": "Shell (UK)", "market": "Global"},
            {"ticker": "XOM", "name": "ExxonMobil (US)", "market": "Global"},
            {"ticker": "CVX", "name": "Chevron (US)", "market": "Global"},
            {"ticker": "TTE", "name": "TotalEnergies (France)", "market": "Global"},
            {"ticker": "BP", "name": "BP (UK)", "market": "Global"},
            {"ticker": "2010.SR", "name": "SABIC", "market": "Tadawul"},
        ],
        "supply_chain": {
            "overview": "Aramco's operations are largely vertically integrated with domestic supply. However, key exposures include:",
            "china_exposure": "HIGH - China is Aramco's largest customer (~18% of crude exports). Aramco has invested heavily in Chinese refining (e.g., joint ventures in Fujian, Yunnan). Trade tensions or Chinese economic slowdown directly impact demand.",
            "risks": [
                "CHINA DEMAND RISK: China accounts for ~18% of crude exports. Any slowdown, trade war escalation, or shift to EVs/renewables in China directly threatens revenue",
                "GLOBAL TRADE RISK: Tariff wars can reduce global oil demand and economic growth, pressuring prices",
                "OPEC+ POLICY: Production cuts limit revenue volume even when prices rise",
                "EQUIPMENT: Some specialized drilling/refining equipment sourced from Western suppliers; sanctions risk is minimal for Saudi Arabia but supply chain disruptions possible",
                "RENEWABLE TRANSITION: Long-term structural threat as global energy mix shifts",
            ],
        },
    },
    "1120.SR": {
        "peers": [
            {"ticker": "1180.SR", "name": "Al Inma Bank", "market": "Tadawul"},
            {"ticker": "1010.SR", "name": "Riyad Bank", "market": "Tadawul"},
            {"ticker": "1140.SR", "name": "Bank AlBilad", "market": "Tadawul"},
            {"ticker": "1050.SR", "name": "BSF", "market": "Tadawul"},
            {"ticker": "1111.SR", "name": "SAB", "market": "Tadawul"},
        ],
        "supply_chain": {
            "overview": "Banking sector has minimal physical supply chain exposure.",
            "china_exposure": "LOW - Limited direct China exposure. Indirect risk through Saudi economic ties to Chinese demand for oil.",
            "risks": [
                "INTEREST RATE RISK: Saudi rates follow US Fed; rate cuts reduce net interest margins",
                "REAL ESTATE EXPOSURE: Heavy mortgage lending; property market correction risk",
                "VISION 2030: Government spending shifts could affect lending demand",
                "FINTECH: Growing competition from digital banks and payment platforms",
            ],
        },
    },
}

# Default for stocks without specific competitor data
DEFAULT_SUPPLY_CHAIN = {
    "overview": "Supply chain analysis based on sector characteristics.",
    "china_exposure": "MODERATE - Saudi economy's dependence on oil exports to China creates indirect exposure for all Tadawul-listed companies.",
    "risks": [
        "CHINA TRADE: Indirect exposure through Saudi oil revenue dependency",
        "GLOBAL SUPPLY CHAINS: Post-COVID disruptions and trade tensions affect imported materials",
        "CURRENCY: SAR pegged to USD; strong dollar impacts competitiveness",
    ],
}


def get_deep_analysis(ticker: str) -> Optional[dict]:
    """Get comprehensive fundamental analysis for a stock."""
    cache_key = f"deep_{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or not info.get("currentPrice"):
            return None

        analysis = {}

        # ── Company Overview ──
        analysis["overview"] = {
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", ""),
            "country": info.get("country", "Saudi Arabia"),
            "city": info.get("city", ""),
            "employees": info.get("fullTimeEmployees", 0),
            "website": info.get("website", ""),
        }

        # ── Key Metrics ──
        analysis["metrics"] = {
            "price": info.get("currentPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "trailing_pe": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "peg_ratio": info.get("pegRatio", 0),
            "price_to_book": info.get("priceToBook", 0),
            "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
            "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
            "ev_to_revenue": info.get("enterpriseToRevenue", 0),
            "beta": info.get("beta", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }

        # ── Financials ──
        analysis["financials"] = {
            "revenue": info.get("totalRevenue", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "gross_margins": info.get("grossMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "profit_margins": info.get("profitMargins", 0),
            "ebitda": info.get("ebitda", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            "eps_trailing": info.get("epsTrailingTwelveMonths", 0),
            "eps_forward": info.get("epsForward", 0),
            "eps_current_year": info.get("epsCurrentYear", 0),
            "roe": info.get("returnOnEquity", 0),
            "roa": info.get("returnOnAssets", 0),
            "free_cash_flow": info.get("freeCashflow", 0),
        }

        # ── Balance Sheet Health ──
        analysis["balance_sheet"] = {
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio": info.get("currentRatio", 0),
            "quick_ratio": info.get("quickRatio", 0),
            "book_value": info.get("bookValue", 0),
        }

        # ── Dividend ──
        analysis["dividend"] = {
            "rate": info.get("dividendRate", 0),
            "yield": info.get("dividendYield", 0),
            "payout_ratio": info.get("payoutRatio", 0),
            "five_year_avg_yield": info.get("fiveYearAvgDividendYield", 0),
        }

        # ── Analyst Ratings ──
        analysis["analyst"] = {
            "rating": info.get("averageAnalystRating", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "target_mean": info.get("targetMeanPrice", 0),
            "target_high": info.get("targetHighPrice", 0),
            "target_low": info.get("targetLowPrice", 0),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
        }

        # ── Executives ──
        officers = info.get("companyOfficers", [])
        analysis["executives"] = []
        for officer in officers:
            analysis["executives"].append({
                "name": officer.get("name", ""),
                "title": officer.get("title", ""),
                "age": officer.get("age"),
                "year_born": officer.get("yearBorn"),
            })

        # ── Income Statement (multi-year) ──
        analysis["income_history"] = _get_income_history(stock)

        # ── Cash Flow ──
        analysis["cash_flow"] = _get_cash_flow(stock)

        # ── Competitors & Supply Chain ──
        comp_data = COMPETITOR_MAP.get(ticker, {})
        analysis["competitors"] = comp_data.get("peers", [])
        analysis["supply_chain"] = comp_data.get("supply_chain", DEFAULT_SUPPLY_CHAIN)

        # Fetch competitor metrics for comparison
        analysis["competitor_comparison"] = _fetch_competitor_metrics(comp_data.get("peers", []))

        # ── Institutional Holders ──
        analysis["ownership"] = {
            "insiders_pct": info.get("heldPercentInsiders", 0),
            "institutions_pct": info.get("heldPercentInstitutions", 0),
        }

        cache.set(cache_key, analysis, ttl=1800)
        return analysis

    except Exception as e:
        logger.error(f"Error in deep analysis for {ticker}: {e}")
        return None


def _get_income_history(stock) -> list[dict]:
    """Get multi-year income statement data."""
    try:
        fs = stock.financials
        if fs is None or fs.empty:
            return []

        history = []
        for col in fs.columns[:4]:  # Last 4 years
            year_data = {
                "year": col.strftime("%Y") if hasattr(col, "strftime") else str(col),
                "revenue": _safe_val(fs, "Total Revenue", col),
                "cost_of_revenue": _safe_val(fs, "Reconciled Cost Of Revenue", col),
                "gross_profit": _safe_val(fs, "Gross Profit", col),
                "operating_income": _safe_val(fs, "Total Operating Income As Reported", col),
                "net_income": _safe_val(fs, "Net Income From Continuing Operation Net Minority Interest", col),
                "ebitda": _safe_val(fs, "EBITDA", col),
                "eps": _safe_val(fs, "Basic EPS", col),
                "total_expenses": _safe_val(fs, "Total Expenses", col),
            }
            history.append(year_data)
        return history
    except Exception as e:
        logger.error(f"Error getting income history: {e}")
        return []


def _get_cash_flow(stock) -> dict:
    """Get latest cash flow data."""
    try:
        cf = stock.cashflow
        if cf is None or cf.empty:
            return {}

        col = cf.columns[0]
        return {
            "operating": _safe_val(cf, "Operating Cash Flow", col),
            "investing": _safe_val(cf, "Investing Cash Flow", col),
            "financing": _safe_val(cf, "Financing Cash Flow", col),
            "capex": _safe_val(cf, "Capital Expenditure", col),
            "free_cash_flow": _safe_val(cf, "Free Cash Flow", col),
            "dividends_paid": _safe_val(cf, "Common Stock Dividend Paid", col),
            "debt_repayment": _safe_val(cf, "Repayment Of Debt", col),
            "debt_issuance": _safe_val(cf, "Issuance Of Debt", col),
        }
    except Exception as e:
        logger.error(f"Error getting cash flow: {e}")
        return {}


def _fetch_competitor_metrics(peers: list[dict]) -> list[dict]:
    """Fetch key metrics for competitor comparison."""
    results = []
    for peer in peers[:5]:  # Limit to 5 competitors
        try:
            ticker = peer["ticker"]
            cache_key = f"comp_metrics_{ticker}"
            cached = cache.get(cache_key)
            if cached:
                results.append(cached)
                continue

            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                continue

            data = {
                "ticker": ticker,
                "name": peer["name"],
                "market": peer.get("market", ""),
                "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "trailing_pe": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "profit_margins": info.get("profitMargins", 0),
                "operating_margins": info.get("operatingMargins", 0),
                "roe": info.get("returnOnEquity", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
                "currency": info.get("currency", ""),
            }
            cache.set(cache_key, data, ttl=1800)
            results.append(data)
        except Exception as e:
            logger.warning(f"Could not fetch metrics for {peer['ticker']}: {e}")
            continue

    return results


def _safe_val(df, row_name, col):
    """Safely extract a value from a DataFrame."""
    try:
        if row_name in df.index:
            val = df.loc[row_name, col]
            if val is not None and str(val) != "nan":
                return float(val)
    except Exception:
        pass
    return None
