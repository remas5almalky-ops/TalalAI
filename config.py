"""Application configuration."""

# Cache settings
CACHE_TTL_SECONDS = 900  # 15 minutes
HISTORICAL_PERIOD = "6mo"
HISTORICAL_INTERVAL = "1d"

# Scoring weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "momentum": 0.25,       # Price momentum + trend strength (primary driver)
    "macd": 0.15,           # MACD crossover and histogram
    "rsi": 0.10,            # RSI - momentum zone, not just oversold
    "trend": 0.15,          # MA alignment and direction
    "volume": 0.10,         # Volume confirmation
    "breakout": 0.10,       # Bollinger breakout + 52w range position
    "volatility": 0.05,     # Moderate volatility preferred
    "support_resistance": 0.10,  # Risk/reward from S/R levels
}

# Recommendation thresholds
STRONG_BUY_THRESHOLD = 75
BUY_THRESHOLD = 58
HOLD_THRESHOLD = 40

# Sharia compliance filter
SHARIA_ONLY = True  # Only analyze Sharia-compliant stocks

# Top picks
TOP_PICKS_COUNT = 10
MAX_PER_SECTOR = 3

# ATR multipliers for targets
TARGET_ATR_MULTIPLIER = 2.5
STOP_LOSS_ATR_MULTIPLIER = 1.5
