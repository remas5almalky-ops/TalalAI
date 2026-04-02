"""Tadawul Advisor - AI Saudi Stock Recommendation System (Streamlit)"""

import streamlit as st
import pandas as pd

from services import stock_screener, data_fetcher, swing_analyzer, opportunity_scanner, bottom_scanner
from services.cache import cache

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Tadawul Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# LANGUAGE STATE
# ──────────────────────────────────────────────

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def t(en: str, ar: str) -> str:
    """Return text based on current language."""
    return ar if st.session_state.lang == "ar" else en


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    # Language toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("English", use_container_width=True, type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with col2:
        if st.button("عربي", use_container_width=True, type="primary" if st.session_state.lang == "ar" else "secondary"):
            st.session_state.lang = "ar"
            st.rerun()

    st.markdown("---")
    st.markdown(f"### 📈 Tadawul Advisor")
    st.caption(t("Sharia-Compliant Stocks Only", "أسهم متوافقة مع الشريعة فقط"))
    st.markdown("---")

    page = st.radio(
        t("Navigate", "التنقل"),
        [
            t("1. 10%+ Opportunities", "1. فرص 10%+"),
            t("2. Bottom Fishing", "2. اسهم بالقاع"),
            t("3. Swing Trade", "3. تحليل الدخول"),
            t("4. Dashboard", "4. لوحة التحكم"),
            t("5. Search All Stocks", "5. البحث في كل الأسهم"),
        ],
        index=0,
    )

    st.markdown("---")
    if st.button(t("🔄 Refresh Data", "🔄 تحديث البيانات"), use_container_width=True):
        cache.clear()
        st.rerun()

    st.caption(t(
        "Data updates every 15 min. Recommendations are AI-driven and not financial advice.",
        "البيانات تتحدث كل 15 دقيقة. التوصيات مبنية على الذكاء الاصطناعي وليست نصيحة مالية."
    ))


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def color_action(action: str) -> str:
    """Return colored markdown for action."""
    colors = {"Strong Buy": "green", "Buy": "green", "Hold": "orange", "Avoid": "red",
              "BUY": "green", "Wait": "orange", "WAIT": "orange", "AVOID": "red"}
    c = colors.get(action, "gray")
    return f":{c}[**{action}**]"


def render_stock_card(rec):
    """Render a single stock recommendation card."""
    action_emoji = {"Strong Buy": "🟢", "Buy": "🟢", "Hold": "🟡", "Avoid": "🔴"}.get(rec.action, "⚪")
    trend_emoji = {"Uptrend": "📈", "Downtrend": "📉", "Sideways": "➡️"}.get(rec.signals.trend, "")

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**{rec.stock.name}** `{rec.stock.ticker}`")
            st.caption(f"{rec.stock.sector} • {rec.stock.name_ar}")
        with c2:
            st.metric(t("Price", "السعر"), f"{rec.stock.current_price:.2f}", f"{rec.stock.change_pct:+.2f}%")
        with c3:
            st.markdown(f"### {action_emoji} {rec.action}")
            st.caption(f"Score: {rec.opportunity_score:.0f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Entry", "دخول"), f"{rec.entry_price:.2f}")
        c2.metric(t("Target", "هدف"), f"{rec.target_price:.2f}", f"+{((rec.target_price - rec.entry_price) / rec.entry_price * 100):.1f}%")
        c3.metric(t("Stop Loss", "وقف خسارة"), f"{rec.stop_loss:.2f}")
        c4.metric(t("Trend", "الاتجاه"), f"{trend_emoji} {rec.signals.trend}")

        if rec.predicted_move > 0:
            st.success(f"{t('Expected Move', 'الحركة المتوقعة')}: **+{rec.predicted_move:.1f}%** | {t('Confidence', 'الثقة')}: {rec.confidence:.0f}%")

        with st.expander(t("Analysis", "التحليل")):
            st.write(rec.explanation)
            if rec.factors:
                for f in rec.factors[:4]:
                    st.markdown(f"- {f}")


# ──────────────────────────────────────────────
# PAGE 1: 10%+ OPPORTUNITIES
# ──────────────────────────────────────────────

def page_opportunities():
    st.markdown(f"# {t('10%+ Profit Opportunities', 'فرص ربح 10%+')}")
    st.caption(t(
        "AI-detected setups with 10%+ profit potential in Sharia-compliant Saudi stocks",
        "فرص مكتشفة بالذكاء الاصطناعي بإمكانية ربح 10%+ في أسهم متوافقة مع الشريعة"
    ))

    with st.spinner(t("Scanning all stocks...", "جاري فحص كل الأسهم...")):
        opps = opportunity_scanner.scan_opportunities()

    buy_opps = [o for o in opps if o.action == "Buy"]
    watch_opps = [o for o in opps if o.action == "Watch"]

    st.markdown(f"**{len(opps)}** {t('opportunities found', 'فرصة تم اكتشافها')}")

    # Buy Now
    if buy_opps:
        st.markdown(f"## 🟢 {t('BUY NOW', 'اشتري الآن')} ({len(buy_opps)})")

        for o in buy_opps:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"### {o.name} `{o.ticker}`")
                    st.caption(f"{o.sector} • {o.setup_type} • {o.probability} Probability")
                with c2:
                    st.metric(t("Price", "السعر"), f"{o.current_price:.2f} SAR")
                with c3:
                    st.markdown(f"### :green[+{o.profit_pct:.0f}%]")
                    st.caption(o.timeframe)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(t("Entry", "دخول"), f"{o.entry_price:.2f}")
                c2.metric(t("Target", "هدف"), f"{o.target_price:.2f}")
                c3.metric(t("Stop", "وقف"), f"{o.stop_loss:.2f}")
                c4.metric(t("Risk", "مخاطرة"), f"-{o.risk_pct:.1f}%")
                c5.metric("R:R", f"{o.risk_reward}:1")

                for r in o.reasons:
                    st.markdown(f"✅ {r}")
                for r in o.risks:
                    st.markdown(f"⚠️ {r}")

    # Watch
    if watch_opps:
        st.markdown(f"## 🟡 {t('WATCH - Wait for better entry', 'راقب - انتظر دخول أفضل')} ({len(watch_opps)})")
        for o in watch_opps:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{o.name}** `{o.ticker}` — {o.setup_type}")
                c1.caption(o.explanation)
                c2.markdown(f"### :orange[+{o.profit_pct:.0f}%]")
                c2.caption(f"Target: {o.target_price:.2f} • {o.timeframe}")

    if not opps:
        st.info(t("No 10%+ opportunities detected right now.", "لا توجد فرص 10%+ حالياً"))

    st.warning(t(
        "⚠️ High-profit targets carry higher risk. Always use stop losses. Not financial advice.",
        "⚠️ أهداف الربح العالي تحمل مخاطر أعلى. دائماً استخدم وقف الخسارة. ليست نصيحة مالية."
    ))


# ──────────────────────────────────────────────
# PAGE 2: BOTTOM FISHING
# ──────────────────────────────────────────────

def page_bottom_fishing():
    st.markdown(f"# {t('Bottom Fishing', 'اسهم بالقاع')}")
    st.caption(t(
        "Honest advice: not every cheap stock will recover. Stocks down 15%+ with candid analysis.",
        "نصيحة صادقة: مو كل سهم رخيص بيرتفع. هذي الاسهم النازلة 15%+ مع تحليل صريح"
    ))

    # Golden rules
    with st.expander(t("📐 Golden Rules - When to buy a bottom stock", "📐 القواعد الذهبية - متى تشتري سهم بالقاع"), expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.info(t("**1. MACD Bullish**\nMomentum must turn positive", "**1. MACD صاعد**\nالزخم لازم يتحول للشراء"))
        c2.info(t("**2. No Downtrend**\nUptrend or Sideways only", "**2. وقف النزول**\nUptrend أو Sideways فقط"))
        c3.info(t("**3. RSI 35-60**\nHealthy zone, not extreme", "**3. RSI بين 35-60**\nمنطقة صحية"))
        c4.info(t("**4. Stable 30 days**\nNot crashing anymore", "**4. مستقر 30 يوم**\nمو لسه ينهار"))

    with st.spinner(t("Scanning bottoms...", "جاري البحث عن القيعان...")):
        candidates = bottom_scanner.scan_bottoms()

    if not candidates:
        st.info(t("No bottom candidates found.", "لا توجد أسهم بالقاع حالياً"))
        return

    # Group by verdict
    verdict_groups = {}
    for b in candidates:
        v = b.honest_verdict_en
        if v not in verdict_groups:
            verdict_groups[v] = []
        verdict_groups[v].append(b)

    verdict_config = {
        "BUY": ("🟢", "green", t("Real opportunity - all signals positive", "فرصة حقيقية - كل الإشارات إيجابية")),
        "ACCUMULATE": ("🔵", "blue", t("Enter partial position, wait for confirmation", "ادخل بجزء وانتظر تأكيد")),
        "WAIT": ("🟡", "orange", t("Has potential but timing not ideal - monitor", "فيه إمكانية بس التوقيت مو مثالي - راقب")),
        "AVOID": ("🟠", "orange", t("More negative than positive - don't enter now", "إشارات سلبية أكثر - لا تدخل")),
        "STAY AWAY": ("🔴", "red", t("Falling knife! Cheap doesn't mean recovery", "سكين ساقط! الرخص مو ضمان للارتفاع")),
    }

    for verdict_en, items in verdict_groups.items():
        emoji, color, desc = verdict_config.get(verdict_en, ("⚪", "gray", ""))
        verdict_label = items[0].honest_verdict if st.session_state.lang == "ar" else verdict_en

        st.markdown(f"## {emoji} {verdict_label} ({len(items)})")
        st.caption(desc)

        for b in items:
            with st.container(border=True):
                # Header
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{b.name}** `{b.ticker}`")
                    st.caption(f"{b.sector} • {b.name_ar} • {b.status}")
                with c2:
                    st.metric(t("Price", "السعر"), f"{b.current_price:.2f}")
                with c3:
                    st.metric(t("Drop", "النزول"), f"-{b.drop_from_high_pct:.0f}%")

                # Stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(t("Upside to High", "ارتفاع متوقع"), f"+{b.upside_to_high_pct:.0f}%")
                c2.metric("RSI", f"{b.rsi:.0f}")
                c3.metric(t("30d Change", "تغير 30 يوم"), f"{b.change_30d:+.1f}%")
                c4.metric(t("vs SMA200", "مقابل SMA200"), f"{b.dist_from_sma200_pct:+.1f}%")

                # Progress bar (52w position)
                st.progress(min(100, max(0, int(b.position_52w))) / 100,
                            text=f"{t('52W Position', 'الموقع من 52 أسبوع')}: {b.position_52w:.0f}% ({b.low_52w:.2f} — {b.high_52w:.2f})")

                # Honest checklist
                with st.expander(t("✅ Honest Checklist", "✅ التحقق الصادق")):
                    for check in b.honest_checklist:
                        icon = "✅" if check["pass"] else "❌"
                        rule_text = check["rule"] if st.session_state.lang == "ar" else check["rule_en"]
                        st.markdown(f"{icon} {rule_text}")

                # Trade plan for Buy/Accumulate
                if b.honest_verdict_en in ("BUY", "ACCUMULATE"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric(t("Conservative Target", "هدف محافظ"), f"{b.target_price:.2f}", f"+{b.profit_pct:.0f}%")
                    c2.metric(t("Full Recovery", "تعافي كامل"), f"{b.target_price_full:.2f}", f"+{b.full_recovery_pct:.0f}%")
                    c3.metric(t("Stop Loss", "وقف خسارة"), f"{b.stop_loss:.2f}", f"-{b.risk_pct:.1f}%")

    st.warning(t(
        "⚠️ A stock down 50% can drop another 50%. Always use stop losses. Recovery takes 1-6 months.",
        "⚠️ السهم اللي ينزل 50% ممكن ينزل 50% ثاني. دائماً استخدم وقف الخسارة. التعافي يحتاج 1-6 شهور."
    ))


# ──────────────────────────────────────────────
# PAGE 3: SWING TRADE
# ──────────────────────────────────────────────

def page_swing():
    st.markdown(f"# {t('Swing Trade Analyzer', 'محلل التداول السريع')}")
    st.caption(t(
        "Enter a stock name or ticker to check if NOW is the right time to buy (3-7 days)",
        "ادخل اسم السهم وشوف هل الحين الوقت المناسب للشراء (3-7 أيام)"
    ))

    c1, c2, c3 = st.columns(3)
    c1.success(t("**Buy** = Enter now", "**اشتري** = ادخل الآن"))
    c2.warning(t("**Wait** = Not yet", "**انتظر** = مو الوقت"))
    c3.error(t("**Avoid** = Don't touch", "**تجنب** = لا تدخل"))

    # Stock selector
    stocks = data_fetcher.load_stock_list()
    stock_options = {f"{s['name']} ({s['ticker']})": s['ticker'] for s in stocks}

    selected = st.selectbox(
        t("Select a stock", "اختر سهم"),
        options=[""] + list(stock_options.keys()),
        index=0,
        placeholder=t("Type to search...", "اكتب للبحث...")
    )

    if selected and selected in stock_options:
        ticker = stock_options[selected]

        with st.spinner(t("Analyzing entry timing...", "جاري تحليل توقيت الدخول...")):
            s = swing_analyzer.analyze_swing(ticker)

        if s is None:
            st.error(t(f"No data for {ticker}", f"لا توجد بيانات لـ {ticker}"))
            return

        # Verdict banner
        if s.action == "Buy":
            st.success(f"### 🟢 {t('BUY NOW', 'اشتري الآن')} — {s.name} | {t('Timing Score', 'سكور التوقيت')}: {s.timing_score:.0f}/100")
        elif s.action == "Wait":
            st.warning(f"### 🟡 {t('WAIT', 'انتظر')} — {s.name} | {t('Timing Score', 'سكور التوقيت')}: {s.timing_score:.0f}/100")
        else:
            st.error(f"### 🔴 {t('AVOID', 'تجنب')} — {s.name} | {t('Timing Score', 'سكور التوقيت')}: {s.timing_score:.0f}/100")

        # Price + Entry Quality
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Price", "السعر"), f"{s.current_price:.2f} SAR", f"{s.change_pct:+.2f}%")
        c2.metric(t("Entry Quality", "جودة الدخول"), s.entry_quality)
        c3.metric(t("Confidence", "الثقة"), f"{s.confidence:.0f}%")
        c4.metric(t("Trend", "الاتجاه"), s.trend)

        # Recent price action
        st.markdown(f"### {t('Recent Price Action', 'حركة السعر الأخيرة')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("3-Day Move", "حركة 3 أيام"), f"{s.gain_last_3d:+.1f}%")
        c2.metric(t("5-Day Move", "حركة 5 أيام"), f"{s.gain_last_5d:+.1f}%")
        c3.metric(t("From Support", "من الدعم"), f"{s.dist_from_support_pct:.1f}%", help=f"Support: {s.support:.2f}")
        c4.metric(t("To Resistance", "للمقاومة"), f"{s.dist_from_resistance_pct:.1f}%", help=f"Resistance: {s.resistance:.2f}")

        # Trade plan
        if s.action in ("Buy", "Wait"):
            st.markdown(f"### {t('Trade Plan', 'خطة التداول')}")
            c1, c2, c3 = st.columns(3)
            c1.metric(t("Entry", "دخول"), f"{s.entry_price:.2f} SAR")
            c2.metric(t("Target", "هدف"), f"{s.target_price:.2f} SAR", f"+{s.expected_profit_pct:.1f}%")
            c3.metric(t("Stop Loss", "وقف"), f"{s.stop_loss:.2f} SAR", f"-{s.risk_pct:.1f}%")

            if s.risk_reward > 0:
                st.info(f"**R:R** {s.risk_reward}:1 | **{t('Profit', 'ربح')}**: +{s.expected_profit_pct:.1f}% | **{t('Risk', 'مخاطرة')}**: -{s.risk_pct:.1f}%")

        # Indicators
        st.markdown(f"### {t('Indicators', 'المؤشرات')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RSI", f"{s.rsi:.0f}")
        c2.metric("Stochastic", f"{s.stoch_k:.0f}")
        c3.metric("MACD", s.macd_crossover)
        c4.metric("Bollinger", s.bb_position)
        c5.metric(t("Volume", "الحجم"), f"{s.volume_ratio:.1f}x")

        # Signals
        if s.timing_reasons:
            st.markdown(f"### ✅ {t('Positive Signals', 'إشارات إيجابية')}")
            for r in s.timing_reasons:
                st.markdown(f"- ✅ {r}")

        if s.warning_flags:
            st.markdown(f"### ⚠️ {t('Warning Flags', 'تحذيرات')}")
            for w in s.warning_flags:
                st.markdown(f"- ⚠️ {w}")

        st.write(s.explanation)


# ──────────────────────────────────────────────
# PAGE 4: DASHBOARD
# ──────────────────────────────────────────────

def page_dashboard():
    st.markdown(f"# {t('Dashboard', 'لوحة التحكم')}")

    with st.spinner(t("Loading market data...", "جاري تحميل بيانات السوق...")):
        summary = stock_screener.get_market_summary()
        top_picks = stock_screener.get_top_picks()

    if summary and summary.get("total"):
        # Market overview
        c1, c2, c3, c4, c5 = st.columns(5)
        mood = summary["mood"]
        mood_emoji = "📈" if "Bullish" in mood else ("📉" if "Bearish" in mood else "➡️")
        c1.metric(t("Market Mood", "حالة السوق"), f"{mood_emoji} {mood}")
        c2.metric(t("Buy Signals", "إشارات شراء"), summary["buy"], help=f"of {summary['total']}")
        c3.metric(t("Hold", "انتظار"), summary["hold"])
        c4.metric(t("Avoid", "تجنب"), summary["avoid"])
        c5.metric(t("Avg Score", "متوسط السكور"), summary["avg_score"])

        # Sector breakdown
        if summary.get("sectors"):
            with st.expander(t("Sector Overview", "نظرة على القطاعات"), expanded=False):
                sector_data = []
                for name, data in summary["sectors"].items():
                    sector_data.append({"Sector": name, "Score": data["avg_score"],
                                       "Buy": data["buy"], "Hold": data["hold"], "Avoid": data["avoid"]})
                st.dataframe(pd.DataFrame(sector_data).sort_values("Score", ascending=False),
                             use_container_width=True, hide_index=True)

    # Top picks
    st.markdown(f"## {t('Top Picks Today', 'أفضل الأسهم اليوم')}")
    st.caption(t("AI-ranked recommendations • Sharia-Compliant", "توصيات مرتبة بالذكاء الاصطناعي • متوافقة مع الشريعة"))

    if top_picks:
        for rec in top_picks:
            render_stock_card(rec)
    else:
        st.info(t("Loading recommendations...", "جاري تحميل التوصيات..."))


# ──────────────────────────────────────────────
# PAGE 5: SEARCH
# ──────────────────────────────────────────────

def page_search():
    st.markdown(f"# {t('Search & Filter All Stocks', 'البحث في كل الأسهم')}")

    # Filters
    sectors = data_fetcher.get_sectors()
    sector_names = ["All"] + [s["id"] for s in sectors]

    c1, c2, c3, c4 = st.columns(4)
    sector = c1.selectbox(t("Sector", "القطاع"), sector_names)
    action_filter = c2.selectbox(t("Recommendation", "التوصية"), ["All", "Strong Buy", "Buy", "Hold", "Avoid"])
    min_price = c3.number_input(t("Min Price", "أقل سعر"), 0.0, 1000.0, 0.0, step=5.0)
    max_price = c4.number_input(t("Max Price", "أعلى سعر"), 0.0, 1000.0, 0.0, step=5.0)

    with st.spinner(t("Loading...", "جاري التحميل...")):
        results = stock_screener.filter_stocks(
            sector=sector if sector != "All" else None,
            action=action_filter if action_filter != "All" else None,
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price > 0 else None,
        )

    st.caption(f"{len(results)} {t('stocks found', 'سهم')}")

    if results:
        table_data = []
        for r in results:
            table_data.append({
                t("Stock", "السهم"): f"{r.stock.name} ({r.stock.ticker})",
                t("Sector", "القطاع"): r.stock.sector,
                t("Price", "السعر"): f"{r.stock.current_price:.2f}",
                t("Change", "التغير"): f"{r.stock.change_pct:+.2f}%",
                "Score": f"{r.opportunity_score:.0f}",
                t("Action", "التوصية"): r.action,
                t("Trend", "الاتجاه"): r.signals.trend,
                t("Entry", "دخول"): f"{r.entry_price:.2f}",
                t("Target", "هدف"): f"{r.target_price:.2f}",
                t("Stop", "وقف"): f"{r.stop_loss:.2f}",
                t("Expected", "متوقع"): f"+{r.predicted_move:.1f}%",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Expandable cards
        with st.expander(t("Show detailed cards", "عرض البطاقات المفصلة")):
            for rec in results[:10]:
                render_stock_card(rec)


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────

page_key = page.split(".")[0].strip() if page else "1"

if "1" in page_key:
    page_opportunities()
elif "2" in page_key:
    page_bottom_fishing()
elif "3" in page_key:
    page_swing()
elif "4" in page_key:
    page_dashboard()
elif "5" in page_key:
    page_search()
