"""
CARE — Calmar-Adaptive Regime Engine
=====================================
builderr Trading Challenge submission.

Optimization target: Calmar ratio (annualized_return / max_drawdown) over 30 days.

Design philosophy
-----------------
The Calmar denominator (max drawdown) is permanently and irreversibly damaged by a
single bad day.  Protecting it beats chasing the numerator.  Every design decision
here flows from that one fact:

  1. Multi-signal composite regime score — continuous 0-1 score across trend,
     momentum, and volatility prevents the abrupt binary ON/OFF flip-flopping that
     churns trades and misses recovery entries.

  2. Four exposure levels (full / moderate / defensive / cash) with hysteresis —
     "full" stays full through small wobbles; it takes a genuine breakdown to de-risk.
     De-risking happens immediately; re-risking waits for the cadence.

  3. Fast crash brake — a hard override that fires on a 3- or 5-day QQQ price plunge
     or an extreme 10-day vol spike.  Moves us to cash within one bar of the event.

  4. Inverse-volatility position sizing — names with lower realized vol receive
     higher allocation.  This directly minimizes each name's expected contribution
     to portfolio drawdown.

  5. Portfolio-level drawdown stop — if OUR equity curve falls 4/8/14% from its
     peak, we automatically cap the regime at moderate/defensive/cash regardless of
     market signals.  The last-resort line that saves us when signals are slow.

  6. No leveraged ETFs — over a 30-day Calmar window, 2x/3x products add more to
     the drawdown denominator (gap risk, vol decay, leverage reset) than to the
     return numerator.  1x long-only, capped at ~0.95x beta-adjusted gross.

All computation uses only the Python standard library — no network calls, no LLM,
no API keys.  All signals are constructed from the historical bars provided in
market_state, so there is zero lookahead bias.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

# Risk-on candidates: broad indices, sectors, mega-cap tech
_RISK_ON = (
    "SPY", "QQQ", "SMH",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE",
    "NVDA", "MSFT", "AAPL", "META", "GOOGL", "AMZN", "AVGO",
    "IWM",
)

# Defensive candidates: low-beta, low-correlation assets
_DEFENSIVE = ("XLP", "XLU", "XLV", "GLD")

# Combined pool used for "moderate" regime ranking
_ALL = _RISK_ON + ("XLP", "XLU", "GLD")

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

_SMA_SHORT = 50          # short trend filter (days)
_SMA_LONG  = 100         # long trend filter (days)
_TREND_BAND = 0.01       # hysteresis: need +1% above SMA to count as "above"

_MOM_FAST = 20           # fast momentum window (days)
_MOM_MED  = 60           # medium momentum window — primary cross-sectional signal
_MOM_SKIP = 5            # skip last N days to avoid short-term reversal effect

_VOL_WIN = 20            # realized-vol lookback (days)

# Crash brake — hard override, fires on extreme, fast moves only
_BRAKE_3D     = -0.050   # 3-day QQQ return threshold
_BRAKE_5D     = -0.070   # 5-day QQQ return threshold
_BRAKE_VOL10  = 0.65     # 10-day annualized QQQ vol (extreme meltdown only)

# Composite score → regime mapping thresholds
# Higher score = more risk-on conditions
_THRESH_FULL_UP   = 0.65  # score above this → full  (from lower regime)
_THRESH_FULL_STAY = 0.38  # score above this → stay full (from full)
_THRESH_MOD_UP    = 0.48  # score above this → moderate (from defensive)
_THRESH_MOD_STAY  = 0.28  # score above this → stay moderate (from moderate)
_THRESH_DEF       = 0.22  # score above this → defensive (otherwise cash)

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

_NAME_CAP   = 0.20       # max weight per name (well under the 30% rule)
_GROSS_MAX  = 0.97       # hard ceiling on total exposure
_TARGET_VOL = 0.13       # annualized portfolio vol target for gross scaling
_VOL_SCALE_FLOOR = 0.50  # vol-scaling never cuts below this fraction of regime gross

_TOP_N_FULL = 6          # names held in full regime
_TOP_N_MOD  = 5          # names held in moderate regime (mix of risk + defensive)
_TOP_N_DEF  = 3          # names held in defensive regime

_DEAD_BAND = 0.025       # ignore rebalance orders smaller than 2.5% of equity

# Target gross by regime (before vol scaling)
_REGIME_GROSS = {
    "full":      0.95,
    "moderate":  0.65,
    "defensive": 0.25,
    "cash":      0.05,
}

_REGIME_ORDER = ["full", "moderate", "defensive", "cash"]  # most→least risk

# ═══════════════════════════════════════════════════════════════════════════════
# REBALANCING / CADENCE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

_REBALANCE_DAYS = 5      # routine rebalance cadence (trading days)
_DRIFT_LIMIT    = 0.25   # trigger early rebalance if any position drifts past this
_COOLDOWN_DAYS  = 1      # days to stay cautious after crash brake fires

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO-LEVEL DRAWDOWN STOPS
# Maximum drawdown from rolling peak equity that forces a regime downgrade
# ═══════════════════════════════════════════════════════════════════════════════

_DD_TO_MODERATE  = 0.04  # 4%  drawdown → cap regime at "moderate"
_DD_TO_DEFENSIVE = 0.08  # 8%  drawdown → cap regime at "defensive"
_DD_TO_CASH      = 0.14  # 14% drawdown → cap regime at "cash"

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE STATE  (resets per subprocess — mirrors engine behavior)
# ═══════════════════════════════════════════════════════════════════════════════

_tick           = 0
_last_rebalance = -(10 ** 9)
_last_regime    = None     # type: str | None
_cooldown_left  = 0
_peak_equity    = None     # type: float | None

_ANN = 252 ** 0.5          # annualization constant for daily returns

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def _closes(bars: list) -> list:
    """Extract close prices from bar list, oldest first."""
    return [float(b["close"]) for b in bars] if bars else []


def _sma(c: list, n: int):
    """Simple moving average of last n values. Returns None if insufficient data."""
    return sum(c[-n:]) / n if len(c) >= n else None


def _ann_vol(c: list, n: int):
    """
    Annualized realized volatility from the last n daily returns.
    Returns None if insufficient data; uses population std with small-sample guard.
    """
    if len(c) < n + 1:
        return None
    rets = [c[i] / c[i - 1] - 1.0
            for i in range(len(c) - n, len(c))
            if c[i - 1] > 0]
    if len(rets) < 5:
        return None
    mean = sum(rets) / len(rets)
    # Use sample variance (n-1) for unbiased estimate
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return var ** 0.5 * _ANN


def _ret(c: list, days: int, skip: int = 0):
    """
    Return over a window of `days`, optionally skipping the last `skip` bars.
    The skip handles the short-term reversal effect in momentum literature.
    """
    need = days + skip + 1
    if len(c) < need:
        return None
    end   = c[-(skip + 1)] if skip > 0 else c[-1]
    start = c[-(days + skip + 1)]
    return end / start - 1.0 if start > 0 else None


def _momentum_score(c: list, require_uptrend: bool = True):
    """
    Multi-timeframe momentum composite for cross-sectional ranking.

    Formula: (0.40 * r_med + 0.30 * r_fast + 0.15 * trend_gap) * vol_adj

    - r_med: 60-day return skipping last 5 days (removes reversal drag)
    - r_fast: 20-day return (recency weight)
    - trend_gap: distance above 50-day SMA (trend confirmation)
    - vol_adj: penalizes high-vol names since they are bigger DD contributors

    Returns None if data is insufficient or (when require_uptrend=True) if
    the name is below its 50-day SMA — we never buy downtrending assets.
    """
    sma   = _sma(c, _SMA_SHORT)
    r_f   = _ret(c, _MOM_FAST)
    r_m   = _ret(c, _MOM_MED, _MOM_SKIP)
    v     = _ann_vol(c, _VOL_WIN)
    if sma is None or r_f is None or r_m is None or not c:
        return None
    if require_uptrend and c[-1] < sma:
        return None  # below own 50-day SMA → not in uptrend, skip
    trend_gap = c[-1] / sma - 1.0
    v_safe    = max(v, 0.05) if v is not None else 0.20
    vol_adj   = max(0.10, 1.0 - v_safe / 0.50)  # 0.10–1.0; high vol → lower score
    return (0.40 * r_m + 0.30 * r_f + 0.15 * trend_gap) * vol_adj


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _crash_brake(market_state: dict) -> bool:
    """
    Hard override: returns True on a fast crash or extreme vol spike.
    Deliberately keeps the vol trigger very high — moderate elevated vol is
    handled by vol-targeting the gross, not by going to cash and missing the
    snapback whose realized vol stays high while prices recover.
    """
    qqq = _closes(market_state.get("QQQ") or [])
    r3  = _ret(qqq, 3)
    r5  = _ret(qqq, 5)
    v10 = _ann_vol(qqq, 10)
    if r3 is not None and r3 < _BRAKE_3D:
        return True
    if r5 is not None and r5 < _BRAKE_5D:
        return True
    if v10 is not None and v10 > _BRAKE_VOL10:
        return True
    return False


def _composite_score(market_state: dict) -> float:
    """
    Composite regime score in [0, 1].  Higher = more risk-on conditions.
    Three components, each normalized to [0, 1]:

      Trend component    (35% weight)
        SPY & QQQ position vs. their 50-day and 100-day SMAs.

      Momentum component (30% weight)
        SPY & QQQ 20-day and 60-day returns (signed: positive = +1, negative = 0).

      Volatility component (35% weight)
        QQQ 20-day annualized vol mapped to a 0-1 scale (lower vol → higher score).
    """
    qqq = _closes(market_state.get("QQQ") or [])
    spy = _closes(market_state.get("SPY") or [])
    if len(qqq) < 10 or len(spy) < 10:
        return 0.0

    score  = 0.0
    weight = 0.0

    # ── Trend (35%) ───────────────────────────────────────────────────────
    trend_pts = 0.0
    for closes, label in ((spy, "SPY"), (qqq, "QQQ")):
        sma50  = _sma(closes, _SMA_SHORT)
        sma100 = _sma(closes, _SMA_LONG)
        last   = closes[-1]
        if sma50 is not None:
            if last > sma50 * (1 + _TREND_BAND):
                trend_pts += 0.25          # clearly above 50d SMA
            elif last > sma50:
                trend_pts += 0.15          # just above (partial credit)
        if sma100 is not None and last > sma100:
            trend_pts += 0.25              # above 100d SMA (secondary confirmation)
    score  += 0.35 * trend_pts
    weight += 0.35

    # ── Momentum (30%) ────────────────────────────────────────────────────
    # Note: no skip here — we want the most current reading for regime
    # detection.  The skip is only applied in cross-sectional name ranking
    # where the short-term reversal effect is relevant.
    mom_pts = 0.0
    for closes in (spy, qqq):
        r20 = _ret(closes, _MOM_FAST)
        r60 = _ret(closes, _MOM_MED)   # no skip — use current 60d window
        if r20 is not None:
            mom_pts += 0.25 if r20 > 0 else 0.0
        if r60 is not None:
            mom_pts += 0.25 if r60 > 0 else 0.0
    score  += 0.30 * mom_pts
    weight += 0.30

    # ── Volatility (35%) ──────────────────────────────────────────────────
    # Linear: 1.0 at 10% vol, 0.0 at 42% vol — no step-boundary oscillation.
    v20 = _ann_vol(qqq, _VOL_WIN)
    if v20 is not None:
        vol_pts = max(0.0, min(1.0, 1.0 - (v20 - 0.10) / 0.32))
        score  += 0.35 * vol_pts
        weight += 0.35

    return score / weight if weight > 0 else 0.0


def _detect_regime(market_state: dict) -> str:
    """
    Returns one of: 'full', 'moderate', 'defensive', 'cash'.

    Hysteresis logic: once in a higher-risk regime, we require a clear
    breakdown to drop — avoiding the daily SMA-cross flip-flop in choppy
    markets.  The crash brake always overrides this and goes directly to cash.

    Cooldown: after crash brake fires, we stay cautious for _COOLDOWN_DAYS
    even if signals immediately recover — prevents jumping into a dead-cat-
    bounce snapback.
    """
    global _cooldown_left

    if _crash_brake(market_state):
        _cooldown_left = _COOLDOWN_DAYS
        return "cash"

    score = _composite_score(market_state)

    if _cooldown_left > 0:
        _cooldown_left -= 1
        return "defensive" if score > 0.45 else "cash"

    prev = _last_regime

    if prev == "full":
        if score >= _THRESH_FULL_STAY:
            return "full"
        elif score >= _THRESH_MOD_STAY:
            return "moderate"
        else:
            return "defensive"

    elif prev == "moderate":
        if score >= _THRESH_FULL_UP:
            return "full"
        elif score >= _THRESH_MOD_STAY:
            return "moderate"
        else:
            return "defensive"

    elif prev == "defensive":
        if score >= _THRESH_FULL_UP:
            return "full"
        elif score >= _THRESH_MOD_UP:
            return "moderate"
        elif score >= _THRESH_DEF:
            return "defensive"
        else:
            return "cash"

    else:
        # Cold start: be conservative — require a clear bull signal to go full
        if score >= _THRESH_FULL_UP:
            return "full"
        elif score >= _THRESH_MOD_UP:
            return "moderate"
        elif score >= _THRESH_DEF:
            return "defensive"
        else:
            return "cash"


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO-LEVEL DRAWDOWN STOP
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_dd_override(regime: str, equity: float, peak: float) -> str:
    """
    Cap the regime based on how far our own portfolio has drawn down from peak.
    This is the last-resort safety net: if our signals are wrong and we ARE
    losing money, we automatically reduce exposure to protect the Calmar
    denominator regardless of what market indicators say.
    """
    if peak <= 0:
        return regime
    dd = (peak - equity) / peak
    if dd >= _DD_TO_CASH:
        return "cash"
    if dd >= _DD_TO_DEFENSIVE:
        # Clamp: regime cannot be better than "defensive"
        idx = max(_REGIME_ORDER.index(regime), _REGIME_ORDER.index("defensive"))
        return _REGIME_ORDER[idx]
    if dd >= _DD_TO_MODERATE:
        idx = max(_REGIME_ORDER.index(regime), _REGIME_ORDER.index("moderate"))
        return _REGIME_ORDER[idx]
    return regime


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_by_momentum(
    market_state: dict,
    universe: tuple,
    require_uptrend: bool = True,
) -> list:
    """Rank names in universe by momentum score, best first. Filters out None scores."""
    scored = []
    for t in universe:
        bars = market_state.get(t)
        if not bars:
            continue
        s = _momentum_score(_closes(bars), require_uptrend=require_uptrend)
        if s is not None:
            scored.append((s, t))
    scored.sort(reverse=True)
    return [t for _, t in scored]


def _inv_vol_weights(
    names: list,
    market_state: dict,
    gross: float,
) -> dict:
    """
    Inverse-volatility weighting: names with lower realized vol receive
    proportionally higher allocation, capped at _NAME_CAP per name.

    The gross target controls total exposure.  If some names hit the cap,
    the remaining gross is lost to cash — that is intentional and conservative.
    """
    if not names:
        return {}
    inv = {}
    for t in names:
        c = _closes(market_state.get(t) or [])
        v = _ann_vol(c, _VOL_WIN)
        if v and v > 1e-6:
            inv[t] = 1.0 / v
        else:
            inv[t] = 1.0 / 0.20  # fallback: assume 20% annual vol
    total = sum(inv.values())
    return {t: min(_NAME_CAP, gross * x / total) for t, x in inv.items()}


def _vol_scaled_gross(regime: str, market_state: dict) -> float:
    """
    Scale the regime's base gross down when market vol is elevated, targeting
    _TARGET_VOL portfolio annualized volatility.  Never goes below 50% of the
    base gross — vol-targeting shouldn't lock us out of a recovering market.
    """
    base = _REGIME_GROSS[regime]
    if regime in ("cash", "defensive", "moderate"):
        return base  # already de-risked; don't compound with vol scaling
    qqq = _closes(market_state.get("QQQ") or [])
    mkt_vol = _ann_vol(qqq, _VOL_WIN) or 0.20
    scaled = min(base, _TARGET_VOL / mkt_vol)
    floor  = base * _VOL_SCALE_FLOOR
    return max(floor, min(_GROSS_MAX, scaled))


def _target_weights(market_state: dict, regime: str) -> dict:
    """
    Compute target allocation weights for each position given the current regime.

      full      → top _TOP_N_FULL risk-on names, inv-vol weighted
      moderate  → top risk names + defensive ballast, mixed inv-vol weighted
      defensive → top _TOP_N_DEF defensive names, inv-vol weighted
      cash      → tiny XLP/XLU sleeve to avoid sitting completely flat
                  (helps if the market rips back — even 5% participation helps Calmar)
    """
    gross = _vol_scaled_gross(regime, market_state)

    if regime == "full":
        ranked = _rank_by_momentum(market_state, _RISK_ON)
        selected = ranked[:_TOP_N_FULL]
        if not selected:
            # Fallback: if nothing passes the SMA filter, go defensive
            ranked_def = _rank_by_momentum(market_state, _DEFENSIVE, require_uptrend=False)
            return _inv_vol_weights(ranked_def[:_TOP_N_DEF], market_state,
                                    _REGIME_GROSS["defensive"])
        return _inv_vol_weights(selected, market_state, gross)

    if regime == "moderate":
        # ~70% risk, ~30% defensive — blended by inverse vol
        ranked_risk = _rank_by_momentum(market_state, _RISK_ON)
        ranked_def  = _rank_by_momentum(market_state, ("XLP", "XLU", "GLD"),
                                        require_uptrend=False)
        risk_sel = ranked_risk[:3]
        def_sel  = [t for t in ranked_def[:2] if t not in risk_sel]
        combined = risk_sel + def_sel
        if not combined:
            return _inv_vol_weights(
                _rank_by_momentum(market_state, _DEFENSIVE, require_uptrend=False)[:_TOP_N_DEF],
                market_state, _REGIME_GROSS["defensive"],
            )
        return _inv_vol_weights(combined, market_state, gross)

    if regime == "defensive":
        ranked = _rank_by_momentum(market_state, _DEFENSIVE, require_uptrend=False)
        selected = ranked[:_TOP_N_DEF]
        if not selected:
            return {}
        return _inv_vol_weights(selected, market_state, gross)

    # regime == "cash"
    fallback = [t for t in ("XLP", "XLU") if market_state.get(t)]
    return _inv_vol_weights(fallback, market_state, gross) if fallback else {}


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_orders(
    targets: dict,
    positions: dict,
    last_prices: dict,
    equity: float,
) -> list:
    """
    Produce buy/sell orders to move from current holdings toward target weights.
    Skips trivial trades smaller than _DEAD_BAND * equity to control turnover.
    Sells come before buys to free cash first.
    """
    orders = []

    # ── Sells first ───────────────────────────────────────────────────────
    for ticker, pos in positions.items():
        qty = pos.get("quantity", 0)
        if qty <= 0:
            continue
        px = last_prices.get(ticker, pos.get("avg_cost", 0))
        if px <= 0:
            continue

        target_wt = targets.get(ticker, 0.0)
        target_val = equity * target_wt
        current_val = qty * px

        if ticker not in targets:
            # Full exit
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
            continue

        delta = target_val - current_val
        if abs(delta) < _DEAD_BAND * equity:
            continue
        if delta < 0:
            sell_qty = min(qty, int(abs(delta) / px))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})

    # ── Buys ──────────────────────────────────────────────────────────────
    for ticker, weight in targets.items():
        if weight <= 0:
            continue
        px = last_prices.get(ticker, 0)
        if px <= 0:
            continue
        cur_qty = positions.get(ticker, {}).get("quantity", 0)
        target_val  = equity * weight
        current_val = cur_qty * px
        delta       = target_val - current_val
        if abs(delta) < _DEAD_BAND * equity:
            continue
        if delta > 0:
            buy_qty = int(delta / px)
            if buy_qty > 0:
                orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})

    return orders


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def decide(market_state, portfolio_state, cash):
    """
    Called once per trading day by the builderr engine.

    Returns a list of orders: [{"ticker": str, "side": "buy"|"sell", "quantity": float}]
    Orders fill at the NEXT day's open + slippage (5 bps regular, 10 bps leveraged).
    An empty list means no action this tick.
    """
    global _tick, _last_rebalance, _last_regime, _peak_equity

    _tick += 1

    # ── 1. Compute equity, update rolling peak ────────────────────────────
    positions    = {p["ticker"]: p for p in portfolio_state.get("positions", [])}
    last_prices  = portfolio_state.get("last_prices", {})
    equity       = portfolio_state.get("cash", cash)
    for tk, pos in positions.items():
        equity += pos["quantity"] * last_prices.get(tk, pos.get("avg_cost", 0))
    if equity <= 0:
        return []

    if _peak_equity is None or equity > _peak_equity:
        _peak_equity = equity

    # ── 2. Detect market regime ───────────────────────────────────────────
    regime = _detect_regime(market_state)

    # ── 3. Apply portfolio-level drawdown stop (last-resort override) ─────
    regime = _apply_dd_override(regime, equity, _peak_equity)

    # ── 4. Decide whether to rebalance this tick ──────────────────────────
    # De-risking: always execute immediately (don't wait for cadence).
    # Re-risking: only on the routine cadence or a drift breach.
    regime_degraded = (
        _last_regime is not None
        and _REGIME_ORDER.index(regime) > _REGIME_ORDER.index(_last_regime)
    )
    drifted = any(
        pos["quantity"] * last_prices.get(t, pos.get("avg_cost", 0)) / equity > _DRIFT_LIMIT
        for t, pos in positions.items()
    ) if positions else False
    on_cadence = (_tick - _last_rebalance) >= _REBALANCE_DAYS

    _last_regime = regime  # update AFTER reading for hysteresis comparison

    if not (regime_degraded or drifted or on_cadence):
        return []

    # ── 5. Compute target allocation ──────────────────────────────────────
    targets = _target_weights(market_state, regime)

    # ── 6. Generate and return orders ─────────────────────────────────────
    orders = _generate_orders(targets, positions, last_prices, equity)

    if orders:
        _last_rebalance = _tick

    return orders
