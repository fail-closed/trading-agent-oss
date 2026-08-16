"""
ml_trainer.py — build the dip confidence ML model.

Collects historical outcomes from yfinance (max history) across a large
universe of stocks, extracts signal features, trains an XGBoost classifier
to predict next-day bounce probability, and saves the model.

Usage:
    python3 ml_trainer.py                      # default: S&P 500 + Russell 1000
    python3 ml_trainer.py --years 10           # last 10 years only
    python3 ml_trainer.py --quick              # quick test on 50 stocks

Output:
    models/dip_confidence.pkl   — trained model
    models/feature_importance.json
    models/training_report.json
"""
import argparse
import json
import os
import pickle
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Fetch stock universe ──────────────────────────────────────────────────────

def fetch_sp500_symbols() -> list:
    """Get S&P 500 symbols — tries Wikipedia first, then falls back to Alpaca universe."""
    # Try Wikipedia
    try:
        import requests
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        ).text
        tables = pd.read_html(html, header=0)
        df = tables[0]
        syms = df.iloc[:, 0].str.replace(".", "-", regex=False).str.strip().tolist()
        result = [s for s in syms if s.isalpha() and len(s) <= 5]
        if result:
            print(f"  S&P 500: {len(result)} symbols from Wikipedia")
            return result
    except Exception as e:
        print(f"  Wikipedia unavailable ({e}) — using universe.json")

    # Fall back to our existing universe file
    import json
    try:
        with open("universe.json") as f:
            u = json.load(f)
        syms = [s["symbol"] for s in u["stocks"] if s["symbol"].isalpha()]
        print(f"  Using universe.json: {len(syms)} symbols")
        return syms
    except Exception:
        pass

    # Last resort: curated large-cap list
    print("  Using fallback curated list")
    return [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","V",
        "MA","UNH","XOM","LLY","JNJ","PG","HD","MRK","ABBV","CVX","KO","PEP",
        "COST","AVGO","WMT","BAC","CRM","MCD","CSCO","ACN","PFE","TMO","INTC",
        "AMD","TXN","PM","QCOM","NKE","ORCL","IBM","GE","CAT","BA","RTX","HON",
        "LMT","GS","MS","C","WFC","AXP","BLK","SCHW","SPY","QQQ","IWM","DIA",
    ]


def fetch_russell1000_symbols() -> list:
    """Get additional Russell 1000 symbols (mid-caps not in S&P 500)."""
    # Common ETF holdings proxy — top mid-caps frequently in Russell 1000
    # These supplement the S&P 500 for better coverage of the full large/mid cap universe
    extra = [
        "DECK","PODD","CSGP","ELS","NVR","MOH","IDXX","WST","RMD","CPRT",
        "ODFL","DXCM","ANSS","MPWR","PAYC","SNPS","CDNS","ZBRA","BR","VRSN",
        "INTU","MKTX","MANH","FICO","TRMB","EXR","MAA","CPT","HST","INVH",
        "AME","IEX","ROP","TYL","GNRC","POOL","TECH","FMC","CE","PKI",
        "HII","DPZ","MTCH","NDSN","JBHT","CHRW","EXPD","LSTR","SAIA","WERN",
        "ALSN","NXST","SBAC","AMG","EV","BEN","TROW","IVZ","PFG","VOYA",
        "RF","FITB","CFG","HBAN","KEY","CMA","ZION","MTB","NTRS","STT",
    ]
    return [s for s in extra if s.isalpha()]


# ── Feature computation ────────────────────────────────────────────────────────

def compute_features_series(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Vectorised feature computation for all dates in a price series.
    Much faster than per-date loops.
    """
    df = pd.DataFrame(index=close.index)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    # MA crossover
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["sma50_vs_200"] = (sma50 - sma200) / sma200 * 100  # % difference

    # MACD(12,26,9) gap
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_gap_pct"] = (macd - signal) / close * 100  # normalised by price

    # Bollinger pct_b
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    df["pct_b"] = (close - (sma20 - 2*std20)) / (4 * std20.where(std20 > 0, np.nan))

    # Dip depth vs SMA20
    df["dip_depth_pct"] = (sma20 - close) / sma20 * 100  # positive = below SMA20

    # Volume ratio
    avg_vol = volume.rolling(20).mean().shift(1)           # exclude today
    df["volume_ratio"] = volume / avg_vol.where(avg_vol > 0, np.nan)

    # Adaptive weighted score (simplified from research.py)
    s_rsi  = np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))
    s_ma   = np.where(sma50 > sma200, 1, -1)
    s_macd = np.where(macd > signal, 1, -1)
    s_bb   = np.where(df["pct_b"] < 0.10, 1, np.where(df["pct_b"] > 0.90, -1, 0))
    s_vol  = np.where(df["volume_ratio"] > 1.5, 1, np.where(df["volume_ratio"] < 0.8, -1, 0))

    df["raw_score"]      = s_rsi + s_ma + s_macd + s_bb + s_vol
    df["weighted_score"] = (s_rsi  * np.where(s_rsi  > 0, 1.5, 0.3) +
                            s_ma   * np.where(s_ma   > 0, 1.0, 0.4) +
                            s_macd * np.where(s_macd > 0, 1.0, 0.3) +
                            s_bb   * np.where(s_bb   > 0, 1.5, 0.3) +
                            s_vol  * np.where(s_vol  > 0, 1.0, 0.5))

    # Price momentum
    df["mom_5d"]  = close.pct_change(5)  * 100
    df["mom_20d"] = close.pct_change(20) * 100

    # Volatility (20-day realised vol)
    df["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Support score — 52-week range position (vectorised approximation for training)
    # Full support detection is too slow per-row; use 52w range as proxy
    rolling_high = close.rolling(252, min_periods=50).max()
    rolling_low  = close.rolling(252, min_periods=50).min()
    rng          = rolling_high - rolling_low
    pct_from_low = (close - rolling_low) / rng.where(rng > 0, np.nan)
    # Simple support score: lower in 52w range = higher score
    df["support_score"] = (1 - pct_from_low.clip(0, 1)) * 60  # 0-60 from 52w position

    # ── Expanded (challenger) features — close/volume derived, no extra data ──────
    # Multi-timeframe trend alignment: in an uptrend stack (price>SMA50>SMA200)?
    df["mtf_aligned"] = ((close > sma50) & (sma50 > sma200)).astype(float)
    # Bollinger squeeze: BB width in the bottom 20% of its trailing year (vol compression)
    bb_width = (4 * std20) / sma20.where(sma20 > 0, np.nan)
    df["bb_squeeze"] = (bb_width < bb_width.rolling(252, min_periods=60).quantile(0.20)).astype(float)
    # Donchian position: where in the 20-day range (0=at low, 1=at high)
    dch_hi = close.rolling(20).max(); dch_lo = close.rolling(20).min()
    df["donchian_pos"] = (close - dch_lo) / (dch_hi - dch_lo).where(dch_hi > dch_lo, np.nan)
    # Distance below the 52-week high (extension/room)
    df["dist_52w_high"] = (rolling_high - close) / rolling_high.where(rolling_high > 0, np.nan) * 100
    # Longer momentum horizons
    df["mom_60d"]  = close.pct_change(60)  * 100
    df["mom_120d"] = close.pct_change(120) * 100
    # Volume trend (recent vs base) and volatility regime (vol percentile in trailing year)
    df["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean().where(volume.rolling(20).mean() > 0, np.nan)
    df["vol_pct_252"] = df["vol_20d"].rolling(252, min_periods=60).rank(pct=True)

    return df


def triple_barrier_labels(close: pd.Series, up: float = 0.05, dn: float = 0.03,
                          horizon: int = 10) -> pd.Series:
    """
    Triple-barrier label (López de Prado style, daily-close path): for each day,
    look forward `horizon` days — 1 if price hits the +up% profit barrier BEFORE the
    −dn% stop barrier, 0 if the stop is hit first, and (timeout) the sign of the move
    at the horizon. A trading-relevant target, unlike raw next-day return.
    """
    c = close.values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        entry = c[i]
        if entry <= 0:
            continue
        hi, lo = entry * (1 + up), entry * (1 - dn)
        lab = None
        for k in range(1, horizon + 1):
            v = c[i + k]
            if v >= hi:
                lab = 1; break
            if v <= lo:
                lab = 0; break
        out[i] = lab if lab is not None else (1 if c[i + horizon] > entry else 0)
    return pd.Series(out, index=close.index)


BENCH_SYMBOL = os.getenv("BENCHMARK_SYMBOL", "SPY")   # what an edge must beat


def target_column(df, model_type_label: str = "dip", target: str = "next_ret") -> str:
    """The label column to train AND validate on — one source of truth.

    Training and walk-forward previously chose their labels independently, so an
    excess-trained model would have been graded against the absolute label and
    the verdict that gates deployment would have been meaningless. Both call this."""
    excess = os.getenv("ML_TARGET_EXCESS", "").strip().lower() in ("1", "true", "yes", "on")
    if model_type_label == "breakout":
        c = "breakout_success_5d_excess"
        return c if (excess and c in df.columns) else "breakout_success_5d"
    return "next_ret_excess" if (excess and "next_ret_excess" in df.columns) else target


def extract_outcomes(sym: str, close: pd.Series, volume: pd.Series,
                     target_horizon: int = 1, bench_close: pd.Series = None) -> pd.DataFrame:
    """Build feature-outcome rows for one symbol.

    When `bench_close` is supplied, also emits benchmark-relative outcomes. The
    absolute labels below are contaminated by market drift: over 2024-01→2026-08
    on this watchlist, `next_ret > 0` is true 53.9% of the time versus 48.4% for
    `next_ret > benchmark`. So a constant "predict UP" scores 53.9% accuracy on
    the production label, and 5.5pp of what the model can learn is just "markets
    rise" — a bias it cannot trade on. Excess labels remove it.
    """
    if len(close) < 230:
        return pd.DataFrame()

    features = compute_features_series(close, volume)
    next_ret  = close.pct_change().shift(-target_horizon) * 100
    bounce_5d = (close.shift(-5) > close).astype(int)  # did price rise within 5 days?

    # Breakout success: did price gain 3% within 5 days?
    future_5d_high = close.rolling(5).max().shift(-5)
    breakout_success = (future_5d_high > close * 1.03).astype(int)

    df = features.copy()
    df["next_ret"]  = next_ret
    if bench_close is not None:
        b = bench_close.reindex(close.index).ffill()
        bench_ret = b.pct_change().shift(-target_horizon) * 100
        df["bench_ret"]      = bench_ret
        df["next_ret_excess"] = next_ret - bench_ret
        # 5-day excess: the stock's 5-day move minus the benchmark's over the same days
        fwd5      = (close.shift(-5) / close - 1) * 100
        bench_5d  = (b.shift(-5) / b - 1) * 100
        df["excess_5d"]           = fwd5 - bench_5d
        df["breakout_success_5d_excess"] = (fwd5 > bench_5d + 3.0).astype(int)
    df["bounce_5d"] = bounce_5d
    df["breakout_success_5d"] = breakout_success
    df["tb_label"]  = triple_barrier_labels(close, up=0.05, dn=0.03, horizon=10)
    df["symbol"]    = sym
    df["date"]      = df.index   # survives concat(ignore_index=True); walk-forward needs it

    # Drop rows with NaN (insufficient history) and future (last N rows). Only the
    # CHAMPION features gate row-keeping; the expanded challenger features (longer
    # windows) are allowed to be NaN here and are fillna(0)'d at train time, so
    # adding them never shrinks the production dataset.
    _base = [c for c in FEATURES if c in features.columns]
    df = df.dropna(subset=_base + ["next_ret"])
    df = df.iloc[200:-target_horizon]   # need 200+ bars of history, exclude last N

    return df


# ── Download and build dataset ────────────────────────────────────────────────

def build_dataset(symbols: list, max_years: int = None, batch: int = 50) -> pd.DataFrame:
    """
    Download historical bars using Alpaca directly (bypasses DataClient/Polygon routing).
    Alpaca free tier: ~2,500 bars per symbol (10 years), batch of 50, generous rate limits.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import datetime, timedelta, timezone

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    alpaca_available = bool(api_key and secret_key)

    years  = max_years or 10
    end    = datetime.now(timezone.utc) - timedelta(hours=1)
    start  = end - timedelta(days=365 * years)
    print(f"  Data source: Alpaca ({start.date()} → {end.date()}) | Symbols: {len(symbols)}")

    # Benchmark series, fetched once and passed to every symbol so each row can
    # carry its benchmark-relative outcome alongside the absolute one.
    bench_close = None
    if alpaca_available:
        try:
            _bs = StockHistoricalDataClient(api_key, secret_key).get_stock_bars(
                StockBarsRequest(symbol_or_symbols=BENCH_SYMBOL, timeframe=TimeFrame.Day,
                                 start=start, end=end)).data.get(BENCH_SYMBOL, [])
            if _bs:
                bench_close = pd.Series([b.close for b in _bs],
                                        index=pd.DatetimeIndex([b.timestamp for b in _bs]))
                print(f"  Benchmark: {BENCH_SYMBOL} ({len(bench_close)} bars) — excess labels available")
        except Exception as e:
            print(f"  Benchmark fetch failed ({str(e)[:60]}) — absolute labels only")

    all_rows = []
    processed, skipped = 0, 0

    if alpaca_available:
        alpaca_client = StockHistoricalDataClient(api_key, secret_key)
    else:
        alpaca_client = None
        print("  WARNING: No Alpaca keys — falling back to yfinance")

    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i+batch]
        n_batches = (len(symbols) - 1) // batch + 1
        print(f"  Batch {i//batch+1}/{n_batches} — {len(chunk)} symbols...")

        if alpaca_client:
            try:
                import accounts
                r = alpaca_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start, end=end, feed=accounts.data_feed(),
                ))
                bar_data = r.data
            except Exception as e:
                print(f"    Alpaca batch failed: {e} — skipping")
                skipped += len(chunk)
                continue
        else:
            # yfinance fallback
            try:
                import yfinance as yf
                raw_yf = yf.download(chunk, start=start.strftime("%Y-%m-%d"),
                                     end=end.strftime("%Y-%m-%d"), interval="1d",
                                     auto_adjust=True, progress=False, group_by="ticker")
                bar_data = {}
                for sym in chunk:
                    try:
                        df_ = raw_yf[sym] if isinstance(raw_yf.columns, pd.MultiIndex) else raw_yf
                        if not df_.empty:
                            bar_data[sym] = df_
                    except Exception:
                        pass
            except Exception as e:
                print(f"    yfinance batch failed: {e}")
                skipped += len(chunk)
                continue

        for sym in chunk:
            try:
                if alpaca_client:
                    bars = bar_data.get(sym, [])
                    if not bars:
                        skipped += 1
                        continue
                    close  = pd.Series([b.close  for b in bars],
                                       index=pd.DatetimeIndex([b.timestamp for b in bars]))
                    volume = pd.Series([b.volume for b in bars],
                                       index=close.index)
                else:
                    sym_df = bar_data.get(sym, pd.DataFrame())
                    if sym_df.empty:
                        skipped += 1
                        continue
                    close  = sym_df["Close"].dropna()
                    volume = sym_df["Volume"].dropna() if "Volume" in sym_df.columns else pd.Series(1, index=close.index)

                rows = extract_outcomes(sym, close, volume, bench_close=bench_close)
                if len(rows) > 0:
                    all_rows.append(rows)
                    processed += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    print(f"\n  Dataset: {len(df):,} rows from {processed} symbols ({skipped} skipped)")
    return df


# ── Train model ───────────────────────────────────────────────────────────────

FEATURES = [
    "rsi", "sma50_vs_200", "macd_gap_pct", "pct_b",
    "dip_depth_pct", "volume_ratio", "raw_score", "weighted_score",
    "support_score",   # new: historical price support level detection
    "mom_5d", "mom_20d", "vol_20d",
]

def train_model(df: pd.DataFrame, target: str = "next_ret", model_type_label: str = "dip"):
    """Train an XGBoost classifier to predict bounce or breakout probability."""
    try:
        from xgboost import XGBClassifier
        ModelClass = XGBClassifier
        model_kwargs = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "use_label_encoder": False, "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        ModelClass = GradientBoostingClassifier
        model_kwargs = {"n_estimators": 200, "max_depth": 4,
                        "learning_rate": 0.05, "subsample": 0.8, "random_state": 42}
        model_name = "GradientBoosting"

    print(f"Training {model_name} {model_type_label} classifier...")

    # ML_TARGET_EXCESS=true trains against the BENCHMARK-RELATIVE outcome instead
    # of the absolute one. Default off so the production model is unchanged until
    # a retrain is explicitly run and compared. See extract_outcomes() for why the
    # absolute label is drift-contaminated.
    col = target_column(df, model_type_label, target)
    y = (df[col] > 0).astype(int)
    print(f"  target: {col}  (positive class {y.mean()*100:.1f}%)"
          + ("" if col.endswith("excess") else "  [absolute — includes market drift]"))

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, shuffle=True)

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = ModelClass(**model_kwargs)
    try:
        model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
    except TypeError:
        model.fit(X_tr_s, y_tr)

    # Evaluate
    proba    = model.predict_proba(X_te_s)[:, 1]
    auc      = roc_auc_score(y_te, proba)
    baseline = y.mean()

    print(f"  AUC-ROC: {auc:.4f}  (baseline: {baseline:.4f})")
    print(f"  Lift over baseline: {(auc - 0.5) / (1 - 0.5) * 100:.1f}%")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = dict(sorted(zip(FEATURES, model.feature_importances_.tolist()),
                           key=lambda x: x[1], reverse=True))
    else:
        imp = {}

    # Calibration check — does P(bounce) > 0.6 actually win 60%+ of the time?
    buckets = {}
    for threshold in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        mask  = proba >= threshold
        if mask.sum() > 0:
            actual_win = y_te[mask].mean()
            buckets[f"p>={threshold:.2f}"] = {
                "n": int(mask.sum()),
                "actual_win_rate": round(float(actual_win), 3),
                "predicted_threshold": threshold,
            }

    return model, scaler, {
        "model_type":    model_name,
        "auc_roc":       round(auc, 4),
        "baseline":      round(float(baseline), 4),
        "feature_importance": imp,
        "calibration":   buckets,
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=None,
                        help="Limit to last N years (default: max available)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test on 50 symbols")
    parser.add_argument("--type", choices=["dip", "breakout"], default="dip",
                        help="Model type to train")
    args = parser.parse_args()

    print(f"=== ML {args.type.capitalize()} Confidence Model Trainer ===\n")

    # Build universe
    print("Fetching stock universe...")
    sp500   = fetch_sp500_symbols()
    extra   = fetch_russell1000_symbols()
    symbols = list(dict.fromkeys(sp500 + extra))   # deduplicate, preserve order

    if args.quick:
        symbols = symbols[:50]
        print(f"  Quick mode: using {len(symbols)} symbols")
    else:
        print(f"  Universe: {len(symbols)} symbols (S&P 500 + Russell 1000 additions)")

    # Build dataset
    print(f"\nBuilding dataset (max history = {'max' if not args.years else str(args.years)+'y'})...")
    df = build_dataset(symbols, max_years=args.years)

    if df.empty or len(df) < 1000:
        print("ERROR: Insufficient data to train model")
        return

    # Filter for training context
    if args.type == "breakout":
        # Breakout model: train on setups where price is above SMA50 or has volume
        # Filter rows to only include "breakout-like" environments to reduce noise
        df = df[(df["weighted_score"] >= 1) | (df["volume_ratio"] >= 1.2)]
        print(f"  Filtered for breakout setups: {len(df):,} rows remaining")
    else:
        # Dip model: train on pullbacks
        df = df[df["dip_depth_pct"] > 0]
        print(f"  Filtered for dip setups: {len(df):,} rows remaining")

    print(f"\nDataset summary:")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Unique symbols:   {df['symbol'].nunique()}")
    date_range = f"{df.index.min() if hasattr(df.index, 'min') else 'n/a'}"
    if args.type == "breakout":
        print(f"  5-day breakout win rate (>3%): {df['breakout_success_5d'].mean():.1%}")
    else:
        print(f"  Next-day bounce win rate: {(df['next_ret'] > 0).mean():.1%}")

    # Train
    print()
    model, scaler, report = train_model(df, model_type_label=args.type)

    # Walk-forward validation — strictly out-of-sample, gates deployment.
    # The shuffled split above leaks future rows into training; this is the
    # honest measurement of whether the model generalises forward in time.
    print("\nWalk-forward validation (out-of-sample):")
    from walk_forward import validate
    _wf_col = target_column(df, args.type)
    y_wf = (df[_wf_col] > 0).astype(int)
    print(f"  validating against: {_wf_col}")
    wf = validate(df.reset_index(drop=True), FEATURES, y_wf.reset_index(drop=True))
    report["walk_forward"] = wf
    print(f"  Verdict: {wf['verdict']}"
          + (f"  (OOS AUC {wf['mean_oos_auc']} ± {wf['std_oos_auc']}, "
             f"IS−OOS gap {wf['is_oos_gap']})" if "mean_oos_auc" in wf else
             f"  ({wf.get('reason', '')})"))

    # Save
    os.makedirs("models", exist_ok=True)
    filename = f"{args.type}_confidence.pkl"
    report_name = f"{args.type}_training_report.json"

    if wf["verdict"] == "OVERFITTED":
        filename = f"{args.type}_confidence_rejected.pkl"
        print(f"\n*** DEPLOYMENT BLOCKED: walk-forward verdict OVERFITTED — "
              f"keeping existing production model, saving rejected model to models/{filename} ***")

    with open(f"models/{filename}", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler,
                     "features": FEATURES, "trained_at": datetime.now(ET).isoformat(),
                     "walk_forward_verdict": wf["verdict"]}, f)

    with open(f"models/{report_name}", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved to models/{filename}")
    print(f"\nCalibration (does confidence = actual win rate?):")
    for bucket, stats in report["calibration"].items():
        print(f"  {bucket}: {stats['n']:,} samples → actual win {stats['actual_win_rate']:.1%}")

    print(f"\nTop features by importance:")
    for feat, imp in list(report["feature_importance"].items())[:6]:
        print(f"  {feat:20s}: {imp:.4f}")


if __name__ == "__main__":
    main()
