"""
data_client.py — unified market data abstraction.

Providers (selected automatically from environment):
  Polygon.io  — if POLYGON_API_KEY is set
                POLYGON_TIER=free    → 5 req/min, sequential with rate limiting
                POLYGON_TIER=starter → unlimited, parallel downloads
  yfinance    — fallback (no API key needed); batch downloads

All methods return a pd.DataFrame with a (Symbol, OHLCV) MultiIndex so that
every script accesses data the same way regardless of provider:

    raw = client.get_daily_bars(symbols)
    close = client.close(raw, "AAPL")   # → pd.Series

Usage:
    from data_client import DataClient
    client = DataClient()
    print(client.provider)              # "polygon" or "yfinance"

    raw   = client.get_daily_bars(["AAPL", "NVDA"])
    intra = client.get_intraday_bars(["AAPL"], interval="15m")
"""
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_to_dates(period: str):
    """Convert a yfinance-style period string to (start, end) date strings."""
    end   = datetime.now(ET)
    units = {"d": 1, "w": 7, "mo": 30, "y": 365}
    for suffix, days_per_unit in units.items():
        if period.endswith(suffix):
            n     = int(period[: -len(suffix)])
            start = end - timedelta(days=n * days_per_unit)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    raise ValueError(f"Unrecognised period: {period!r}")


def _build_multiindex(frames: dict) -> pd.DataFrame:
    """
    Build a (Symbol, OHLCV) MultiIndex DataFrame from a dict of per-symbol DataFrames.
    Matches the output format of yf.download(..., group_by='ticker').
    """
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined.sort_index(inplace=True)
    return combined


def _yf_sym(raw: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Extract per-symbol OHLCV DataFrame from a yfinance/DataClient multi-ticker result."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            return raw[sym].dropna(how="all")
        return raw.dropna(how="all")
    except (KeyError, TypeError):
        return pd.DataFrame()


# ── Polygon provider ──────────────────────────────────────────────────────────

class _PolygonProvider:
    """Wraps polygon-api-client with rate limiting and output normalisation."""

    FREE_DELAY = 12.5   # seconds between calls at free tier (5/min with buffer)

    def __init__(self, api_key: str, tier: str):
        try:
            from polygon import RESTClient
        except ImportError:
            raise ImportError("Run: pip install polygon-api-client")
        self.client   = RESTClient(api_key=api_key)
        self.tier     = tier          # "free" or "starter"
        self.parallel = (tier == "starter")

    def _fetch_daily(self, sym: str, start: str, end: str) -> pd.DataFrame:
        try:
            aggs = self.client.get_aggs(
                ticker=sym, multiplier=1, timespan="day",
                from_=start, to=end,
                adjusted=True, limit=50_000,
            )
            if not aggs:
                return pd.DataFrame()
            records = [
                {
                    "Date":   pd.Timestamp(a.timestamp, unit="ms", tz="UTC").normalize(),
                    "Open":   a.open, "High": a.high, "Low": a.low,
                    "Close":  a.close, "Volume": a.volume,
                }
                for a in aggs
            ]
            df = pd.DataFrame(records).set_index("Date")
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"  Polygon: failed {sym} — {e}")
            return pd.DataFrame()

    def _fetch_intraday(self, sym: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """interval: '15m', '30m', '1h' etc."""
        mult_map = {"1m": (1,"minute"), "5m": (5,"minute"), "15m": (15,"minute"),
                    "30m": (30,"minute"), "1h": (1,"hour"), "60m": (1,"hour")}
        mult, span = mult_map.get(interval, (15, "minute"))
        try:
            aggs = self.client.get_aggs(
                ticker=sym, multiplier=mult, timespan=span,
                from_=start, to=end,
                adjusted=True, limit=50_000,
            )
            if not aggs:
                return pd.DataFrame()
            records = [
                {
                    "Datetime": pd.Timestamp(a.timestamp, unit="ms", tz="US/Eastern"),
                    "Open":  a.open, "High": a.high, "Low": a.low,
                    "Close": a.close, "Volume": a.volume,
                }
                for a in aggs
            ]
            df = pd.DataFrame(records).set_index("Datetime")
            return df
        except Exception as e:
            print(f"  Polygon intraday: failed {sym} — {e}")
            return pd.DataFrame()

    def _fetch_indicators(self, sym: str, short_bars: int = 25) -> dict:
        """
        Fetch pre-computed indicators from Polygon + a small bar window for
        Bollinger Bands and volume (endpoints Polygon doesn't cover).

        Returns dict with keys: rsi, macd_val, macd_sig, sma50, sma200, sma20,
                                 close (latest), bars_df (25 days for Bollinger/vol)
        """
        result = {}
        c = self.client

        try:
            r = list(c.get_rsi(sym, timespan="day", window=14, series_type="close", limit=1))
            result["rsi"] = float(r[0].value) if r else None
        except Exception: result["rsi"] = None

        try:
            r = list(c.get_macd(sym, timespan="day", short_window=12,
                                 long_window=26, signal_window=9,
                                 series_type="close", limit=1))
            if r:
                result["macd_val"] = float(r[0].value)
                result["macd_sig"] = float(r[0].signal)
        except Exception: result["macd_val"] = result["macd_sig"] = None

        for key, window in [("sma50", 50), ("sma200", 200), ("sma20", 20)]:
            try:
                r = list(c.get_sma(sym, timespan="day", window=window,
                                    series_type="close", limit=1))
                result[key] = float(r[0].value) if r else None
            except Exception: result[key] = None

        # Recent bars for Bollinger Bands + volume signal (Polygon has no BB endpoint)
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=short_bars * 2)   # buffer for weekends
        df    = self._fetch_daily(sym, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        result["bars_df"] = df.tail(short_bars) if not df.empty else pd.DataFrame()
        result["close"]   = float(df["Close"].iloc[-1]) if not df.empty else None

        return result

    def get_indicators_bulk(self, symbols: list) -> dict:
        """
        Fetch pre-computed indicators for many symbols.
        Starter tier: parallel (fast).
        Free tier:    sequential with rate limiting (slow — consider upgrading).

        Returns {sym: {rsi, macd_val, macd_sig, sma50, sma200, sma20, close, bars_df}}
        """
        from datetime import timezone

        if self.parallel:
            print(f"  [STARTER] Fetching indicators for {len(symbols)} symbols in parallel...")
            results = {}
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(self._fetch_indicators, s): s for s in symbols}
                for fut in as_completed(futures):
                    sym = futures[fut]
                    results[sym] = fut.result()
            return results

        # Free tier — sequential with rate limiting
        # Each symbol = ~6 API calls → use delay between symbols (not calls)
        print(f"  [FREE] Fetching indicators for {len(symbols)} symbols (~{len(symbols)*6//5//60+1} min)...")
        results = {}
        for i, sym in enumerate(symbols):
            results[sym] = self._fetch_indicators(sym)
            if i < len(symbols) - 1:
                time.sleep(self.FREE_DELAY * 6)   # ~6 calls per symbol
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(symbols)} done...")
        return results

    def get_daily_bars(self, symbols: list, period: str = "2y") -> pd.DataFrame:
        start, end = _period_to_dates(period)
        print(f"  [{self.tier.upper()}] Fetching {len(symbols)} symbols via Polygon ({start} → {end})")

        if self.parallel:
            frames = {}
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(self._fetch_daily, s, start, end): s for s in symbols}
                for fut in as_completed(futures):
                    sym = futures[fut]
                    df  = fut.result()
                    if not df.empty:
                        frames[sym] = df
            return _build_multiindex(frames)

        # Free tier — sequential with rate limiting
        frames = {}
        for i, sym in enumerate(symbols):
            df = self._fetch_daily(sym, start, end)
            if not df.empty:
                frames[sym] = df
            if i < len(symbols) - 1:
                time.sleep(self.FREE_DELAY)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(symbols)} done...")
        return _build_multiindex(frames)

    def get_intraday_bars(self, symbols: list, interval: str = "15m", period: str = "1d") -> pd.DataFrame:
        start, end = _period_to_dates(period)
        frames = {}
        if self.parallel:
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(self._fetch_intraday, s, interval, start, end): s for s in symbols}
                for fut in as_completed(futures):
                    sym = futures[fut]
                    df  = fut.result()
                    if not df.empty:
                        frames[sym] = df
        else:
            for i, sym in enumerate(symbols):
                df = self._fetch_intraday(sym, interval, start, end)
                if not df.empty:
                    frames[sym] = df
                if i < len(symbols) - 1:
                    time.sleep(self.FREE_DELAY)
        return _build_multiindex(frames)


# ── yfinance provider ─────────────────────────────────────────────────────────

class _YFinanceProvider:
    """Wraps yfinance batch downloads with retry logic."""

    def get_daily_bars(self, symbols: list, period: str = "2y",
                       batch: int = 200) -> pd.DataFrame:
        frames = []
        for i in range(0, len(symbols), batch):
            chunk = symbols[i : i + batch]
            for attempt in range(3):
                try:
                    raw = yf.download(
                        chunk, period=period, interval="1d",
                        auto_adjust=True, progress=False, group_by="ticker",
                    )
                    if raw.empty and attempt < 2:
                        time.sleep(5)
                        continue
                    frames.append(raw)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"  yfinance: batch {i//batch+1} failed — {e}")
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1)
        return combined.loc[:, ~combined.columns.duplicated()]

    def get_intraday_bars(self, symbols: list, interval: str = "15m",
                          period: str = "1d") -> pd.DataFrame:
        for attempt in range(3):
            try:
                raw = yf.download(
                    symbols, period=period, interval=interval,
                    auto_adjust=True, progress=False, group_by="ticker",
                )
                if not raw.empty:
                    return raw
                if attempt < 2:
                    time.sleep(5)
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"  yfinance intraday failed — {e}")
        return pd.DataFrame()


# ── Alpaca provider ───────────────────────────────────────────────────────────

class _AlpacaProvider:
    """Wraps alpaca-py historical data client."""

    def __init__(self, api_key: str, secret_key: str):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            self.client = StockHistoricalDataClient(api_key, secret_key)
        except ImportError:
            raise ImportError("Run: pip install alpaca-py")

    def get_daily_bars(self, symbols: list, period: str = "2y") -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        start, end = _period_to_dates(period)
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        try:
            import accounts
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                                   start=start_dt, end=end_dt, feed=accounts.data_feed())
            bars = self.client.get_stock_bars(req)
            if not bars or bars.df.empty:
                return pd.DataFrame()

            df = bars.df.copy()
            # Standardise columns and index to match yfinance MultiIndex (Ticker, OHLCV)
            df.index = df.index.set_levels(df.index.levels[1].tz_localize(None), level=1)
            df.columns = [c.capitalize() for c in df.columns]
            
            # Reshape: symbol is currently level 0 index. We want it as level 0 columns.
            # df has index [symbol, timestamp] and columns [Open, High, Low, Close, Volume, ...]
            # We want columns [symbol, OHLCV]
            df = df.unstack(level=0).swaplevel(0, 1, axis=1).sort_index(axis=1)
            return df
        except Exception as e:
            print(f"  Alpaca data: failed — {e}")
            return pd.DataFrame()

    def get_intraday_bars(self, symbols: list, interval: str = "15m", period: str = "1d") -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        # NB: the 2nd arg must be a TimeFrameUnit — passing TimeFrame.Minute (a
        # TimeFrame instance) serialised to "151Min" → "period > 59" API error.
        tf_map = {"1m": TimeFrame(1, TimeFrameUnit.Minute), "5m": TimeFrame(5, TimeFrameUnit.Minute),
                  "15m": TimeFrame(15, TimeFrameUnit.Minute), "1h": TimeFrame(1, TimeFrameUnit.Hour)}
        tf = tf_map.get(interval, TimeFrame(15, TimeFrameUnit.Minute))

        start, end = _period_to_dates(period)
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.now(timezone.utc)

        try:
            import accounts
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf,
                                   start=start_dt, end=end_dt, feed=accounts.data_feed())
            bars = self.client.get_stock_bars(req)
            if not bars or bars.df.empty:
                return pd.DataFrame()
            
            df = bars.df.copy()
            df.index = df.index.set_levels(df.index.levels[1].tz_convert("US/Eastern").tz_localize(None), level=1)
            df.columns = [c.capitalize() for c in df.columns]
            df = df.unstack(level=0).swaplevel(0, 1, axis=1).sort_index(axis=1)
            return df
        except Exception as e:
            print(f"  Alpaca intraday: failed — {e}")
            return pd.DataFrame()


# ── Public interface ──────────────────────────────────────────────────────────

class DataClient:
    """
    Unified market data client. Automatically selects the best available provider.

    Environment variables:
      POLYGON_API_KEY   — enables Polygon.io (get free key at polygon.io)
      POLYGON_TIER      — "free" (default, 5 req/min) or "starter" (unlimited)
    """

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        polygon_key  = os.getenv("POLYGON_API_KEY", "").strip()
        polygon_tier = os.getenv("POLYGON_TIER", "free").lower()
        alpaca_key   = os.getenv("ALPACA_API_KEY", "").strip()
        alpaca_sec   = os.getenv("ALPACA_SECRET_KEY", "").strip()

        # Is the system clock in 2026? If so, prioritize Alpaca (simulated future)
        self.is_future = datetime.now(ET).year >= 2026

        if polygon_key:
            try:
                self._provider = _PolygonProvider(polygon_key, polygon_tier)
                self.provider  = f"polygon:{polygon_tier}"
                if self.is_future:
                    print("  Note: System in future (2026+) — Polygon may fail, will fallback to Alpaca")
            except ImportError:
                print("polygon-api-client not installed")
                self._provider = None

        if not getattr(self, "_provider", None) or self.is_future:
            if alpaca_key and alpaca_sec:
                try:
                    self._provider = _AlpacaProvider(alpaca_key, alpaca_sec)
                    self.provider  = "alpaca"
                except ImportError:
                    print("alpaca-py not installed")

        if not getattr(self, "_provider", None):
            self._provider = _YFinanceProvider()
            self.provider  = "yfinance"

    # ── Core methods ──────────────────────────────────────────────────────────

    def get_daily_bars(self, symbols, period: str = "2y",
                       batch: int = 200) -> pd.DataFrame:
        """
        Download daily OHLCV bars.

        Returns a DataFrame with (Symbol, OHLCV) MultiIndex columns.
        Access: raw[sym]["Close"]  or  client.close(raw, sym)
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        if hasattr(self._provider, "get_daily_bars"):
            if isinstance(self._provider, _YFinanceProvider):
                return self._provider.get_daily_bars(symbols, period=period, batch=batch)
            return self._provider.get_daily_bars(symbols, period=period)
        return pd.DataFrame()

    def get_intraday_bars(self, symbols, interval: str = "15m",
                          period: str = "1d") -> pd.DataFrame:
        """
        Download intraday OHLCV bars.

        interval: '1m', '5m', '15m', '30m', '1h'
        Returns a DataFrame with (Symbol, OHLCV) MultiIndex columns.
        Note: yfinance free tier only has intraday data for the last 60 days.
              Polygon Starter provides 15-min delayed intraday.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        return self._provider.get_intraday_bars(symbols, interval=interval, period=period)

    # ── Convenience accessors ─────────────────────────────────────────────────

    def sym_df(self, raw: pd.DataFrame, sym: str) -> pd.DataFrame:
        """Extract per-symbol OHLCV DataFrame from a get_daily_bars result."""
        return _yf_sym(raw, sym)

    def close(self, raw: pd.DataFrame, sym: str) -> pd.Series:
        """Extract the Close price series for a symbol."""
        df = self.sym_df(raw, sym)
        return df["Close"].dropna() if "Close" in df.columns else pd.Series()

    def __repr__(self):
        return f"DataClient(provider={self.provider!r})"
