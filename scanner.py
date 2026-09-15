import asyncio, math, time, logging
from dataclasses import dataclass
import aiohttp
import numpy as np

BYBIT_BASE = "https://api.bybit.com"
BINGX_BASE = "https://open-api.bingx.com"

@dataclass
class Signal:
    exchange: str
    symbol: str
    timeframe: str
    price: float
    current_rsi: float
    previous_rsi: float
    oversold_seen: bool
    quote_volume: float
    previous_pivot_time: str
    latest_pivot_time: str
    latest_pivot_ts: int
    confirmed: bool

def rsi_wilder(closes, period=14):
    x = np.asarray(closes, dtype=float)
    if len(x) < period + 2:
        return np.full(len(x), np.nan)
    delta = np.diff(x)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(x), np.nan)
    avg_loss = np.full(len(x), np.nan)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, len(x)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    out = 100 - (100 / (1 + rs))
    out[(avg_loss == 0) & np.isfinite(avg_gain)] = 100
    return out

def pivot_lows(values, left=3, right=3):
    v = np.asarray(values, dtype=float)
    out = []
    for i in range(left, len(v)-right):
        window = v[i-left:i+right+1]
        if np.isfinite(v[i]) and v[i] == np.nanmin(window) and np.sum(window == v[i]) == 1:
            out.append(i)
    return out

def find_divergence(
    candles,
    rsi_period,
    left,
    right,
    max_gap,
    min_ll_pct,
    min_rsi_diff,
    oversold,
):
    # candles: oldest -> newest
    # (ts, open, high, low, close, volume, quote_volume)

    if len(candles) < rsi_period + left + right + 20:
        return None

    closes = [c[4] for c in candles]
    lows = [c[3] for c in candles]

    rsi = rsi_wilder(closes, rsi_period)
    pivots = pivot_lows(lows, left, right)

    if len(pivots) < 2:
        return None

    # Нас интересуют только свежие дивергенции.
    # Последний минимум дивергенции должен быть не старше 10 свечей.
    MAX_SIGNAL_AGE = 100

    candidates = []

    # Проверяем несколько последних pivot-low,
    # а не только самый последний.
    recent_pivots = pivots[-8:]

    for b in reversed(recent_pivots):

        signal_age = (len(candles) - 1) - b

        if signal_age > MAX_SIGNAL_AGE:
            continue

        price_b = lows[b]
        rsi_b = rsi[b]

        if not np.isfinite(rsi_b):
            continue

        # Ищем предыдущий минимум перед b
        for a in reversed(pivots):

            if a >= b:
                continue

            gap = b - a

            if gap > max_gap:
                break

            price_a = lows[a]
            rsi_a = rsi[a]

            if not np.isfinite(rsi_a):
                continue

            # Цена делает Lower Low
            lower_low = (
                price_b
                < price_a * (1 - min_ll_pct / 100)
            )

            # RSI делает Higher Low
            higher_rsi_low = (
                rsi_b >= rsi_a + min_rsi_diff
            )

            if not (lower_low and higher_rsi_low):
                continue

            # Между минимумами RSI должен побывать
            # ниже уровня перепроданности
            rsi_slice = rsi[a:b + 1]

            finite_rsi = [
                float(x)
                for x in rsi_slice
                if np.isfinite(x)
            ]

            if not finite_rsi:
                continue

            oversold_seen = min(finite_rsi) < oversold

            if not oversold_seen:
                continue

            current_rsi = float(rsi[-1])

            if not np.isfinite(current_rsi):
                continue

            # Подтверждение:
            # RSI сейчас уже выше уровня oversold
            confirmed = current_rsi > oversold

            if not confirmed:
                continue

            # Чем свежее сигнал — тем выше приоритет
            # При равной свежести берем больший рост RSI
            candidates.append(
                {
                    "a": a,
                    "b": b,
                    "age": signal_age,
                    "rsi_diff": float(rsi_b - rsi_a),
                }
            )

            # Для этого b достаточно первого подходящего
            # предыдущего минимума
            break

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["age"],
            -x["rsi_diff"],
        )
    )

    best = candidates[0]

    a = best["a"]
    b = best["b"]

    price_a = lows[a]
    price_b = lows[b]

    rsi_a = float(rsi[a])
    rsi_b = float(rsi[b])

    current_rsi = float(rsi[-1])

    latest_ts = int(candles[b][0])

    latest_pivot_time = time.strftime(
        "%Y-%m-%d %H:%M UTC",
        time.gmtime(latest_ts / 1000),
    )

    previous_pivot_time = time.strftime(
        "%Y-%m-%d %H:%M UTC",
        time.gmtime(int(candles[a][0]) / 1000),
    )

    quote_volume = (
        float(sum(c[6] for c in candles[-24:]))
        if len(candles) >= 24
        else 0.0
    )

    logging.info(
        "SIGNAL FOUND | "
        "price %.8f -> %.8f | "
        "RSI %.1f -> %.1f | "
        "current RSI %.1f | "
        "age %d candles",
        price_a,
        price_b,
        rsi_a,
        rsi_b,
        current_rsi,
        best["age"],
    )

    return {
        "pivot_index": b,
        "signal": {
            "current_rsi": current_rsi,
            "previous_rsi": rsi_a,
            "oversold_seen": True,
            "confirmed": True,
            "latest_pivot_ts": latest_ts,
            "latest_pivot_time": latest_pivot_time,
            "previous_pivot_time": previous_pivot_time,
            "price": float(candles[-1][4]),
            "quote_volume": quote_volume,
        },
    }


class ExchangeScanner:
    def __init__(self, name):
        self.name = name.lower()

    async def _get_json(self, session, url, params=None, headers=None):
        for attempt in range(3):
            try:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=20
                ) as r:
                    data = await r.json()

                    if r.status == 200:
                        return data

                    if r.status in (429, 418):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue

                    raise RuntimeError(
                        f"{self.name} HTTP {r.status}: {data}"
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 2:
                    raise

                await asyncio.sleep(1.0 * (attempt + 1))

        raise RuntimeError("request failed")

    async def get_usdt_symbols(self, session, min_volume, max_symbols):
        if self.name == "bybit":
            data = await self._get_json(session, BYBIT_BASE + "/v5/market/tickers",
                                        {"category": "linear"})
            rows = data["result"]["list"]
            out = []
            for x in rows:
                sym = x["symbol"]
                if not sym.endswith("USDT"):
                    continue
                try:
                    turnover = float(x.get("turnover24h", 0))
                except Exception:
                    turnover = 0
                if turnover >= min_volume:
                    out.append((sym, turnover))
            out.sort(key=lambda z: z[1], reverse=True)
            return [s for s, _ in out[:max_symbols]]

        # BingX perpetual USDT contracts
        data = await self._get_json(session, BINGX_BASE + "/openApi/swap/v2/quote/ticker")
        rows = data.get("data", [])
        out = []
        for x in rows:
            sym = x.get("symbol", "")
            if not sym.endswith("-USDT"):
                continue
            try:
                volume = float(x.get("volume", 0))
                last = float(x.get("lastPrice", 0))
                turnover = volume * last
            except Exception:
                turnover = 0
            if turnover >= min_volume:
                out.append((sym, turnover))
        out.sort(key=lambda z: z[1], reverse=True)
        return [s for s, _ in out[:max_symbols]]

    async def get_klines(self, session, symbol, timeframe, limit=220):
        if self.name == "bybit":
            interval = {"1h": "60", "4h": "240", "1d": "D"}[timeframe]
            data = await self._get_json(session, BYBIT_BASE + "/v5/market/kline", {
                "category": "linear", "symbol": symbol, "interval": interval, "limit": limit
            })
            rows = data["result"]["list"]
            rows = list(reversed(rows))
            return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6])) for x in rows]

        interval = timeframe
        data = await self._get_json(session, BINGX_BASE + "/openApi/swap/v3/quote/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })
        rows = data.get("data", [])
        rows = sorted(rows, key=lambda x: int(x[0]))
        return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7]) if len(x) > 7 else float(x[5]) * float(x[4])) for x in rows]

    async def scan_symbols(self, session, symbols, timeframe, **kwargs):
        sem = asyncio.Semaphore(8 if self.name == "bybit" else 5)

        async def one(symbol):
            async with sem:
                try:
                    candles = await self.get_klines(session, symbol, timeframe)
                    result = find_divergence(candles, **kwargs)
                    if not result:
                        return None
                    s = result["signal"]
                    return Signal(
                        exchange=self.name, symbol=symbol, timeframe=timeframe,
                        price=s["price"], current_rsi=s["current_rsi"],
                        previous_rsi=s["previous_rsi"], oversold_seen=s["oversold_seen"],
                        quote_volume=s["quote_volume"],
                        previous_pivot_time=s["previous_pivot_time"],
                        latest_pivot_time=s["latest_pivot_time"],
                        latest_pivot_ts=s["latest_pivot_ts"],
                        confirmed=s["confirmed"],
                    )
                except Exception as e:
                    logging.exception(
                        "ERROR scanning %s %s %s: %s",
                        self.name,
                        symbol,
                        timeframe,
                        e,
                    )
                    return None

        results = await asyncio.gather(*(one(s) for s in symbols))
        return [x for x in results if x is not None]
