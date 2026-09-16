import os, json, time, asyncio, logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from scanner import ExchangeScanner, Signal

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EXCHANGES = [x.strip().lower() for x in os.getenv("EXCHANGES", "bybit,bingx").split(",") if x.strip()]
TIMEFRAMES = [x.strip() for x in os.getenv("TIMEFRAMES", "1d,4h,1h").split(",") if x.strip()]
SCAN_EVERY_MINUTES = int(os.getenv("SCAN_EVERY_MINUTES", "15"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
MIN_24H_USDT_VOLUME = float(os.getenv("MIN_24H_USDT_VOLUME", "1000000"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "3"))
MAX_PIVOT_GAP = int(os.getenv("MAX_PIVOT_GAP", "30"))
MIN_PRICE_LOWER_LOW_PCT = float(os.getenv("MIN_PRICE_LOWER_LOW_PCT", "0.5"))
MIN_RSI_HIGHER_LOW = float(os.getenv("MIN_RSI_HIGHER_LOW", "5.0"))
OVERSOLD_RSI = float(os.getenv("OVERSOLD_RSI", "30"))
STATE_FILE = Path(os.getenv("STATE_FILE", "sent_signals.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_CHAT_ID:
    raise SystemExit("TELEGRAM_CHAT_ID is not set")

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

async def telegram_send(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    while True:
        async with session.post(url, json=payload, timeout=20) as r:
            if r.status == 200:
                return

            try:
                body = await r.json()
            except Exception:
                body = {"description": await r.text()}

            if r.status == 429:
                retry_after = body.get("parameters", {}).get("retry_after", 5)
                logging.warning(
                    "Telegram rate limit. Waiting %s seconds",
                    retry_after,
                )
                await asyncio.sleep(retry_after + 1)
                continue

            raise RuntimeError(
                f"Telegram HTTP {r.status}: {body}"
            )

def fmt_price(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")

def signal_text(s: Signal):
    icon = "🟢" if s.confirmed else "🟡"
    stage = "ПОДТВЕРЖДЁННАЯ" if s.confirmed else "ФОРМИРУЕТСЯ"

    rsi_diff = s.latest_rsi - s.previous_rsi

    tv_symbol = s.symbol.replace("-", "")
    tv_exchange = "BYBIT" if s.exchange.lower() == "bybit" else "BINGX"
    tradingview_url = f"https://www.tradingview.com/chart/?symbol={tv_exchange}%3A{tv_symbol}.P"
    
    return (
        f"{icon} {stage} BULLISH DIVERGENCE\n\n"
        f"Биржа: {s.exchange.upper()}\n"
        f"Монета: {s.symbol}\n"
        f"ТФ: {s.timeframe}\n"
        f"Цена: {fmt_price(s.price)}\n\n"

        f"Цена: Lower Low ✅\n"
        f"RSI: Higher Low ✅\n\n"

        f"RSI первого минимума: {s.previous_rsi:.1f}\n"
        f"RSI второго минимума: {s.latest_rsi:.1f}\n"
        f"Разница RSI: {rsi_diff:+.1f}\n"
        f"RSI сейчас: {s.current_rsi:.1f}\n\n"

        f"24h объём: ${s.quote_volume:,.0f}\n"
        f"Пивоты: {s.previous_pivot_time} → {s.latest_pivot_time}\n\n"

        f"📊 График TradingView:\n"
        f"{tradingview_url}\n\n"
        
        f"⚠️ Это технический сигнал, а не гарантия разворота."
    )

async def scan_once(state):
    scanners = [ExchangeScanner(x) for x in EXCHANGES]
    all_signals =[]
    
    async with aiohttp.ClientSession() as session:
        for scanner in scanners:
            try:
                symbols = await scanner.get_usdt_symbols(
                    session,
                    max_symbols=MAX_SYMBOLS,
                    min_volume=MIN_24H_USDT_VOLUME,
                )
                logging.info("%s: %d symbols selected", scanner.name, len(symbols))

                for tf in TIMEFRAMES:
                    signals = await scanner.scan_symbols(
                        session, symbols, tf,
                        rsi_period=RSI_PERIOD,
                        pivot_left=PIVOT_LEFT,
                        pivot_right=PIVOT_RIGHT,
                        max_pivot_gap=MAX_PIVOT_GAP,
                        min_price_lower_low_pct=MIN_PRICE_LOWER_LOW_PCT,
                        min_rsi_higher_low=MIN_RSI_HIGHER_LOW,
                        oversold_rsi=OVERSOLD_RSI,
                    )
                    all_signals.extend(signals)
                    logging.info("%s %s: %d signals", scanner.name, tf, len(signals))

            except Exception:
                logging.exception("Scanner failed")   
                                                
        for s in all_signals:
            key = f"{s.exchange}:{s.symbol}:{s.timeframe}:{s.latest_pivot_ts}"
            if key in state:
                continue
            await telegram_send(session, signal_text(s))
            state[key] = int(time.time())
            save_state(state)

async def main():
    state = load_state()
    async with aiohttp.ClientSession() as session:
        
        while True:
            try:
                await scan_once(state)
            except Exception:
                logging.exception("Scan failed")
            await asyncio.sleep(SCAN_EVERY_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(main())
