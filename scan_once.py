
import os, json, asyncio, logging
from pathlib import Path
import aiohttp
from dotenv import load_dotenv
from scanner import ExchangeScanner, Signal

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
EXCHANGES = [x.strip().lower() for x in os.getenv("EXCHANGES", "bybit,bingx").split(",") if x.strip()]
TIMEFRAMES = [x.strip() for x in os.getenv("TIMEFRAMES", "1d,4h,1h").split(",") if x.strip()]
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "100"))
MIN_24H_USDT_VOLUME = float(os.getenv("MIN_24H_USDT_VOLUME", "5000000"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "3"))
MAX_PIVOT_GAP = int(os.getenv("MAX_PIVOT_GAP", "80"))
MIN_PRICE_LOWER_LOW_PCT = float(os.getenv("MIN_PRICE_LOWER_LOW_PCT", "0.5"))
MIN_RSI_HIGHER_LOW = float(os.getenv("MIN_RSI_HIGHER_LOW", "2.0"))
OVERSOLD_RSI = float(os.getenv("OVERSOLD_RSI", "30"))

STATE_FILE = Path("sent_signals.json")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

async def send_telegram(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20) as r:
        if r.status != 200:
            raise RuntimeError(await r.text())

def fmt_price(x):
    if x >= 100: return f"{x:.2f}"
    if x >= 1: return f"{x:.4f}"
    if x >= 0.01: return f"{x:.5f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")

def make_text(s):
    icon = "🟢" if s.confirmed else "🟡"
    stage = "ПОДТВЕРЖДЁННАЯ" if s.confirmed else "РАННЯЯ"
    return (
        f"{icon} {stage} BULLISH DIVERGENCE\n\n"
        f"Биржа: {s.exchange.upper()}\n"
        f"Монета: {s.symbol}\n"
        f"ТФ: {s.timeframe}\n"
        f"Цена: {fmt_price(s.price)}\n"
        f"RSI(14): {s.current_rsi:.1f}\n\n"
        f"Цена: Lower Low ✅\n"
        f"RSI: Higher Low ✅\n"
        f"Предыдущий RSI: {s.previous_rsi:.1f}\n"
        f"RSI <30: {'✅' if s.oversold_seen else '❌'}\n"
        f"RSI сейчас >30: {'✅' if s.current_rsi > OVERSOLD_RSI else '❌'}\n"
        f"24h объём: ${s.quote_volume:,.0f}\n\n"
        f"⚠️ Технический сигнал, не гарантия разворота."
    )

async def main():
    state = load_state()
    async with aiohttp.ClientSession() as session:
        for exchange in EXCHANGES:
            scanner = ExchangeScanner(exchange)
            symbols = await scanner.get_usdt_symbols(session, MIN_24H_USDT_VOLUME, MAX_SYMBOLS)
            logging.info("%s: %s symbols", exchange, len(symbols))
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
                for s in signals:
                    key = f"{s.exchange}:{s.symbol}:{s.timeframe}:{s.latest_pivot_ts}"
                    if key not in state:
                        await send_telegram(session, make_text(s))
                        state[key] = 1
                        save_state(state)
                logging.info("%s %s: %s signals", exchange, tf, len(signals))

if __name__ == "__main__":
    asyncio.run(main())
