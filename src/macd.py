import pandas as pd
from ta.trend import MACD
import time
import ccxt
import datetime
import math
import requests

# --- CONFIG ---
SYMBOL = "FARTCOIN/USDC:USDC"
INTERVAL = "1m"
TRADE_SIZE_PCT = 0.1
INITIAL_CASH = 1000
TRAILING_STOP_PCT = 0.05
TAKE_PROFIT_PCT = 0.1

TELEGRAM_BOT_TOKEN = "7656305504:AAH5WHM4G2rwcWll1D2PRe5KJpu_YUaYPwU"
TELEGRAM_CHAT_ID = 1517594294

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram send error: {e}")

# --- INIT CCXT EXCHANGE FOR OHLCV ---
def init_ccxt_exchange():
    return ccxt.hyperliquid({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })

# --- Fetch Historical Candles ---
def get_historical_ohlcv(symbol, interval, limit):
    exchange = init_ccxt_exchange()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.tail(limit)

# --- Apply MACD ---
def apply_macd(df):
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    df['hist'] = macd.macd_diff()
    return df

# --- Trading Signal ---
def get_signal(df):
    if df.iloc[-1]['macd'] > df.iloc[-1]['signal'] and df.iloc[-2]['macd'] <= df.iloc[-2]['signal']:
        return 'buy'
    elif df.iloc[-1]['macd'] < df.iloc[-1]['signal'] and df.iloc[-2]['macd'] >= df.iloc[-2]['signal']:
        return 'sell'
    return None

# --- Portfolio State ---
portfolio = {
    'position': 0,
    'cash': INITIAL_CASH,
    'entry_price': None,
    'highest_price': None
}

# --- Simulated Trade Execution ---
def simulate_trade(signal, price):
    global portfolio

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pos_value = portfolio['position'] * price
    upnl = (price - portfolio['entry_price']) * portfolio['position'] if portfolio['entry_price'] else 0
    total_value = portfolio['cash'] + pos_value

    print(f"{now} | Price: {price:.4f} | Cash: {portfolio['cash']:.2f} | Pos: {pos_value:.2f} | Total: {total_value:.2f} | UPNL: {upnl:.2f}")

    # --- Trailing Stop ---
    if portfolio['position'] > 0:
        portfolio['highest_price'] = max(portfolio['highest_price'], price) if portfolio['highest_price'] else price
        
        if(price > portfolio['entry_price'] * (1 + TAKE_PROFIT_PCT)):
            portfolio['cash'] += pos_value
            portfolio['position'] = 0
            portfolio['entry_price'] = None
            portfolio['highest_price'] = None
            pnl = upnl
            msg = f"💰 TAKE PROFIT @ {price:.4f} | PNL {pnl:.2f}"
            print(msg)
            send_telegram_message(msg)
            send_telegram_message(f"Total: {portfolio['cash']}")
        else:
            drop_pct = (portfolio['highest_price'] - price) / portfolio['highest_price']
            if drop_pct >= TRAILING_STOP_PCT:
                pnl = upnl
                portfolio['cash'] += pos_value
                msg = f"🛑 STOPPED @ {price:.2f} | PNL {pnl:.2f}"
                print(msg)
                send_telegram_message(msg)
                send_telegram_message(f"Total: {portfolio['cash']}")

                portfolio['position'] = 0
                portfolio['entry_price'] = None
                portfolio['highest_price'] = None
                return

    # --- Entry Logic ---
    if signal == 'buy' and portfolio['position'] == 0:
        qty = math.floor((portfolio['cash'] * TRADE_SIZE_PCT) / price)
        cost = price * qty
        portfolio['position'] = qty
        portfolio['cash'] -= cost
        portfolio['entry_price'] = price
        portfolio['highest_price'] = price
        msg = f"📈 LONG OPENED @ {price:.4f}"
        print(msg)
        send_telegram_message(msg)

# --- Backtest ---
def backtest():
    df = get_historical_ohlcv(SYMBOL, INTERVAL, limit=10800)
    df = apply_macd(df)

    for i in range(35, len(df)):
        sub_df = df.iloc[:i]
        signal = get_signal(sub_df)
        simulate_trade(signal, sub_df.iloc[-1]['close'])

    final_price = df.iloc[-1]['close']
    final_value = portfolio['cash'] + (portfolio['position'] * final_price)
    print(f"✅ BACKTEST COMPLETE | Final Portfolio Value: {final_value:.2f}")

# --- Live Paper Trader ---
def paper_trader():
    while True:
        df = get_historical_ohlcv(SYMBOL, INTERVAL, limit=200)
        df = apply_macd(df)
        signal = get_signal(df)
        simulate_trade(signal, df.iloc[-1]['close'])
        time.sleep(20)

# --- RUN ---
if __name__ == "__main__":
    # backtest()
    paper_trader()
