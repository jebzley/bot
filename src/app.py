import dotenv
import os

from logger import logger
from portfolio import Portfolio
from exchange import HyperliquidExchange
from bot import TradingBot

dotenv.load_dotenv()

def main():
    exchange = HyperliquidExchange()
    exchange.init_exchange()
    if not exchange.hyperliquid:
        logger.error("Failed to initialize exchange. Exiting application.")
        return
    
    portfolio = Portfolio(exchange.hyperliquid)
    portfolio.initialize_portfolio() 
    if not portfolio.initial_cash:
        logger.error("Failed to initialize portfolio. Exiting application.")
        return
    bot = TradingBot(exchange, portfolio)

    try:
        mode = os.getenv("TRADING_MODE", "backtest").lower()
        
        if mode == 'backtest':
            bot.backtest()
        else:
            bot.start_trading()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        bot.stop()  

if __name__ == "__main__":
    main()