import dotenv
import os
import argparse

from logger import logger
from portfolio import Portfolio
from exchange import HyperliquidExchange
from bot import TradingBot

dotenv.load_dotenv()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--csv', type=str, help='Path to CSV file for backtesting')
    parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'paper'], 
                       help='Override trading mode from environment')
    args = parser.parse_args()
    
    # Override mode if specified in command line
    if args.mode:
        os.environ['TRADING_MODE'] = args.mode
    
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
            # Pass CSV file if provided
            bot.backtest(csv_filepath=args.csv)
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