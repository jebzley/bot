from config import TradingConfig
from bot import TradingBot
import os
import dotenv

from logger import logger

dotenv.load_dotenv()

def main():
    config = TradingConfig()
    bot = TradingBot(config)
    
    try:
        mode = os.getenv("TRADING_MODE", "backtest").lower()
        
        if mode == 'backtest':
            bot.backtest()
        else:
            bot.paper_trader()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        bot.stop()

if __name__ == "__main__":
    main()