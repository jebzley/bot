import dotenv

from logger import logger
from portfolio import Portfolio
from exchange import HyperliquidExchange


dotenv.load_dotenv()

def main():
    exchange = HyperliquidExchange()
    exchange.init_exchange()
    if not exchange.hyperliquid:
        logger.error("Failed to initialize exchange. Exiting application.")
        return
    
    portfolio = Portfolio(exchange.hyperliquid)
    portfolio.initialize_portfolio()   

if __name__ == "__main__":
    main()