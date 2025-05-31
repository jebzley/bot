import os
from ccxt import hyperliquid

from typing import Optional

from logger import logger

class HyperliquidExchange:
    def __init__(self):
        self.hyperliquid: Optional[hyperliquid] = None
        self.is_live = os.getenv("TRADING_MODE", "backtest").lower() == "live"

    def init_exchange(self) -> None:
        try:
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap', 
                },
                'timeout': 30000,
            }
            
            if self.is_live:
                private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
                wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
                
                if not private_key or not wallet_address:
                    raise ValueError("Missing Hyperliquid credentials for live trading")
                
                exchange_config['privateKey'] = private_key
                exchange_config['walletAddress'] = wallet_address
                logger.info("Initializing exchange with live credentials")
            else:
                logger.info("Initializing exchange for paper trading")
            
            exchange = hyperliquid(exchange_config)
            
            exchange.load_markets()
            if not exchange.markets:
                raise ValueError("Failed to load markets, check your credentials or network connection")
            
            self.hyperliquid = exchange 

            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
        