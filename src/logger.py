import logging
import json
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeLogger:
    """Logger specifically for trade data in JSON format"""
    
    def __init__(self):
        self.log_dir = "trade_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.trade_log_file = os.path.join(
            self.log_dir,
            f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        logger.info(f"Trade logger initialized: {self.trade_log_file}")
    
    def log_trade(self, trade_data: dict):
        """Log trade to JSON file for analysis"""
        try:
            with open(self.trade_log_file, 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    **trade_data
                }, f)
                f.write('\n')
            logger.debug(f"Trade logged: {trade_data.get('order_id', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def get_trades(self) -> list:
        """Read all trades from log file"""
        trades = []
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            trades.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read trades: {e}")
        return trades