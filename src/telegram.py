import requests
import os

from logger import logger

class TelegramNotifier:
    """Handle Telegram notifications"""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.is_backtest = os.getenv("TRADING_MODE", "backtest").lower() == "backtest"
        
        if not self.bot_token or not self.chat_id or self.is_backtest:
            logger.warning("Telegram credentials not found in environment variables")
            self.enabled = False
        else:
            self.enabled = True
    
    def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            logger.info(f"Telegram disabled - Message: {message}")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False