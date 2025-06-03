import requests
import os
import threading
import time
from typing import Optional

from logger import logger
class TelegramNotifier:
    """Handle Telegram notifications and commands"""
    
    def __init__(self, bot_instance=None, portfolio=None):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.is_backtest = os.getenv("TRADING_MODE", "backtest").lower() == "backtest"
        self.bot_instance = bot_instance  # Reference to the trading bot
        self.portfolio = portfolio  # Reference to the portfolio for trading info
        
        # Command polling state
        self.last_update_id = 0
        self.command_thread = None
        self.is_polling = False
        
        if not self.bot_token or not self.chat_id or self.is_backtest:
            logger.warning("Telegram credentials not found or in backtest mode")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram bot initialized successfully")
    
    def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            logger.info(f"Telegram disabled - Message: {message}")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id, 
            "text": message,
            "parse_mode": "HTML"  # Enable HTML formatting
        }
        
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def get_updates(self, offset: Optional[int] = None) -> Optional[dict]:
        """Get updates from Telegram Bot API"""
        if not self.enabled:
            return None
            
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {
            "timeout": 10,
            "allowed_updates": ["message"]
        }
        
        if offset:
            params["offset"] = offset
            
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get Telegram updates: {e}")
            return None
    
    def process_command(self, message_text: str, chat_id: str) -> str:
        """Process incoming Telegram commands"""
        if not self.bot_instance:
            return "❌ Bot instance not available"
        
        command = message_text.lower().strip()
        
        try:
            if command == "/status":
                return self.get_status_message()
            
            elif command == "/portfolio":
                return self.get_portfolio_message()
            
            elif command == "/close":
                return self.close_position_command()
            
            elif command == "/stop":
                return self.stop_bot_command()
            
            elif command == "/metrics":
                return self.get_metrics_message()
            
            elif command == "/regime":
                return self.get_regime_message()
            
            elif command == "/help":
                return self.get_help_message()
            
            elif command.startswith("/"):
                return f"❓ Unknown command: {command}\nSend /help for available commands"
            
            else:
                # Ignore non-command messages
                return None
                
        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}")
            return f"❌ Error processing command: {str(e)}"
    
    def get_status_message(self) -> str:
        """Get current bot status"""
        try:
            portfolio = self.portfolio
            is_running = getattr(self.bot_instance, 'is_running', False)
            
            # Get current price (if available)
            current_price = "N/A"
            if hasattr(self.bot_instance, 'last_price'):
                current_price = f"{self.bot_instance.last_price:.4f}"
            
            status_emoji = "🟢" if is_running else "🔴"
            position_emoji = "📈" if portfolio.position > 0 else "📉" if portfolio.position < 0 else "⚪"
            
            message = f"""<b>{status_emoji} BOT STATUS</b>
            
<b>State:</b> {'Running' if is_running else 'Stopped'}
<b>Symbol:</b> {self.bot_instance.config.SYMBOL}
<b>Price:</b> {current_price}
<b>Position:</b> {position_emoji} {portfolio.position:.4f}
<b>Cash:</b> ${portfolio.cash:.2f}

<i>Use /portfolio for detailed info</i>"""
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return f"❌ Error getting status: {str(e)}"
    
    def get_portfolio_message(self) -> str:
        """Get detailed portfolio information"""
        try:
            portfolio = self.portfolio
            
            # Try to get current price from recent data
            current_price = self.bot_instance.get_current_price()
            
            if current_price is None:
                return "❌ Unable to fetch current price for portfolio calculation"
            
            position_value = portfolio.get_position_value(current_price)
            unrealized_pnl = portfolio.get_unrealized_pnl(current_price)
            total_value = portfolio.get_total_value(current_price)
            return_pct = portfolio.get_total_return_pct(current_price)
            
            position_info = ""
            if portfolio.position != 0:
                position_type = "LONG" if portfolio.position > 0 else "SHORT"
                entry_price = portfolio.entry_price or 0
                
                position_info = f"""
<b>🎯 POSITION DETAILS</b>
<b>Type:</b> {position_type}
<b>Size:</b> {abs(portfolio.position):.4f}
<b>Entry:</b> ${entry_price:.4f}
<b>Current:</b> ${current_price:.4f}
<b>Value:</b> ${position_value:.2f}
<b>UPNL:</b> ${unrealized_pnl:.2f}"""
                
                if portfolio.stop_loss_price:
                    position_info += f"\n<b>Stop Loss:</b> ${portfolio.stop_loss_price:.4f}"
                if portfolio.take_profit_price:
                    position_info += f"\n<b>Take Profit:</b> ${portfolio.take_profit_price:.4f}"
            
            message = f"""<b>💼 PORTFOLIO STATUS</b>

<b>💰 ACCOUNT</b>
<b>Cash:</b> ${portfolio.cash:.2f}
<b>Position Value:</b> ${position_value:.2f}
<b>Total Value:</b> ${total_value:.2f}
<b>Total Return:</b> {return_pct:.2f}%
<b>Initial Capital:</b> ${portfolio.initial_cash:.2f}
{position_info}

<b>📊 RISK METRICS</b>
<b>Daily PNL:</b> ${portfolio.daily_pnl:.2f}
<b>Consecutive Losses:</b> {portfolio.consecutive_losses}
<b>In Drawdown:</b> {'Yes' if portfolio.in_drawdown else 'No'}"""
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting portfolio info: {e}")
            return f"❌ Error getting portfolio info: {str(e)}"
    
    def close_position_command(self) -> str:
        """Close current position"""
        try:
            portfolio = self.portfolio
            
            if portfolio.position == 0:
                return "ℹ️ No open position to close"
            
            # Get current price
            df = self.bot_instance.get_historical_ohlcv(
                self.bot_instance.config.SYMBOL,
                self.bot_instance.config.INTERVAL,
                5
            )
            
            if df is None or len(df) == 0:
                return "❌ Unable to fetch current price for position closure"
            
            current_price = self.bot_instance.get_current_price()
            
            # Close the position
            success = self.bot_instance.close_position(current_price, "📱 MANUAL CLOSE (Telegram)")
            
            if success:
                return f"✅ Position closed successfully at ${current_price:.4f}"
            else:
                return "❌ Failed to close position"
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return f"❌ Error closing position: {str(e)}"
    
    def stop_bot_command(self) -> str:
        """Stop the trading bot"""
        try:
            if hasattr(self.bot_instance, 'is_running'):
                self.bot_instance.is_running = False
                
            # Stop command polling as well
            self.stop_command_polling()
            
            return "🛑 Bot stop signal sent. The bot will shutdown gracefully."
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return f"❌ Error stopping bot: {str(e)}"
    
    def get_metrics_message(self) -> str:
        """Get trading performance metrics"""
        try:
            metrics = self.bot_instance.calculate_performance_metrics()
            
            if metrics.get('total_trades', 0) == 0:
                return "📊 No completed trades yet"
            
            message = f"""<b>📊 PERFORMANCE METRICS</b>

<b>TRADE STATISTICS</b>
<b>Total Trades:</b> {metrics.get('total_trades', 0)}
<b>Winning Trades:</b> {metrics.get('winning_trades', 0)}
<b>Losing Trades:</b> {metrics.get('losing_trades', 0)}
<b>Win Rate:</b> {metrics.get('win_rate', 0):.2%}

<b>PROFITABILITY</b>
<b>Profit Factor:</b> {metrics.get('profit_factor', 0):.2f}
<b>Average Win:</b> ${metrics.get('avg_win', 0):.2f}
<b>Average Loss:</b> ${metrics.get('avg_loss', 0):.2f}
<b>Largest Win:</b> ${metrics.get('largest_win', 0):.2f}
<b>Largest Loss:</b> ${metrics.get('largest_loss', 0):.2f}"""
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return f"❌ Error getting metrics: {str(e)}"
    
    def get_regime_message(self) -> str:
        """Get current market regime information"""
        try:
            regime_detector = self.bot_instance.market_regime
            current_regime = regime_detector.current_regime
            
            # Get recent regime history
            recent_history = regime_detector.regime_history[-5:] if regime_detector.regime_history else []
            
            message = f"""<b>📈 MARKET REGIME</b>

<b>Current Regime:</b> {current_regime.upper()}

<b>Recent History:</b>"""
            
            for entry in recent_history:
                timestamp = entry['timestamp'].strftime('%H:%M')
                regime = entry['regime'].upper()
                adx = entry.get('adx', 0)
                message += f"\n{timestamp}: {regime} (ADX: {adx:.1f})"
            
            if not recent_history:
                message += "\nNo regime history available"
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting regime info: {e}")
            return f"❌ Error getting regime info: {str(e)}"
    
    def get_help_message(self) -> str:
        """Get help message with available commands"""
        return """<b>🤖 TELEGRAM BOT COMMANDS</b>

<b>/status</b> - Bot status and basic info
<b>/portfolio</b> - Detailed portfolio information
<b>/close</b> - Close current position
<b>/metrics</b> - Trading performance metrics
<b>/regime</b> - Current market regime
<b>/stop</b> - Stop the trading bot
<b>/help</b> - Show this help message

<i>💡 Commands are case-insensitive</i>
<i>⚠️ Use /stop with caution - it shuts down the bot</i>"""
    
    def start_command_polling(self):
        """Start polling for Telegram commands in a separate thread"""
        if not self.enabled or self.is_polling:
            return
        
        self.is_polling = True
        self.command_thread = threading.Thread(target=self._command_polling_loop, daemon=True)
        self.command_thread.start()
        logger.info("Telegram command polling started")
    
    def stop_command_polling(self):
        """Stop command polling"""
        self.is_polling = False
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=5)
        logger.info("Telegram command polling stopped")
    
    def _command_polling_loop(self):
        """Main command polling loop"""
        while self.is_polling:
            try:
                updates = self.get_updates(offset=self.last_update_id + 1)
                
                if updates and updates.get('ok'):
                    for update in updates.get('result', []):
                        self.last_update_id = update['update_id']
                        
                        # Process only text messages
                        if 'message' in update and 'text' in update['message']:
                            message = update['message']
                            text = message['text']
                            chat_id = str(message['chat']['id'])
                            
                            # Only respond to messages from authorized chat
                            if chat_id == self.chat_id:
                                response = self.process_command(text, chat_id)
                                if response:  # Only send response if command was recognized
                                    self.send_message(response)
                            else:
                                logger.warning(f"Unauthorized chat ID attempted command: {chat_id}")
                
                time.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in command polling loop: {e}")
                time.sleep(5)  # Wait longer on error