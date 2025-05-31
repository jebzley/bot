import os

from datetime import datetime
from typing import Optional, Literal
from ccxt import hyperliquid
from config import TradingConfig
from logger import logger
from telegram import TelegramNotifier

class Portfolio:    
    def __init__(self, exchange: Optional[hyperliquid]):
        self.is_live = os.getenv("TRADING_MODE", "backtest").lower() == "live"
        self.exchange = exchange
        self.config = TradingConfig()

        self.position = 0.0
        self.cash = 0.0
        self.initial_cash = 0.0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None  
        self.highest_price: Optional[float] = None
        self.lowest_price: float = None
        self.position_type: Literal['long', 'short', None] = None 
        self.stop_loss_price: Optional[float] = None  
        self.take_profit_price: Optional[float] = None
        self.unrealized_pnl = 0.0
        self.position_value = 0.0
        
        # Drawdown tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.consecutive_losses = 0
        self.last_trade_dt: datetime = None
        self.in_drawdown = False
    
    def initialize_portfolio(self, initial_cash = 1000.0):
        if not self.is_live:
            self.cash = initial_cash
            self.initial_cash = initial_cash
            self.notifier = TelegramNotifier()
        else:
            try:
                
                self.sync_with_exchange()
                
                balance = self.exchange.fetch_balance()
                usdc_balance = balance['USDC']['free']
                self.initial_cash = usdc_balance
                logger.info(f"Initialized portfolio with USDC balance: ${usdc_balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to initialize portfolio: {e}")
                raise
            
    def check_account_balance(self):
        try:
            if not self.is_live:
                return
                
            balance = self.exchange.fetch_balance()
            usdc_balance = balance.get('USDC', {}).get('free', 0)
            
            logger.info(f"Account USDC Balance: ${usdc_balance:.2f}")
            self.notifier.send_message(f"💰 Account Balance: ${usdc_balance:.2f}")
            
            if abs(self.cash - usdc_balance) > 1:
                if self.position == 0:  
                    self.cash = usdc_balance
                    
        except Exception as e:
            logger.error(f"Failed to check account balance: {e}")

    def sync_with_exchange(self):
        try:
            if not self.is_live:
                return
            
            balance = self.exchange.fetch_balance()
            usdc_balance = balance['USDC']['free']
            self.cash = usdc_balance
                
            # Fetch positions from exchange
            position = self.exchange.fetch_position(self.config.SYMBOL)
            
            if position:
                exchange_size = float(position['contracts'])
                exchange_side = position['side']
                
                if exchange_size > 0 and exchange_side:
                    self.position = exchange_size if exchange_side == 'long' else -exchange_size
                    self.entry_price = float(position['averagePrice'])
                    self.position_type = exchange_side
                    self.entry_time = datetime.fromisoformat(position['datetime'])
                    self.unrealized_pnl = position['unrealizedPnl']
                    self.position_value = position['notional']
                    
                    logger.info(f"Portfolio synced: {self.position} @ {self.entry_price}")
                else:
                    # No position on exchange
                    if self.position != 0:
                        logger.warning("Local position exists but exchange shows no position - resetting")
                        self.position = 0
                        self.entry_price = None
                        self.position_type = None
                        self.entry_time = None
                        self.unrealized_pnl = 0.0
                        self.position_value = 0.0
            else:
                # No positions returned
                if self.position != 0:
                    logger.warning("No positions on exchange but local position exists - resetting")
                    self.position = 0
                    self.entry_price = None
                    self.position_type = None
                    self.entry_time = None
                    self.unrealized_pnl = 0.0
                    self.position_value = 0.0
                    
        except Exception as e:
            logger.error(f"Failed to sync position: {e}")
            # self.notifier.send_message(f"⚠️ Failed to sync position: {str(e)}")
 
    def get_position_value(self, current_price: float) -> float:
        if self.is_live:
            self.sync_with_exchange()
            return self.position_value
        
        if self.position == 0:
            return 0.0
        if self.position_type == 'long':
            return self.position * current_price
        elif self.position_type == 'short':
            return (abs(self.position) * self.entry_price) + self.get_unrealized_pnl(current_price)
        return 0.0
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.is_live:
            self.sync_with_exchange()
            return self.unrealized_pnl
        
        if not self.entry_price or self.position == 0:
            return 0.0
        
        if self.position_type == 'long':
            upnl = (current_price - self.entry_price) * self.position
            self.unrealized_pnl = upnl
            return upnl
        elif self.position_type == 'short':
            upnl = (self.entry_price - current_price) * abs(self.position)
            self.unrealized_pnl = upnl
            return upnl
        return 0.0
    
    def get_total_value(self, current_price: float) -> float:
        position_value = self.get_position_value(current_price)
        return self.cash + position_value
    
    def get_total_return_pct(self, current_price: float) -> float:
        current_value = self.get_total_value(current_price)
        return ((current_value - self.initial_cash) / self.initial_cash) * 100
    
    def is_max_loss_exceeded(self, current_price: float) -> bool:
        return_pct = self.get_total_return_pct(current_price)
        return return_pct <= -self.config.MAX_DAILY_LOSS_PCT * 100