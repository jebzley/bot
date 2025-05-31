import os

from datetime import datetime
from typing import Optional, Literal
from ccxt import hyperliquid
from config import TradingConfig
from logger import logger

class Portfolio:
    """Manage portfolio state and calculations"""
    
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
        
        # Drawdown tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.in_drawdown = False
    
    def initialize_portfolio(self, initial_cash = 1000.0):
        if not self.is_live:
            self.cash = initial_cash
            self.initial_cash = initial_cash
        else:
            try:
                balance = self.exchange.fetch_balance()
                position = self.exchange.fetch_position(self.config.SYMBOL)
                if position:
                    self.position = position['contracts']
                    self.entry_price = position['entryPrice']
                    self.position_type = position['side']
                    self.entry_time = datetime.fromisoformat(position['datetime'])
                    self.position_type = 'long' if self.position > 0 else 'short' if self.position < 0 else None
                    self.stop_loss_price = position['entryPrice'] - (self.config.STOP_LOSS_PCT * position['entryPrice']) if self.position_type == 'long' else position['entryPrice'] + (self.config.STOP_LOSS_PCT * position['entryPrice']) 

                usdc_balance = balance['USDC']['free']
                self.cash = usdc_balance
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
            
            if abs(self.portfolio.cash - usdc_balance) > 1:
                if self.portfolio.position == 0:  
                    self.portfolio.cash = usdc_balance
                    
        except Exception as e:
            logger.error(f"Failed to check account balance: {e}")

    
    def get_position_value(self, current_price: float) -> float:
        if self.position == 0:
            return 0.0
        if self.position_type == 'long':
            return self.position * current_price
        elif self.position_type == 'short':
            return (abs(self.position) * self.entry_price) + self.get_unrealized_pnl(current_price)
        return 0.0
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        if not self.entry_price or self.position == 0:
            return 0.0
        
        if self.position_type == 'long':
            return (current_price - self.entry_price) * self.position
        elif self.position_type == 'short':
            return (self.entry_price - current_price) * abs(self.position)
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