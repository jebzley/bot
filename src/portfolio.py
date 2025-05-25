class Portfolio:
    """Manage portfolio state and calculations"""
    
    def __init__(self, initial_cash: float):
        self.position = 0.0
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.entry_price = None
        self.entry_time = None  # Added for time-based stops
        self.highest_price = None
        self.lowest_price = None
        self.position_type = None  # 'long', 'short', or None
        self.stop_loss_price = None  # Dynamic stop loss
        self.take_profit_price = None  # Dynamic take profit
        
        # Drawdown tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.in_drawdown = False
        
    def get_position_value(self, current_price: float) -> float:
        """Calculate current position value"""
        if self.position == 0:
            return 0.0
        if self.position_type == 'long':
            return self.position * current_price
        elif self.position_type == 'short':
            return (abs(self.position) * self.entry_price) + self.get_unrealized_pnl(current_price)
        return 0.0
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if not self.entry_price or self.position == 0:
            return 0.0
        
        if self.position_type == 'long':
            return (current_price - self.entry_price) * self.position
        elif self.position_type == 'short':
            return (self.entry_price - current_price) * abs(self.position)
        return 0.0
    
    def get_total_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = self.get_position_value(current_price)
        return self.cash + position_value
    
    def get_total_return_pct(self, current_price: float) -> float:
        """Calculate total return percentage"""
        current_value = self.get_total_value(current_price)
        return ((current_value - self.initial_cash) / self.initial_cash) * 100
    
    def is_max_loss_exceeded(self, current_price: float, max_loss_pct: float) -> bool:
        """Check if maximum loss threshold is exceeded"""
        return_pct = self.get_total_return_pct(current_price)
        return return_pct <= -max_loss_pct * 100