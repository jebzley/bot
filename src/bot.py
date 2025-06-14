import pandas as pd
import time
import os
import datetime
import math
from typing import Optional, Dict

from config import TradingConfig
from portfolio import Portfolio
from telegram import TelegramNotifier
from technicals import TechnicalAnalysis
from exchange import HyperliquidExchange
from logger import logger, TradeLogger

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, exchange: HyperliquidExchange, portfolio: Portfolio):
        self.config = TradingConfig()
        self.portfolio = portfolio
        self.ta = TechnicalAnalysis(self.config, exchange)
        self.exchange: HyperliquidExchange = exchange
        self.is_running = False
        self.is_live = os.getenv("TRADING_MODE", "backtest").lower() == "live"
        self.trade_history = []
        self.last_price = None 
        self.trade_logger = TradeLogger()
        self.notifier = TelegramNotifier(self, portfolio)  
 
        
        
    def calculate_position_size(self, price: float, signal: str, atr: float, signal_score: int = 0) -> float:
        """Calculate position size using hybrid approach based on config"""
        try:
            base_size = 0.0
            
            # Simple sizing (original method)
            if self.config.POSITION_SIZING_MODE == "simple":
                return self.calculate_simple_position_size(price, signal)
            
            # ATR-based sizing only
            elif self.config.POSITION_SIZING_MODE == "atr" and atr > 0:
                risk_per_trade = self.portfolio.cash * 0.015
                stop_distance = self.config.ATR_MULTIPLIER * atr
                base_size = risk_per_trade / stop_distance
            
            # Signal strength sizing only
            elif self.config.POSITION_SIZING_MODE == "signal":
                base_allocation = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                multiplier = self.get_signal_strength_multiplier(signal_score)
                base_size = (base_allocation * multiplier) / price
            
            # Hybrid sizing (best of both)
            elif self.config.POSITION_SIZING_MODE == "hybrid":
                # Start with ATR-based calculation
                if atr > 0:
                    risk_per_trade = self.portfolio.cash * 0.015
                    stop_distance = self.config.ATR_MULTIPLIER * atr
                    atr_size = risk_per_trade / stop_distance
                else:
                    # Fallback to simple if no ATR
                    atr_size = (self.portfolio.cash * self.config.TRADE_SIZE_PCT) / price
                
                # Apply signal strength multiplier
                multiplier = self.get_signal_strength_multiplier(signal_score)
                base_size = atr_size * multiplier
            
            else:
                # Default to simple
                return self.calculate_simple_position_size(price, signal)
            
            # Apply maximum position size limit (30% of capital)
            max_position_value = self.portfolio.cash * 0.3
            max_position_size = max_position_value / price
            final_size = min(base_size, max_position_size)
            
            # Apply drawdown reduction if needed
            if self.portfolio.in_drawdown:
                final_size *= self.config.POSITION_REDUCTION_FACTOR
                logger.info(f"Position size reduced by {(1-self.config.POSITION_REDUCTION_FACTOR)*100:.0f}% due to drawdown")
            
            # Round and return with correct sign
            if signal == 'buy':
                return math.floor(final_size * 100) / 100
            else:  # sell
                return -math.floor(final_size * 100) / 100
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.calculate_simple_position_size(price, signal)
    
    def get_signal_strength_multiplier(self, score: int) -> float:
        """Get position size multiplier based on signal strength"""
        abs_score = abs(score)
        
        if abs_score >= self.config.SIGNAL_VERY_STRONG:
            return self.config.MULTIPLIER_VERY_STRONG
        elif abs_score >= self.config.SIGNAL_STRONG:
            return self.config.MULTIPLIER_STRONG
        else:
            return self.config.MULTIPLIER_NORMAL
    
    def calculate_simple_position_size(self, price: float, signal: str) -> float:
        """Simple position size calculation as fallback"""
        try:
            if signal == 'buy':
                available_cash = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                max_qty = available_cash / price
                return math.floor(max_qty * 100) / 100
            elif signal == 'sell':
                available_cash = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                max_qty = available_cash / price
                return -math.floor(max_qty * 100) / 100
            return 0.0
        except Exception as e:
            logger.error(f"Error in simple position size calculation: {e}")
            return 0.0
    
    def check_dynamic_exits(self, price: float, df: pd.DataFrame) -> tuple[bool, str]:
        if self.portfolio.position == 0:
            return False, ""
             
        # 2. Time-based profit taking
        if self.portfolio.entry_time:
            time_in_position = datetime.datetime.now() - self.portfolio.entry_time
            hours_in_position = time_in_position.total_seconds() / 3600
            
            current_pnl_pct = self.portfolio.get_unrealized_pnl(price) / (self.portfolio.entry_price * abs(self.portfolio.position))
            
            # Scale out if profitable after certain time
            if hours_in_position > 2 and current_pnl_pct > 0.01:  # 1% profit after 2 hours
                return True, "⏰ TIME-BASED PROFIT"
        
        # 3. Support/Resistance exit
        bb_upper = df.iloc[-1]['bb_upper']
        bb_lower = df.iloc[-1]['bb_lower']
        
        if self.portfolio.position_type == 'long' and price > bb_upper:
            if self.portfolio.get_unrealized_pnl(price) > 0:
                return True, "📊 RESISTANCE EXIT"
        elif self.portfolio.position_type == 'short' and price < bb_lower:
            if self.portfolio.get_unrealized_pnl(price) > 0:
                return True, "📊 SUPPORT EXIT"
        
        return False, ""

    def check_exit_conditions(self, price: float, signal: Optional[str], regime: str, df: pd.DataFrame) -> bool:
        """Check various exit conditions including time-based stops"""
        if self.portfolio.position == 0:
            return False
        
        try:
            # Check maximum loss
            if self.portfolio.is_max_loss_exceeded(price):
                self.close_position(price, "🚨 MAX LOSS STOP")
                return True
            
            # Time-based stop loss - only for positions open too long
            if self.portfolio.entry_time:
                time_in_position = datetime.datetime.now() - self.portfolio.entry_time
                if time_in_position > datetime.timedelta(hours=self.config.MAX_POSITION_DURATION_HOURS):
                    if self.portfolio.get_unrealized_pnl(price) < -self.portfolio.cash * 0.01:  # Only if losing more than 1%
                        self.close_position(price, "⏰ TIME STOP")
                        return True
            
            # Get current ATR for dynamic stops
            current_atr = df.iloc[-1].get('atr', None)
            atr_5_avg = df['atr'].tail(5).mean()

            if current_atr > atr_5_avg * 1.5:  # High volatility
                # Wider stops in volatile conditions
                trailing_multiplier = 2
            else:
                trailing_multiplier = 1.2
            
            # Long position management
            if self.portfolio.position_type == 'long':
                # Update highest price
                if not self.portfolio.highest_price or price > self.portfolio.highest_price:
                    self.portfolio.highest_price = price
                
                # Dynamic take profit (ATR-based)
                if self.portfolio.take_profit_price and price >= self.portfolio.take_profit_price:
                    self.close_position(price, "💰 TAKE PROFIT (LONG)")
                    return True
                
                # Dynamic stop loss (ATR-based)
                if self.portfolio.stop_loss_price and price <= self.portfolio.stop_loss_price:
                    self.close_position(price, "🛑 STOP LOSS (LONG)")
                    return True
                
                # Trailing stop
                if self.portfolio.highest_price and current_atr:
                    trailing_stop = self.portfolio.highest_price - (trailing_multiplier * current_atr)
                    if price <= trailing_stop:
                        self.close_position(price, "🛑 TRAILING STOP (LONG)")
                        return True
                
                # Signal-based exits - exit on opposite signal if we're in profit
                if signal == 'sell' and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, f"📉 SIGNAL EXIT (LONG) - {regime.upper()}")
                    return True
                
                # Also exit if momentum is clearly shifting (MACD histogram declining)
                if len(df) >= 3 and self.portfolio.entry_time and self.portfolio.entry_time <= datetime.datetime.now() - datetime.timedelta(minutes=15):
                    macd_hist_current = df.iloc[-1].get('macd_hist', 0)
                    macd_hist_prev = df.iloc[-2].get('macd_hist', 0)
                    macd_hist_prev2 = df.iloc[-3].get('macd_hist', 0)
                    
                    if (macd_hist_current < macd_hist_prev < macd_hist_prev2 and 
                        macd_hist_current < 0 and 
                        self.portfolio.get_unrealized_pnl(price) > 0 ):
                        self.close_position(price, "📊 MOMENTUM EXIT (LONG)")
                        return True
                
                # Exit if price crosses below EMA 50 (similar to Pine script)
                ema_50 = df.iloc[-1].get('ema_50', None)
                if ema_50 and price < ema_50 and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, "📉 EMA50 CROSS EXIT (LONG)")
                    return True
            
            # Short position management
            elif self.portfolio.position_type == 'short':
                # Update lowest price
                if not self.portfolio.lowest_price or price < self.portfolio.lowest_price:
                    self.portfolio.lowest_price = price
                
                # Dynamic take profit (ATR-based)
                if self.portfolio.take_profit_price and price <= self.portfolio.take_profit_price:
                    self.close_position(price, "💰 TAKE PROFIT (SHORT)")
                    return True
                
                # Dynamic stop loss (ATR-based)
                if self.portfolio.stop_loss_price and price >= self.portfolio.stop_loss_price:
                    self.close_position(price, "🛑 STOP LOSS (SHORT)")
                    return True
                
                # Trailing stop
                if self.portfolio.lowest_price and current_atr:

                    trailing_stop = self.portfolio.lowest_price + (trailing_multiplier * current_atr)
                    if price >= trailing_stop:
                        self.close_position(price, "🛑 TRAILING STOP (SHORT)")
                        return True
                
                # Signal-based exits - exit on opposite signal if we're in profit
                if signal == 'buy' and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, f"📈 SIGNAL EXIT (SHORT) - {regime.upper()}")
                    return True
                
                # Also exit if momentum is clearly shifting (MACD histogram rising)
                if len(df) >= 3 and self.portfolio.entry_time <= datetime.datetime.now() - datetime.timedelta(minutes=15):
                    macd_hist_current = df.iloc[-1].get('macd_hist', 0)
                    macd_hist_prev = df.iloc[-2].get('macd_hist', 0)
                    macd_hist_prev2 = df.iloc[-3].get('macd_hist', 0)
                    
                    if (macd_hist_current > macd_hist_prev > macd_hist_prev2 and 
                        macd_hist_current > 0 and 
                        self.portfolio.get_unrealized_pnl(price) > 0):
                        self.close_position(price, "📊 MOMENTUM EXIT (SHORT)")
                        return True
                
                # Exit if price crosses above EMA 50 (similar to Pine script)
                ema_50 = df.iloc[-1].get('ema_50', None)
                if ema_50 and price > ema_50 and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, "📈 EMA50 CROSS EXIT (SHORT)")
                    return True

            # Check for dynamic exits based on volatility and time
            dynamic_exit, reason = self.check_dynamic_exits(price, df)
            if dynamic_exit:
                self.close_position(price, reason)
                return True
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def log_portfolio_status(self, price: float, regime: str):
        """Log current portfolio status"""
        try:
            position_value = self.portfolio.get_position_value(price)
            unrealized_pnl = self.portfolio.get_unrealized_pnl(price)
            total_value = self.portfolio.get_total_value(price)
            return_pct = self.portfolio.get_total_return_pct(price)
            
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode = "LIVE" if self.is_live else "PAPER"
            status = (f"{now} | {mode} | Price: {price:.4f} | Cash: {self.portfolio.cash:.2f} | "
                     f"Pos: {position_value:.2f} | UPNL: {unrealized_pnl:.2f} | "
                     f"Total: {total_value:.2f} | Return: {return_pct:.2f}% | Regime: {regime.upper()}")
            
            logger.info(status)
            
        except Exception as e:
            logger.error(f"Error logging portfolio status: {e}")
    
    def execute_trade(self, signal: str, price: float, regime: str, atr: float, signal_desc: str, signal_score: int) -> bool:
        """Execute trade based on signal with dynamic stops"""
        try:
            if self.check_daily_loss_limit():
                logger.warning("Daily loss limit reached - no new trades")
                return False
            
            if self.is_live and not self.check_trading_cooldown():
                return False
            
            if self.is_live and not self.check_daily_trade_limit():
                return False
            
            qty = self.calculate_position_size(price, signal, atr, signal_score)
            
            if abs(qty) < 0.01:
                logger.warning(f"Position size too small: {qty}")
                return False
            
            cost = abs(qty) * price
            
            if cost > self.portfolio.cash:
                logger.warning(f"Insufficient funds: need {cost}, have {self.portfolio.cash}")
                return False
                        
            actual_price = price
            
            # Execute live order if in live mode
            if self.is_live:
                try:
                    order = self.execute_live_order(
                        symbol=self.config.SYMBOL,
                        side=signal,
                        amount=abs(qty)
                    )
                    
                    if not order:
                        return False
                    
                    # Update with actual fill price - FIX: Handle None values properly
                    avg_price = order.get('average')
                    if avg_price is not None:
                        actual_price = float(avg_price)
                    else:
                        actual_price = float(order.get('price', price))
                    
                    # Get cost from order
                    order_cost = order.get('cost')
                    if order_cost is not None:
                        cost = float(order_cost)
                    
                    # Update trade counters
                    self.portfolio.last_trade_dt = datetime.datetime.now()
                    self.portfolio.daily_trade_count += 1
                    
                except Exception as e:
                    logger.error(f"Live order failed: {e}")
                    return False
            
            # Execute the trade (update portfolio)
            if(self.is_live):
                self.portfolio.sync_with_exchange()
            else:
                self.portfolio.position = qty
                self.portfolio.cash -= cost
                self.portfolio.entry_price = actual_price
                self.portfolio.entry_time = datetime.datetime.now()
                self.portfolio.position_type = 'long' if qty > 0 else 'short'
            
            # Update last trade date
            self.portfolio.last_trade_dt = datetime.datetime.now()
            
            # Get position size info for logging
            multiplier = self.get_signal_strength_multiplier(signal_score)
            size_info = f"Size: {abs(qty):.2f} (x{multiplier:.1f})"
            
            # Set stops - use hybrid approach: tighter of ATR or percentage-based
            if signal == 'buy':
                self.portfolio.highest_price = actual_price
                self.portfolio.lowest_price = None
                
                # Calculate both stop types and use the tighter one
                atr_stop = actual_price - (self.config.ATR_MULTIPLIER * atr)
                pct_stop = actual_price * (1 - self.config.STOP_LOSS_PCT)
                self.portfolio.stop_loss_price = max(atr_stop, pct_stop)
                
                # Take profit uses the original percentage
                self.portfolio.take_profit_price = actual_price * (1 + self.config.TAKE_PROFIT_PCT)
                
                msg = f"📈 {'LIVE' if self.is_live else 'PAPER'} LONG @ {actual_price:.4f} | {size_info} | SL: {self.portfolio.stop_loss_price:.4f} | TP: {self.portfolio.take_profit_price:.4f} | {regime.upper()} | {signal_desc}"
            else:
                self.portfolio.lowest_price = actual_price
                self.portfolio.highest_price = None
                
                # Calculate both stop types and use the tighter one
                atr_stop = actual_price + (self.config.ATR_MULTIPLIER * atr)
                pct_stop = actual_price * (1 + self.config.STOP_LOSS_PCT)
                self.portfolio.stop_loss_price = min(atr_stop, pct_stop)
                
                # Take profit uses the original percentage
                self.portfolio.take_profit_price = actual_price * (1 - self.config.TAKE_PROFIT_PCT)
                
                msg = f"📉 {'LIVE' if self.is_live else 'PAPER'} SHORT @ {actual_price:.4f} | {size_info} | SL: {self.portfolio.stop_loss_price:.4f} | TP: {self.portfolio.take_profit_price:.4f} | {regime.upper()} | {signal_desc}"
            
            logger.info(msg)
            self.notifier.send_message(msg)
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False
    
    def close_position(self, price: float, reason: str) -> bool:
        try:
            if self.portfolio.position == 0:
                return False
            
            position_value = self.portfolio.get_position_value(price)
            pnl = self.portfolio.get_unrealized_pnl(price)
            
            if self.is_live:
                order = self.close_live_position()  
                if not order:
                    logger.error("Failed to close live position")
                    return False
                
                # Get actual price from order
                avg_price = order.get('average')
                if avg_price is not None:
                    actual_price = float(avg_price)
                else:
                    actual_price = float(order.get('price', price))
                
                fee = order.get('fee', {})
                feeCost = fee.get('cost', 0.0) if isinstance(fee, dict) else 0.0
                
                order_cost = order.get('cost')
                if order_cost is not None:
                    pnl = float(order_cost) - feeCost
            else:
                actual_price = price
            
            # Update daily PnL
            self.portfolio.daily_pnl += pnl
            
            # Update consecutive losses/wins
            if pnl < 0:
                self.portfolio.consecutive_losses += 1
            else:
                self.portfolio.consecutive_losses = 0
            
            # Record trade for metrics
            trade_data = {
                'entry_price': self.portfolio.entry_price,
                'exit_price': actual_price,
                'pnl': pnl,
                'position_type': self.portfolio.position_type,
                'entry_time': self.portfolio.entry_time,
                'exit_time': datetime.datetime.now().isoformat(),
                'reason': reason
            }
            self.trade_history.append(trade_data)
            self.portfolio.daily_trades.append(trade_data)
            
            # Log to trade logger if live
            if self.trade_logger:
                self.trade_logger.log_trade(trade_data)
            
            self.portfolio.cash += position_value
            self.portfolio.position = 0
            self.portfolio.entry_price = None
            self.portfolio.entry_time = None
            self.portfolio.highest_price = None
            self.portfolio.lowest_price = None
            self.portfolio.position_type = None
            self.portfolio.stop_loss_price = None
            self.portfolio.take_profit_price = None
            
            self.update_drawdown_status()
            
            mode = "LIVE" if self.is_live else "PAPER"
            msg = f"{reason} @ {actual_price:.4f} | PNL: {pnl:.2f} | Total: {self.portfolio.cash:.2f} | Mode: {mode}"
            logger.info(msg)
            self.notifier.send_message(msg)
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def execute_live_order(self, symbol: str, side: str, amount: float, price: float = None, reduce_only: bool = None) -> Optional[dict]:
        """Execute a live order on the exchange"""
        try:
            # Get current price for slippage calculation
            current_price = price or self.get_current_price()
            if not current_price:
                raise ValueError("Unable to get current price for order execution")
            
            # Round amount to avoid precision issues
            amount = round(abs(amount), 8)  # 8 decimal places should be sufficient
            
            # Safety check for maximum position value
            if abs(amount) * current_price > self.config.MAX_POSITION_VALUE:
                logger.warning(f"Order exceeds max position value of ${self.config.MAX_POSITION_VALUE}")
                amount = self.config.MAX_POSITION_VALUE / current_price
                amount = math.floor(amount * 100) / 100
            
            # Market order with slippage protection
            order_type = 'market'
            params = {}
            
            # Auto-detect reduce-only if not specified
            if reduce_only is None:
                reduce_only = (
                    (side == 'buy' and self.portfolio.position_type == 'short') or 
                    (side == 'sell' and self.portfolio.position_type == 'long')
                )
            
            if reduce_only:
                params['reduceOnly'] = True
                logger.info(f"Using reduce-only order to close position")
            
            # Calculate slippage price (use configured slippage)
            slippage = self.config.SLIPPAGE_TOLERANCE
            if side == 'buy':
                slippage_price = current_price * (1 + slippage)
            else:  # sell
                slippage_price = current_price * (1 - slippage)
            
            logger.info(f"Executing {order_type} {side} order for {amount} {symbol} with max price {slippage_price:.4f} ({slippage*100:.1f}% slippage)")
            
            # Create order with slippage price
            order = self.exchange.hyperliquid.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=slippage_price,  # Required for Hyperliquid market orders
                params=params
            )
            
            # Log the order
            logger.info(f"Order executed: {order['id']} - {side} {amount} @ {order.get('price', 'market')}")
            
            # Log to trade logger
            if self.trade_logger:
                self.trade_logger.log_trade({
                    'order_id': order['id'],
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': order.get('price', 'N/A'),
                    'type': order_type,
                    'status': order.get('status', 'unknown'),
                    'reduce_only': reduce_only
                })
            
            self.portfolio.sync_with_exchange()
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            self.notifier.send_message(f"❌ ORDER FAILED: {str(e)}")
            raise

    def close_live_position(self):  
        try:
            if self.portfolio.position == 0:
                return None
            
            self.portfolio.sync_with_exchange()
            
            if self.portfolio.position == 0:
                logger.info("No position to close after sync")
                return None
            
            current_price = self.get_current_price()
            if not current_price:
                logger.error("Unable to get current price for position closure")
                return None
            
            # Get the exact position from exchange
            try:
                exchange_position = self.exchange.hyperliquid.fetch_position(self.config.SYMBOL)
                if exchange_position and exchange_position.get('contracts'):
                    exact_amount = abs(float(exchange_position['contracts']))
                    logger.info(f"Using exact position from exchange: {exact_amount}")
                else:
                    # Fallback to portfolio position
                    exact_amount = abs(self.portfolio.position)
                    logger.warning(f"No exchange position found, using portfolio amount: {exact_amount}")
            except Exception as e:
                logger.error(f"Failed to fetch position from exchange: {e}")
                exact_amount = abs(self.portfolio.position)
            
            # Determine side based on position type
            if self.portfolio.position_type == 'long':
                side = 'sell'
            else:  # short
                side = 'buy'
            
            # Execute the closing order with reduce-only flag
            order = self.execute_live_order(
                symbol=self.config.SYMBOL,
                side=side,
                amount=exact_amount,
                price=current_price
            )
            
            if order:
                # Wait a moment for the order to settle
                time.sleep(1)
                
                # Verify position is fully closed
                remaining_position = self.exchange.hyperliquid.fetch_position(self.config.SYMBOL)
                if remaining_position and remaining_position.get('contracts'):
                    remaining_amount = abs(float(remaining_position['contracts']))
                    if remaining_amount > 0.00001:  # Tiny threshold
                        logger.warning(f"Residual position detected: {remaining_amount}")
                        # Attempt to close the residual
                        self.execute_live_order(
                            symbol=self.config.SYMBOL,
                            side=side,
                            amount=remaining_amount,
                            price=current_price
                        )
            
            return order
        
        except Exception as e:
            logger.error(f"Failed to close live position: {e}")
            return None
    
    def handle_trade(self, signal: Optional[str], price: float, regime: str, df: pd.DataFrame, signal_desc: str, signal_score: int):
        """Main trading logic simulation"""
        try:
            self.log_portfolio_status(price, regime)
            
            # Check exit conditions first
            if self.check_exit_conditions(price, signal, regime, df):
                return
            
            # Entry logic - only enter if no position
            if signal and self.portfolio.position == 0:
                current_atr = df.iloc[-1].get('atr', None)
                if current_atr and signal in ['buy', 'sell']:
                    # Only take trades in direction of trend
                    ema_20 = df.iloc[-1].get('ema_20', price)
                    ema_50 = df.iloc[-1].get('ema_50', price)
                    
                    # For weaker signals, ensure price is on the right side of both EMAs
                    if abs(signal_score) < 39:
                        if signal == 'buy':
                            if price < ema_20 or price < ema_50:
                                return
                        elif signal == 'sell':
                            if price > ema_20 or price > ema_50:
                                return
                    
                    self.execute_trade(signal, price, regime, current_atr, signal_desc, signal_score)
                    
        except Exception as e:
            logger.error(f"Error in trade simulation: {e}")

    def update_drawdown_status(self):
        """Update drawdown status based on recent performance"""
        try:
            # Reset daily stats if new day
            current_date = datetime.datetime.now()
            if self.portfolio.last_trade_dt is not None and current_date != self.portfolio.last_trade_dt.date():
                self.portfolio.daily_pnl = 0.0
                self.portfolio.daily_trades = []
                self.portfolio.last_trade_dt = current_date
            
            # Check daily loss limit
            if self.portfolio.daily_pnl < -self.portfolio.initial_cash * self.config.MAX_DAILY_LOSS_PCT:
                self.portfolio.in_drawdown = True
                logger.warning(f"Daily loss limit reached: ${self.portfolio.daily_pnl:.2f}")
            
            # Check consecutive losses
            if self.portfolio.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
                self.portfolio.in_drawdown = True
                logger.warning(f"Consecutive losses limit reached: {self.portfolio.consecutive_losses}")
            
            # Reset drawdown status if we have a winning streak
            if self.portfolio.consecutive_losses == 0 and self.portfolio.daily_pnl > 0:
                if self.portfolio.in_drawdown:
                    logger.info("Drawdown status cleared - back to normal position sizing")
                self.portfolio.in_drawdown = False
                
        except Exception as e:
            logger.error(f"Error updating drawdown status: {e}")
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        if self.portfolio.daily_pnl < -self.portfolio.initial_cash * self.config.MAX_DAILY_LOSS_PCT:
            return True
        return False
       
    def get_current_price(self) -> Optional[float]:
        try:
            df = self.ta.get_historical_ohlcv(
                self.config.SYMBOL,
                self.config.INTERVAL,
                5
            )
            if df is not None and len(df) > 0:
                price = df.iloc[-1]['close']
                self.last_price = price  # Store for status updates
                return price
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0
                }
            
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
            
            total_trades = len(self.trade_history)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = gross_profit / win_count if win_count > 0 else 0
            avg_loss = gross_loss / loss_count if loss_count > 0 else 0
            
            largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def backtest(self):
        """Run backtest on historical data"""
        logger.info("Starting backtest...")
        
        try:
            df = self.ta.get_historical_ohlcv(
                self.config.SYMBOL, 
                self.config.INTERVAL, 
                self.config.BACKTEST_LIMIT
            )
            
            if df is None or len(df) < 100:
                logger.error("Insufficient data for backtest")
                return
            
            df = self.ta.apply_indicators(df)
            if df is None:
                logger.error("Failed to apply indicators for backtest")
                return
            
            # Start backtest from a point where indicators are stable
            start_idx = 100
            
            for i in range(start_idx, len(df)):
                sub_df = df.iloc[:i+1]
                
                # Detect market regime
                regime = self.ta.detect_market_regime(sub_df)
                
                # Get signal with scoring system
                signal, signal_desc, score = self.ta.get_signal_score(sub_df)
                
                current_price = sub_df.iloc[-1]['close']
                
                # Call handle_trade instead of execute_trade directly
                self.handle_trade(signal, current_price, regime, sub_df, signal_desc, score)
            
            # Final results
            final_price = df.iloc[-1]['close']
            final_value = self.portfolio.get_total_value(final_price)
            total_return = self.portfolio.get_total_return_pct(final_price)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics()
            
            # Calculate regime statistics
            regime_stats = {}
            for entry in self.ta.regime_history:
                regime = entry['regime']
                if regime not in regime_stats:
                    regime_stats[regime] = 0
                regime_stats[regime] += 1
            
            regime_summary = " | ".join([f"{k}: {v}" for k, v in regime_stats.items()])
            
            results = (f"✅ BACKTEST COMPLETE\n"
                      f"Initial: ${self.config.INITIAL_CASH:.2f}\n"
                      f"Final: ${final_value:.2f}\n"
                      f"Return: {total_return:.2f}%\n"
                      f"Regime Distribution: {regime_summary}\n\n"
                      f"📊 PERFORMANCE METRICS:\n"
                      f"Total Trades: {metrics.get('total_trades', 0)}\n"
                      f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                      f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                      f"Avg Win: ${metrics.get('avg_win', 0):.2f}\n"
                      f"Avg Loss: ${metrics.get('avg_loss', 0):.2f}\n"
                      f"Largest Win: ${metrics.get('largest_win', 0):.2f}\n"
                      f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
            
            logger.info(results)
            self.notifier.send_message(results)
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
    
    def check_trading_cooldown(self) -> bool:
        if not self.is_live or not self.portfolio.last_trade_dt:
            return True
        
        last_trade = None
        if(self.portfolio.last_trade_dt):
            last_trade = self.portfolio.last_trade_dt
        if not last_trade:
            time_since_last_trade = (datetime.datetime.now() - self.portfolio.last_trade_dt).total_seconds()
            if time_since_last_trade < self.config.LIVE_TRADE_COOLDOWN:
                remaining = self.config.LIVE_TRADE_COOLDOWN - time_since_last_trade
                logger.info(f"Trade cooldown active: {remaining:.0f}s remaining")
                return False
        return True
    
    def start_trading(self):
        logger.info(f"Starting {'LIVE' if self.is_live else 'PAPER'} trader...")
        self.is_running = True
        
        # Send startup message
        mode = "LIVE" if self.is_live else "PAPER"
        startup_msg = f"🚀 {mode} TRADING STARTED\n"
        startup_msg += f"Symbol: {self.config.SYMBOL}\n"
        startup_msg += f"Interval: {self.config.INTERVAL}\n"
        if self.is_live:
            startup_msg += f"Max Position: ${self.config.MAX_POSITION_VALUE}\n"
            startup_msg += f"Trade Cooldown: {self.config.LIVE_TRADE_COOLDOWN}s\n"
            startup_msg += f"Max Daily Trades: {self.config.MAX_DAILY_TRADES}"

        self.notifier.send_message(startup_msg)
        
        self.exchange.hyperliquid.set_leverage(1, self.config.SYMBOL)
        # Start Telegram command polling
        self.notifier.start_command_polling()
        
        try:
            while self.is_running:
                try:
                    df = self.ta.get_historical_ohlcv(
                        self.config.SYMBOL, 
                        self.config.INTERVAL, 
                        self.config.HISTORICAL_LIMIT
                    )
                    
                    if df is None or len(df) < 100:
                        logger.warning("Insufficient data, skipping iteration")
                        time.sleep(self.config.SLEEP_INTERVAL)
                        continue
                    
                    df = self.ta.apply_indicators(df)
                    if df is None:
                        logger.warning("Failed to apply indicators, skipping iteration")
                        time.sleep(self.config.SLEEP_INTERVAL)
                        continue
                    
                    # Store current price for status updates
                    self.last_price = df.iloc[-1]['close']
                    
                    # Detect market regime
                    regime = self.ta.detect_market_regime(df)
                    
                    # Get signal with scoring system
                    signal, signal_desc, score = self.ta.get_signal_score(df)
                    
                    current_price = df.iloc[-1]['close']
                    
                    self.handle_trade(signal, current_price, regime, df, signal_desc, score)
                    
                    # Log performance metrics periodically
                    if len(self.trade_history) > 0 and len(self.trade_history) % 10 == 0:
                        metrics = self.calculate_performance_metrics()
                        logger.info(f"Performance Update - Trades: {metrics['total_trades']}, "
                                f"Win Rate: {metrics['win_rate']:.2%}, "
                                f"Profit Factor: {metrics['profit_factor']:.2f}")
                    
                    # Sync position periodically for live trading
                    if self.is_live and len(self.trade_history) % 5 == 0:
                        self.portfolio.sync_with_exchange()
                    
                    time.sleep(self.config.SLEEP_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(self.config.SLEEP_INTERVAL)
                    
        except Exception as e:
            logger.error(f"Paper trader failed: {e}")
        finally:
            self.is_running = False
            
            self.notifier.stop_command_polling()
            
            # Final performance report
            if self.trade_history:
                metrics = self.calculate_performance_metrics()
                final_report = (f"📊 FINAL PERFORMANCE REPORT ({mode}):\n"
                            f"Total Trades: {metrics.get('total_trades', 0)}\n"
                            f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                logger.info(final_report)
                self.notifier.send_message(final_report)
            
            logger.info(f"{mode} trader stopped")
    
    def check_daily_trade_limit(self) -> bool:
        if not self.is_live:
            return True
            
        current_date = datetime.datetime.now().date()
        if self.portfolio.last_trade_dt and self.portfolio.last_trade_dt.date() != current_date:
            self.portfolio.daily_trade_count = 0
            self.portfolio.last_trade_dt = datetime.datetime.now()
            
        if self.portfolio.daily_trade_count >= self.config.MAX_DAILY_TRADES:
            logger.warning(f"Daily trade limit reached: {self.portfolio.daily_trade_count}/{self.config.MAX_DAILY_TRADES}")
            return False
        return True
    
    def stop(self):
        self.is_running = False
        self.close_position(self.last_price, "🛑 BOT STOPPED")
        logger.info("Stop signal sent to trading bot")

    def start(self):
        """Start the trading bot"""
        if not self.is_running:
            self.is_running = True
            logger.info("Trading bot started")
            self.start_trading()
        else:
            logger.warning("Trading bot is already running")